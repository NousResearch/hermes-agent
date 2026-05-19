"""Shared helper classes for gateway platform adapters.

Extracts common patterns that were duplicated across 5-7 adapters:
message deduplication, text batch aggregation, markdown stripping,
thread participation tracking, and the webhook-style render→deliver
pipeline shared by the ``webhook`` and ``svix`` adapters.
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from utils import atomic_json_write

if TYPE_CHECKING:
    from gateway.platforms.base import MessageEvent, SendResult

logger = logging.getLogger(__name__)


# ─── Message Deduplication ────────────────────────────────────────────────────


class MessageDeduplicator:
    """TTL-based message deduplication cache.

    Replaces the identical ``_seen_messages`` / ``_is_duplicate()`` pattern
    previously duplicated in discord, slack, dingtalk, wecom, weixin,
    mattermost, and feishu adapters.

    Usage::

        self._dedup = MessageDeduplicator()

        # In message handler:
        if self._dedup.is_duplicate(msg_id):
            return
    """

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._seen: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window."""
        if not msg_id:
            return False
        now = time.time()
        if msg_id in self._seen:
            if now - self._seen[msg_id] < self._ttl:
                return True
            # Entry has expired — remove it and treat as new
            del self._seen[msg_id]
        self._seen[msg_id] = now
        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if len(self._seen) > self._max_size:
                # TTL pruning alone does not cap the cache when every entry is
                # still fresh. Keep the newest entries so the helper's
                # max_size bound is enforced under sustained traffic.
                newest = sorted(
                    self._seen.items(),
                    key=lambda item: item[1],
                )[-self._max_size:]
                self._seen = dict(newest)
        return False

    def clear(self):
        """Clear all tracked messages."""
        self._seen.clear()


# ─── Text Batch Aggregation ──────────────────────────────────────────────────


class TextBatchAggregator:
    """Aggregates rapid-fire text events into single messages.

    Replaces the ``_enqueue_text_event`` / ``_flush_text_batch`` pattern
    previously duplicated in telegram, discord, matrix, wecom, and feishu.

    Usage::

        self._text_batcher = TextBatchAggregator(
            handler=self._message_handler,
            batch_delay=0.6,
            split_threshold=1900,
        )

        # In message dispatch:
        if msg_type == MessageType.TEXT and self._text_batcher.is_enabled():
            self._text_batcher.enqueue(event, session_key)
            return
    """

    def __init__(
        self,
        handler,
        *,
        batch_delay: float = 0.6,
        split_delay: float = 2.0,
        split_threshold: int = 4000,
    ):
        self._handler = handler
        self._batch_delay = batch_delay
        self._split_delay = split_delay
        self._split_threshold = split_threshold
        self._pending: Dict[str, "MessageEvent"] = {}
        self._pending_tasks: Dict[str, asyncio.Task] = {}

    def is_enabled(self) -> bool:
        """Return True if batching is active (delay > 0)."""
        return self._batch_delay > 0

    def enqueue(self, event: "MessageEvent", key: str) -> None:
        """Add *event* to the pending batch for *key*."""
        chunk_len = len(event.text or "")
        existing = self._pending.get(key)
        if not existing:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending[key] = event
        else:
            existing.text = f"{existing.text}\n{event.text}"
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]

        # Cancel prior flush timer, start a new one
        prior = self._pending_tasks.get(key)
        if prior and not prior.done():
            prior.cancel()
        self._pending_tasks[key] = asyncio.create_task(self._flush(key))

    async def _flush(self, key: str) -> None:
        """Wait then dispatch the batched event for *key*."""
        current_task = self._pending_tasks.get(key)
        pending = self._pending.get(key)
        last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0

        # Use longer delay when the last chunk looks like a split message
        delay = self._split_delay if last_len >= self._split_threshold else self._batch_delay
        await asyncio.sleep(delay)

        event = self._pending.pop(key, None)
        if event:
            try:
                await self._handler(event)
            except Exception:
                logger.exception("[TextBatchAggregator] Error dispatching batched event for %s", key)

        if self._pending_tasks.get(key) is current_task:
            self._pending_tasks.pop(key, None)

    def cancel_all(self) -> None:
        """Cancel all pending flush tasks."""
        for task in self._pending_tasks.values():
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        self._pending.clear()


# ─── Markdown Stripping ──────────────────────────────────────────────────────

# Pre-compiled regexes for performance
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_RE_ITALIC_STAR = re.compile(r"\*(.+?)\*", re.DOTALL)
_RE_BOLD_UNDER = re.compile(r"\b__(?![\s_])(.+?)(?<![\s_])__\b", re.DOTALL)
_RE_ITALIC_UNDER = re.compile(r"\b_(?![\s_])(.+?)(?<![\s_])_\b", re.DOTALL)
_RE_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n?")
_RE_INLINE_CODE = re.compile(r"`(.+?)`")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text platforms (SMS, iMessage, etc.).

    Replaces the identical ``_strip_markdown()`` functions previously
    duplicated in sms.py, bluebubbles.py, and feishu.py.
    """
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_ITALIC_UNDER.sub(r"\1", text)
    text = _RE_CODE_BLOCK.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─── Thread Participation Tracking ───────────────────────────────────────────


class ThreadParticipationTracker:
    """Persistent tracking of threads the bot has participated in.

    Replaces the identical ``_load/_save_participated_threads`` +
    ``_mark_thread_participated`` pattern previously duplicated in
    discord.py and matrix.py.

    Usage::

        self._threads = ThreadParticipationTracker("discord")

        # Check membership:
        if thread_id in self._threads:
            ...

        # Mark participation:
        self._threads.mark(thread_id)
    """

    _MAX_TRACKED = 500

    def __init__(self, platform_name: str, max_tracked: int = 500):
        self._platform = platform_name
        self._max_tracked = max_tracked
        self._threads: dict[str, None] = {
            str(thread_id): None for thread_id in self._load()
        }

    def _state_path(self) -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / f"{self._platform}_threads.json"

    def _load(self) -> list[str]:
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(thread_id) for thread_id in data]
            except Exception:
                pass
        return []

    def _save(self) -> None:
        path = self._state_path()
        thread_list = list(self._threads)
        if len(thread_list) > self._max_tracked:
            thread_list = thread_list[-self._max_tracked:]
            self._threads = dict.fromkeys(thread_list)
        atomic_json_write(path, thread_list, indent=None)

    def mark(self, thread_id: str) -> None:
        """Mark *thread_id* as participated and persist."""
        if thread_id not in self._threads:
            self._threads[thread_id] = None
            self._save()

    def __contains__(self, thread_id: str) -> bool:
        return thread_id in self._threads

    def clear(self) -> None:
        self._threads.clear()


# ─── Phone Number Redaction ──────────────────────────────────────────────────


def redact_phone(phone: str) -> str:
    """Redact a phone number for logging, preserving country code and last 4.

    Replaces the identical ``_redact_phone()`` functions in signal.py,
    sms.py, and bluebubbles.py.
    """
    if not phone:
        return "<none>"
    if len(phone) <= 8:
        return phone[:2] + "****" + phone[-2:] if len(phone) > 4 else "****"
    return phone[:4] + "****" + phone[-4:]


# ─── Webhook-style render → deliver pipeline ──────────────────────────────────

# Built-in platform names that can be a ``deliver`` target. Plugin-registered
# platforms are resolved at send time via the platform registry, so this set is
# just the fast path / offline fallback.
_BUILTIN_DELIVER_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "signal", "sms", "whatsapp",
    "matrix", "mattermost", "homeassistant", "email", "dingtalk",
    "feishu", "wecom", "wecom_callback", "weixin", "bluebubbles",
    "qqbot", "yuanbao",
})

# Matches the ``{a.b.c}`` / ``{__raw__}`` placeholders used by prompt and
# deliver_extra templates.
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z0-9_.]+\}")


class WebhookDeliveryMixin:
    """Shared render → deliver pipeline for the webhook & svix adapters.

    Both adapters ingest an event (an HTTP POST for ``webhook``, a polled
    Svix message for ``svix``), render a prompt/delivery template from the
    payload, optionally inject a skill, then route the agent's response to a
    ``deliver`` target — ``log``, ``github_comment``, or any connected
    platform via the gateway runner. That tail end was duplicated almost
    verbatim; this mixin owns it.

    Concrete adapters must set the following on ``self`` (in ``__init__``):
      - ``_log_tag``: log prefix, e.g. ``"[webhook]"`` / ``"[svix]"``
      - ``_event_noun``: human noun for the no-template fallback, e.g.
        ``"Webhook"`` / ``"Svix"``
      - ``_delivery_info``: ``Dict[str, dict]`` keyed by session chat_id
      - ``_delivery_info_created``: ``Dict[str, float]`` of creation times
      - ``_delivery_info_ttl``: seconds before a delivery_info entry is pruned
      - ``gateway_runner``: set by ``GatewayRunner._create_adapter`` (may be
        ``None`` before the gateway wires it up)

    and inherit ``build_source`` / ``handle_message`` from
    ``BasePlatformAdapter``. Mix in *before* ``BasePlatformAdapter`` so this
    ``send()`` satisfies the abstract method.
    """

    # Attribute type hints for the contract above (set by the concrete adapter).
    _log_tag: str
    _event_noun: str
    _delivery_info: Dict[str, dict]
    _delivery_info_created: Dict[str, float]
    _delivery_info_ttl: float

    # ── Template rendering ────────────────────────────────────────────────

    def _render_prompt(
        self,
        template: str,
        payload: dict,
        event_type: str,
        route_name: str,
    ) -> str:
        """Render a prompt template against the event payload.

        Supports dot-notation access into nested dicts
        (``{pull_request.title}`` → ``payload["pull_request"]["title"]``),
        the ``{__raw__}`` token (entire payload as indented JSON, capped at
        4000 chars), and ``{__event__}`` (the event type). Missing keys are
        left verbatim so a template typo is visible rather than silently
        blanked. An empty template falls back to a JSON dump with event/route
        context.
        """
        if not template:
            truncated = json.dumps(payload, indent=2)[:4000]
            return (
                f"{self._event_noun} event '{event_type}' on route "
                f"'{route_name}':\n\n```json\n{truncated}\n```"
            )

        def _resolve(match: "re.Match") -> str:
            key = match.group(1)
            if key == "__raw__":
                return json.dumps(payload, indent=2)[:4000]
            if key == "__event__":
                return event_type
            value: Any = payload
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part, f"{{{key}}}")
                else:
                    return f"{{{key}}}"
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2)[:2000]
            return str(value)

        return re.sub(r"\{([a-zA-Z0-9_.]+)\}", _resolve, template)

    def _render_delivery_extra(self, extra: dict, payload: dict) -> dict:
        """Render string values in ``deliver_extra`` with payload data.

        Non-string values pass through untouched. Because these values decide
        *where* the response goes (chat_id, repo, pr_number, …), an
        unresolved placeholder left behind by a missing payload path is logged
        loudly — silently mis-routing delivery is worse than a noisy warning.
        """
        rendered: Dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, str):
                out = self._render_prompt(value, payload, "", "")
                unresolved = [
                    tok for tok in _PLACEHOLDER_RE.findall(value) if tok in out
                ]
                if unresolved:
                    logger.warning(
                        "%s deliver_extra.%s still contains unresolved "
                        "placeholder(s) %s after rendering (payload path "
                        "missing?); delivery may be mis-routed. Rendered "
                        "value: %r",
                        self._log_tag, key, unresolved, out,
                    )
                rendered[key] = out
            else:
                rendered[key] = value
        return rendered

    def _inject_skill(self, prompt: str, skills) -> str:
        """Wrap *prompt* in the first configured skill's invocation message.

        Calls ``build_skill_invocation_message`` directly rather than emitting
        a ``/skill`` slash command, which the gateway command parser would
        intercept. Returns *prompt* unchanged when no skill is configured or
        loadable.
        """
        if not skills:
            return prompt
        try:
            from agent.skill_commands import (
                build_skill_invocation_message,
                get_skill_commands,
            )
            skill_cmds = get_skill_commands()
            for skill_name in skills:
                cmd_key = f"/{skill_name}"
                if cmd_key in skill_cmds:
                    skill_content = build_skill_invocation_message(
                        cmd_key, user_instruction=prompt
                    )
                    if skill_content:
                        return skill_content
                else:
                    logger.warning(
                        "%s Skill '%s' not found", self._log_tag, skill_name
                    )
        except Exception as exc:
            logger.warning("%s Skill loading failed: %s", self._log_tag, exc)
        return prompt

    # ── delivery_info bookkeeping ─────────────────────────────────────────

    def _prune_delivery_info(self, now: float) -> None:
        """Drop delivery_info entries older than ``_delivery_info_ttl``.

        Keeps the dict bounded even if many events fire and never receive a
        final response. Called on each new event.
        """
        cutoff = now - self._delivery_info_ttl
        stale = [
            k for k, t in self._delivery_info_created.items() if t < cutoff
        ]
        for k in stale:
            self._delivery_info.pop(k, None)
            self._delivery_info_created.pop(k, None)

    # ── Response delivery ─────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SendResult":
        """Deliver the agent's response to the destination for ``chat_id``.

        The delivery info stored when the event was received is read with
        ``.get()`` (never popped) so interim status messages — fallback-model
        notices, context-pressure warnings — don't consume the entry and
        silently downgrade the final response to the ``log`` deliver type.
        TTL cleanup happens when the next event arrives.
        """
        from gateway.platforms.base import SendResult

        delivery = self._delivery_info.get(chat_id, {})
        deliver_type = delivery.get("deliver", "log")

        if deliver_type == "log":
            logger.info(
                "%s Response for %s: %s", self._log_tag, chat_id, content[:200]
            )
            return SendResult(success=True)

        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)

        # Any platform with a connected gateway adapter — built-in name or a
        # plugin-registered platform.
        _is_known_platform = deliver_type in _BUILTIN_DELIVER_PLATFORMS
        if not _is_known_platform:
            try:
                from gateway.platform_registry import platform_registry
                _is_known_platform = platform_registry.is_registered(deliver_type)
            except Exception:
                pass
        if self.gateway_runner and _is_known_platform:
            return await self._deliver_cross_platform(
                deliver_type, content, delivery
            )

        logger.warning("%s Unknown deliver type: %s", self._log_tag, deliver_type)
        return SendResult(
            success=False, error=f"Unknown deliver type: {deliver_type}"
        )

    async def _direct_deliver(self, content: str, delivery: dict) -> "SendResult":
        """Deliver *content* without invoking the agent (``deliver_only``).

        The rendered template becomes the literal message body, dispatched
        through the same target helpers as the agent-mode ``send()`` flow.
        """
        from gateway.platforms.base import SendResult

        deliver_type = delivery.get("deliver", "log")
        if deliver_type == "log":
            # Startup validation rejects deliver_only + deliver=log; guard anyway.
            logger.info(
                "%s direct-deliver log-only: %s", self._log_tag, content[:200]
            )
            return SendResult(success=True)
        if deliver_type == "github_comment":
            return await self._deliver_github_comment(content, delivery)
        return await self._deliver_cross_platform(deliver_type, content, delivery)

    async def _deliver_github_comment(
        self, content: str, delivery: dict
    ) -> "SendResult":
        """Post *content* as a GitHub PR/issue comment via the ``gh`` CLI."""
        from gateway.platforms.base import SendResult

        extra = delivery.get("deliver_extra", {})
        repo = extra.get("repo", "")
        pr_number = extra.get("pr_number", "")
        if not repo or not pr_number:
            logger.error(
                "%s github_comment delivery missing repo or pr_number",
                self._log_tag,
            )
            return SendResult(success=False, error="Missing repo or pr_number")
        try:
            result = subprocess.run(
                [
                    "gh", "pr", "comment", str(pr_number),
                    "--repo", repo, "--body", content,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info(
                    "%s Posted comment on %s#%s", self._log_tag, repo, pr_number
                )
                return SendResult(success=True)
            logger.error(
                "%s gh pr comment failed: %s", self._log_tag, result.stderr
            )
            return SendResult(success=False, error=result.stderr)
        except FileNotFoundError:
            logger.error(
                "%s 'gh' CLI not found — install GitHub CLI for "
                "github_comment delivery", self._log_tag,
            )
            return SendResult(success=False, error="gh CLI not installed")
        except Exception as exc:
            logger.error(
                "%s github_comment delivery error: %s", self._log_tag, exc
            )
            return SendResult(success=False, error=str(exc))

    async def _deliver_cross_platform(
        self, platform_name: str, content: str, delivery: dict
    ) -> "SendResult":
        """Route *content* to another connected platform via the gateway."""
        from gateway.config import Platform
        from gateway.platforms.base import SendResult

        if not self.gateway_runner:
            return SendResult(
                success=False,
                error="No gateway runner for cross-platform delivery",
            )
        try:
            target_platform = Platform(platform_name)
        except ValueError:
            return SendResult(
                success=False, error=f"Unknown platform: {platform_name}"
            )
        adapter = self.gateway_runner.adapters.get(target_platform)
        if not adapter:
            return SendResult(
                success=False, error=f"Platform {platform_name} not connected"
            )
        extra = delivery.get("deliver_extra", {})
        chat_id = extra.get("chat_id", "")
        if not chat_id:
            home = self.gateway_runner.config.get_home_channel(target_platform)
            if home:
                chat_id = home.chat_id
            else:
                return SendResult(
                    success=False,
                    error=f"No chat_id or home channel for {platform_name}",
                )
        # Pass thread_id from deliver_extra so Telegram forum topics work.
        metadata = None
        thread_id = extra.get("message_thread_id") or extra.get("thread_id")
        if thread_id:
            metadata = {"thread_id": thread_id}
        return await adapter.send(chat_id, content, metadata=metadata)
