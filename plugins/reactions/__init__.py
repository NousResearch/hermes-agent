"""Slack reaction lifecycle plugin.

Migrated from the OLD jleechanorg fork's `gateway/platforms/slack.py` reaction
state machine (lines 1305-1331, 1379-1400, 2216-2221, 2318). The OLD code
lived in `gateway/platforms/slack.py` which was dropped per the user's
"I dont want slack.py I want to follow upstream unless absolutely necessary"
policy. The reaction mechanism is "absolutely necessary" because it's
user-facing UX (Slack users expect to see 👀 when the bot starts processing).
Upstream NousResearch/hermes-agent has NO equivalent (only `send_typing` for
Assistant contexts).

State machine:
  pre_gateway_dispatch  → place 👀 eyes on the incoming message
  on_session_end        → remove 👀; place ✅ (success) or ❌ (failure/cancelled)

Gating (matches OLD fork behavior):
  - Only on Slack (other platforms: no-op)
  - Only on DMs or @-mentions by default (configurable via
    `reactions_require_dm_or_mention`)
  - Only on user-originated messages (bot_id + subtype filtered)

Defensive: every entry point wrapped in try/except. Plugin can never crash
the gateway dispatch loop or session-end path.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State tracking — keyed by incoming message_id so we know which message's
# reaction to clean up at session end.
# ---------------------------------------------------------------------------

# Map: incoming_msg_id -> (channel_id, message_ts)
_pending_reactions: Dict[str, Tuple[str, str]] = {}
# Set: incoming_msg_ids we've already placed an eyes reaction on (so we don't
# double-place if the hook fires twice for the same message).
_eyes_placed: Set[str] = set()
_state_lock = threading.Lock()

DEFAULT_EYES = "eyes"
DEFAULT_SUCCESS = "white_check_mark"
DEFAULT_FAILURE = "x"


# ---------------------------------------------------------------------------
# Config access (read-only)
# ---------------------------------------------------------------------------

def _read_config() -> Tuple[bool, bool, str, str, str]:
    """Return (enabled, require_dm_or_mention, eyes, success, failure).

    Env vars override config for smoke tests and emergency rollout. Runtime
    defaults come from ``~/.hermes/config.yaml`` so ``plugins.enabled`` and
    ``platforms.slack.extra`` control the live gateway without plist churn.
    """
    enabled = True
    require_dm_or_mention = True
    eyes = os.getenv("HERMES_REACTIONS_EYES", DEFAULT_EYES)
    success = os.getenv("HERMES_REACTIONS_SUCCESS", DEFAULT_SUCCESS)
    failure = os.getenv("HERMES_REACTIONS_FAILURE", DEFAULT_FAILURE)

    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        plugins_enabled = config.get("plugins", {}).get("enabled") or []
        if plugins_enabled:
            enabled = "reactions" in plugins_enabled
        extra = config.get("platforms", {}).get("slack", {}).get("extra", {}) or {}
        if "reactions_enabled" in extra:
            enabled = bool(extra.get("reactions_enabled"))
        if "reactions_require_dm_or_mention" in extra:
            require_dm_or_mention = bool(extra.get("reactions_require_dm_or_mention"))
        eyes = str(extra.get("reactions_eyes", eyes))
        success = str(extra.get("reactions_success", success))
        failure = str(extra.get("reactions_failure", failure))
    except Exception as exc:
        logger.debug("reactions: config read failed (%s); falling back to env/defaults", exc)

    if "HERMES_REACTIONS" in os.environ:
        enabled = os.getenv("HERMES_REACTIONS", "true").lower() in ("1", "true", "yes")
    if "HERMES_REACTIONS_REQUIRE_DM_OR_MENTION" in os.environ:
        require_dm_or_mention = os.getenv(
            "HERMES_REACTIONS_REQUIRE_DM_OR_MENTION", "true"
        ).lower() in ("1", "true", "yes")
    return enabled, require_dm_or_mention, eyes, success, failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_key(event: Any) -> str:
    """Stable key for tracking the reaction placed on `event`."""
    msg_id = getattr(event, "message_id", None) or getattr(
        getattr(event, "source", None), "message_id", None
    )
    if msg_id:
        return str(msg_id)
    return f"evt:{id(event)}"


def _resolve_slack_adapter(gateway: Any) -> Any:
    """Return the Slack adapter from `gateway.adapters` or None.

    Uses `getattr` defensively so the plugin doesn't crash on a
    partially-initialised gateway (e.g. mid-bootstrap tests).
    """
    try:
        from gateway.config import Platform  # type: ignore[import-not-found]
    except Exception:
        Platform = None  # type: ignore[assignment]

    adapters = getattr(gateway, "adapters", None)
    if not adapters:
        return None

    if hasattr(adapters, "get"):
        keys = ["slack", "Slack", "SLACK"]
        if Platform is not None:
            keys.insert(0, Platform.SLACK)
        for key in keys:
            try:
                adapter = adapters.get(key)
                if adapter is not None:
                    return adapter
            except Exception:
                continue

    if hasattr(adapters, "items"):
        for key, adapter in adapters.items():
            key_name = getattr(key, "value", str(key)).lower()
            cls_name = type(adapter).__name__.lower()
            if key_name == "slack" or "slack" in cls_name:
                return adapter
    return None


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_user_originated(event: Any) -> bool:
    """Skip reactions on bot messages, message_changed subtypes, etc."""
    raw_message = _field(event, "raw_message", None) or {}
    if (
        _field(event, "bot_id", None)
        or _field(raw_message, "bot_id", None)
        or _field(event, "subtype", None) == "bot_message"
        or _field(raw_message, "subtype", None) == "bot_message"
    ):
        return False

    source = _field(event, "source", None)
    user_id = (
        _field(source, "user_id", None)
        or _field(raw_message, "user", None)
        or _field(event, "user", "")
        or ""
    )
    # Slack bot user IDs start with B; human/user IDs are not bot-originated.
    return bool(user_id) and not str(user_id).startswith("B")


def _is_dm_or_mention(event: Any, adapter: Any) -> bool:
    """True if this event is a DM or @-mentions the bot.

    Reads from the event's ``source`` SessionSource (per the immediate_ack
    plugin pattern that works end-to-end) — NOT from the event's top-level
    fields, which don't exist on the gateway's MessageEvent.
    """
    if isinstance(event, dict):
        text = event.get("text", "") or ""
        source = event.get("source") or {}
        channel_type = source.get("chat_type", "") if isinstance(source, dict) else ""
    else:
        text = getattr(event, "text", "") or ""
        source = getattr(event, "source", None)
        channel_type = getattr(source, "chat_type", "") if source else ""
    # DM channels have chat_type == "im" in older payloads and "dm" upstream.
    if channel_type in ("im", "dm"):
        return True
    # @mention: text contains <@bot_uid> or bot's user id
    bot_uid = getattr(adapter, "_bot_user_id", "") or ""
    if bot_uid and (f"<@{bot_uid}>" in text or bot_uid in text):
        return True
    return False


def _run_coro_in_fresh_loop(coro: Any) -> Any:
    """Run an async coroutine to completion from a sync context that may be
    inside a running event loop.

    The plugin hook is invoked synchronously by PluginManager.invoke_hook
    (which does not await). To run a Slack adapter call (which is
    ``async def``) we need a real event loop. BUT — the gateway's main
    loop is already running by the time the hook fires, so a naive
    ``loop.run_until_complete(coro)`` fails with
    "Cannot run the event loop while another loop is running".

    Strategy: run the coroutine in a SEPARATE THREAD with its own event
    loop. The Slack adapter's HTTP calls (chat.postMessage, etc.) are
    independent and don't share state with the gateway's main loop, so
    a separate thread is safe.
    """
    import asyncio
    import concurrent.futures

    def _runner() -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_runner)
        return future.result()


# ---------------------------------------------------------------------------
# Hook: pre_gateway_dispatch
# ---------------------------------------------------------------------------

def _on_pre_gateway_dispatch_sync(
    event: Any,
    gateway: Any,
    session_store: Any,
) -> Optional[Dict[str, Any]]:
    """Place 👀 eyes reaction on the inbound user message."""
    try:
        enabled, require_dm_or_mention, eyes_emoji, _, _ = _read_config()
        if not enabled:
            return None
        if not _is_user_originated(event):
            return None
        adapter = _resolve_slack_adapter(gateway)
        if adapter is None:
            return None
        if require_dm_or_mention and not _is_dm_or_mention(event, adapter):
            return None

        # Resolve channel_id + ts from the event (per the immediate_ack plugin
        # pattern that just worked end-to-end). The MessageEvent has
        # `source.chat_id` and either `message_id` (top-level) or `thread_id`
        # (when the gateway sets it). For Slack top-level messages the gateway
        # sets `source.thread_id = source.message_id` as a synthetic-thread
        # fallback so reactions thread under the user's own message.
        source = getattr(event, "source", None)
        if source is None:
            return None
        chat_id = getattr(source, "chat_id", None)
        if not chat_id:
            return None
        ts = (
            getattr(event, "message_id", None)
            or getattr(source, "message_id", None)
            or getattr(source, "thread_id", None)
        )
        if not ts:
            return None

        # Place the reaction. The Slack adapter's `_add_reaction` is private
        # (per F1v2 analysis); fall back to a raw reactions.add via the
        # adapter's underlying client when needed.
        add_reaction = getattr(adapter, "_add_reaction", None) or getattr(
            adapter, "add_reaction", None
        )
        if add_reaction is None:
            return None
        try:
            _run_coro_in_fresh_loop(
                add_reaction(channel=chat_id, timestamp=ts, emoji=eyes_emoji)
            )
        except Exception:
            # Try alternate signature (name= instead of emoji=)
            try:
                _run_coro_in_fresh_loop(
                    add_reaction(channel=chat_id, timestamp=ts, name=eyes_emoji)
                )
            except Exception as exc:
                logger.debug("reactions: eyes add failed: %s", exc)
                return None

        key = _tracker_key(event)
        with _state_lock:
            _pending_reactions[key] = (chat_id, ts)
            _eyes_placed.add(key)
        logger.debug(
            "reactions: eyes=%s placed on ts=%s chat=%s key=%s",
            eyes_emoji, ts, chat_id, key,
        )
    except Exception as exc:
        logger.debug("reactions: pre_gateway_dispatch failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Hook: on_session_end
# ---------------------------------------------------------------------------

def _on_session_end_sync(**kwargs: Any) -> None:
    """Swap 👀 for ✅ (success) or ❌ (failure/cancelled) at session end."""
    try:
        enabled, _, eyes_emoji, success_emoji, failure_emoji = _read_config()
        if not enabled:
            return
        session_id = kwargs.get("session_id") or kwargs.get("session_key") or ""
        outcome = (kwargs.get("outcome") or kwargs.get("status") or "success")
        outcome_str = str(outcome).lower()
        is_failure = outcome_str in ("failure", "failed", "error", "cancelled", "timeout")
        target_emoji = failure_emoji if is_failure else success_emoji

        # Find the pending reaction for this session
        with _state_lock:
            target_key = None
            for key, value in list(_pending_reactions.items()):
                if session_id and key in session_id:
                    target_key = key
                    break
            if target_key is None and _pending_reactions:
                target_key = next(iter(_pending_reactions), None)
            if target_key is None:
                return
            pending = _pending_reactions.pop(target_key)
            eyes_were_placed = target_key in _eyes_placed
            _eyes_placed.discard(target_key)

        channel, ts = pending

        # Need the adapter again. The on_session_end hook doesn't pass the
        # gateway; resolve via the runtime module global (best-effort).
        adapter = _find_runtime_adapter()
        if adapter is None:
            return

        # Remove the eyes reaction (best-effort)
        if eyes_were_placed:
            remove_reaction = getattr(adapter, "_remove_reaction", None) or getattr(
                adapter, "remove_reaction", None
            )
            if remove_reaction is not None:
                try:
                    _run_coro_in_fresh_loop(
                        remove_reaction(channel=channel, timestamp=ts, emoji=eyes_emoji)
                    )
                except Exception:
                    try:
                        _run_coro_in_fresh_loop(
                            remove_reaction(channel=channel, timestamp=ts, name=eyes_emoji)
                        )
                    except Exception:
                        pass

        # Place the success/failure reaction
        add_reaction = getattr(adapter, "_add_reaction", None) or getattr(
            adapter, "add_reaction", None
        )
        if add_reaction is not None:
            try:
                _run_coro_in_fresh_loop(
                    add_reaction(channel=channel, timestamp=ts, emoji=target_emoji)
                )
            except Exception:
                try:
                    _run_coro_in_fresh_loop(
                        add_reaction(channel=channel, timestamp=ts, name=target_emoji)
                    )
                except Exception as exc:
                    logger.debug("reactions: %s add failed: %s", target_emoji, exc)
                    return
        logger.debug(
            "reactions: marker %s placed on ts=%s chat=%s outcome=%s",
            target_emoji, ts, channel, outcome_str,
        )
    except Exception as exc:
        logger.debug("reactions: on_session_end failed: %s", exc)


def _find_runtime_adapter() -> Any:
    """Best-effort: locate the live Slack adapter from the gateway runtime.

    The plugin manager singleton holds hooks but not a gateway ref. Try
    the well-known global module references the gateway publishes at
    startup.
    """
    try:
        import sys
        candidates = [
            ("gateway.run", "gateway_runner"),
            ("gateway.run", "runner"),
            ("gateway.run", "_active_runner"),
            ("gateway.run", "_gateway_runner_ref"),
        ]
        for mod_name, attr in candidates:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            runner = getattr(mod, attr, None)
            if runner is None:
                continue
            if callable(runner) and attr.endswith("_ref"):
                runner = runner()
                if runner is None:
                    continue
            adapter = _resolve_slack_adapter(runner)
            if adapter is not None:
                return adapter
    except Exception as exc:
        logger.debug("reactions: runtime adapter lookup failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Plugin registration (called by hermes_cli.plugins)
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register this plugin's hooks with the host (SYNC implementations)."""
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch_sync)
    ctx.register_hook("on_session_end", _on_session_end_sync)
    logger.debug("reactions: registered hooks (sync, 👀→✅/❌)")
