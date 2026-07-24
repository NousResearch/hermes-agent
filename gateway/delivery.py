"""
Delivery routing for cron job outputs and agent responses.

Routes messages to the appropriate destination based on:
- Explicit targets (e.g., "telegram:123456789")
- Platform home channels (e.g., "telegram" → home channel)
- Origin (back to where the job was created)
- Local (always saved to files)
"""

import logging
import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

# Cap before gateway-level truncation of cron output for non-chunking platform
# delivery.  Telegram's hard API limit is 4096; the headroom covers the "full
# output saved to …" footer appended on truncation.  Adapters that split long
# messages natively (BasePlatformAdapter.splits_long_messages) bypass this
# entirely — the adapter chunks in its own send() and the full output is
# preserved.
MAX_PLATFORM_OUTPUT = 4000

# Matches strings that are *only* a "silence" narration with optional markdown
# wrappers. Covers: *(silent)*, _silent_, `silent`, ~silent~, (silent), silent,
# 🔇, a bare ".", "…", and the whitespace/marker-padded variants seen in the
# wild. Anchored to start/end so substantive messages that merely *contain* the
# word "silent" are never matched.
_SILENCE_NARRATION = re.compile(
    r'^[\s*_~`]*\(?\s*(silent|silence|no\s+response|no\s+reply)\s*\.?\)?[\s*_~`]*$'
    r'|^[\s*_~`]*[\U0001F507\.\u2026]+[\s*_~`]*$',
    re.IGNORECASE,
)


def _is_silence_narration(content: Optional[str]) -> bool:
    """Return True when ``content`` is *only* a silence-narration token.

    Length-guarded (real messages are longer) and anchored to the whole string
    so legitimate prose like "The deployment ran silently" or "Silence is
    golden — here is the plan..." is never flagged.
    """
    if not content:
        return False
    stripped = content.strip()
    if not stripped or len(stripped) > 64:  # length guard
        return False
    return bool(_SILENCE_NARRATION.match(stripped))

from .config import Platform, GatewayConfig, PlatformConfig
from .session import SessionSource
from .dead_targets import DeadTargetRegistry


@dataclass(frozen=True)
class DeliveryTransport:
    """Resolved live transport for one logical delivery platform."""

    adapter: Any
    config: Optional[PlatformConfig]
    transport_platform: Platform

    @property
    def is_relay(self) -> bool:
        return self.transport_platform == Platform.RELAY

    async def send(
        self,
        logical_platform: Platform,
        chat_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Any:
        """Send through this transport while preserving the logical platform."""
        if self.is_relay:
            return await self.adapter.send_for_platform(
                logical_platform,
                chat_id,
                content,
                metadata=metadata,
            )
        return await self.adapter.send(chat_id, content, metadata=metadata)


def resolve_delivery_transport(
    platform: Platform,
    config: GatewayConfig,
    adapters: Optional[Dict[Platform, Any]],
) -> Optional[DeliveryTransport]:
    """Resolve a logical platform to its live delivery transport.

    A concrete native adapter always wins. Relay is eligible only when its
    authenticated transport explicitly advertises that it fronts the logical
    platform, which keeps restart-time delivery independent of per-chat caches
    without letting Relay hijack unrelated platform targets.
    """
    live_adapters = adapters or {}
    native = live_adapters.get(platform)
    native_config = config.platforms.get(platform)
    # Preserve DeliveryRouter's historical support for explicitly supplied live
    # adapters with no config block, but never let an explicitly disabled native
    # adapter shadow an enabled Relay transport.
    if native is not None and (native_config is None or native_config.enabled):
        return DeliveryTransport(
            adapter=native,
            config=native_config,
            transport_platform=platform,
        )

    relay = live_adapters.get(Platform.RELAY)
    relay_config = config.platforms.get(Platform.RELAY)
    fronts_platform = getattr(relay, "fronts_platform", None)
    if (
        relay is not None
        and (relay_config is None or relay_config.enabled)
        and callable(fronts_platform)
        and fronts_platform(platform)
    ):
        return DeliveryTransport(
            adapter=relay,
            config=relay_config,
            transport_platform=Platform.RELAY,
        )
    return None


def looks_like_telegram_private_chat_id(chat_id: Optional[str]) -> bool:
    """True when ``chat_id`` is a positive int — Telegram's private-chat shape.

    Telegram private chats use positive chat IDs; groups/channels/supergroups
    use negative IDs. This is the single source of truth for that heuristic,
    reused by the handoff seed path in ``gateway/run.py`` so handoff-created
    DM topics key the same way as inbound DM-topic messages.
    """
    if chat_id is None:
        return False
    try:
        return int(chat_id) > 0
    except (TypeError, ValueError):
        return False


def _looks_like_int(value: Optional[str]) -> bool:
    if value is None:
        return False
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _send_result_failed(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("success") is False
    return getattr(result, "success", True) is False


def _send_result_error(result: Any) -> Optional[str]:
    if isinstance(result, dict):
        error = result.get("error")
    else:
        error = getattr(result, "error", None)
    return str(error) if error else None


def _is_thread_not_found_delivery_error(result: Any) -> bool:
    error = _send_result_error(result)
    return bool(error and "thread not found" in error.lower())


def _send_result_error_kind(result: Any) -> Optional[str]:
    """Return the machine-readable error_kind from a SendResult/dict, if any."""
    if isinstance(result, dict):
        kind = result.get("error_kind")
    else:
        kind = getattr(result, "error_kind", None)
    return str(kind) if kind else None


def _classify_dead_from_error_text(error_text: Optional[str]) -> Optional[str]:
    """Best-effort dead-target classification from a raised error's text.

    ``_deliver_to_platform`` raises (it does not return a SendResult) on a hard
    failure, so the ``deliver()`` loop only has the exception string.  Reuse the
    platform-neutral classifier to recover the error_kind from that text.
    """
    if not error_text:
        return None
    try:
        from .platforms.base import classify_send_error, is_chat_level_not_found
    except Exception:  # pragma: no cover - import guard
        return None
    kind = classify_send_error(None, error_text=error_text)
    if not DeadTargetRegistry.is_dead_error_kind(kind):
        return None
    # ``not_found`` collapses chat-level and thread/topic/message-level failures.
    # Only a whole-chat not_found means the target is dead — a deleted forum topic
    # or an edited-away message must not mark the entire chat (and all of its future
    # deliveries) dead.  See gateway.dead_targets' documented scope.
    if kind == "not_found" and not is_chat_level_not_found(error_text=error_text):
        return None
    return kind


@dataclass
class DeliveryTarget:
    """
    A single delivery target.
    
    Represents where a message should be sent:
    - "origin" → back to source
    - "local" → save to local files
    - "telegram" → Telegram home channel
    - "telegram:123456" → specific Telegram chat
    """
    platform: Platform
    chat_id: Optional[str] = None  # None means use home channel
    thread_id: Optional[str] = None
    is_origin: bool = False
    is_explicit: bool = False  # True if chat_id was explicitly specified
    
    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource] = None) -> "DeliveryTarget":
        """
        Parse a delivery target string.
        
        Formats:
        - "origin" → back to source
        - "local" → local files only
        - "telegram" → Telegram home channel
        - "telegram:123456" → specific Telegram chat
        """
        target_stripped = target.strip()
        target_lower = target_stripped.lower()
        
        if target_lower == "origin":
            if origin:
                return cls(
                    platform=origin.platform,
                    chat_id=origin.chat_id,
                    thread_id=origin.thread_id,
                    is_origin=True,
                )
            else:
                # Fallback to local if no origin
                return cls(platform=Platform.LOCAL, is_origin=True)
        
        if target_lower == "local":
            return cls(platform=Platform.LOCAL)
        
        # Check for platform:chat_id or platform:chat_id:thread_id format
        # Use the original case for chat_id/thread_id to preserve case-sensitive IDs
        if ":" in target_stripped:
            parts = target_stripped.split(":", 2)
            platform_str = parts[0].lower()  # Platform names are case-insensitive
            chat_id = parts[1] if len(parts) > 1 else None
            thread_id = parts[2] if len(parts) > 2 else None
            try:
                platform = Platform(platform_str)
                return cls(platform=platform, chat_id=chat_id, thread_id=thread_id, is_explicit=True)
            except ValueError:
                # Unknown platform, treat as local
                return cls(platform=Platform.LOCAL)
        
        # Just a platform name (use home channel)
        try:
            platform = Platform(target_lower)
            return cls(platform=platform)
        except ValueError:
            # Unknown platform, treat as local
            return cls(platform=Platform.LOCAL)
    
    def to_string(self) -> str:
        """Convert back to string format."""
        if self.is_origin:
            return "origin"
        if self.platform == Platform.LOCAL:
            return "local"
        if self.chat_id and self.thread_id:
            return f"{self.platform.value}:{self.chat_id}:{self.thread_id}"
        if self.chat_id:
            return f"{self.platform.value}:{self.chat_id}"
        return self.platform.value


class DeliveryRouter:
    """
    Routes messages to appropriate destinations.
    
    Handles the logic of resolving delivery targets and dispatching
    messages to the right platform adapters.
    """
    
    def __init__(self, config: GatewayConfig, adapters: Dict[Platform, Any] = None,
                 dead_targets: Optional[DeadTargetRegistry] = None):
        """
        Initialize the delivery router.
        
        Args:
            config: Gateway configuration
            adapters: Dict mapping platforms to their adapter instances
            dead_targets: Optional shared registry of confirmed-unreachable
                targets.  When omitted, a profile-local registry is created.
        """
        self.config = config
        self.adapters = adapters or {}
        self.output_dir = get_hermes_home() / "cron" / "output"
        self.dead_targets = dead_targets or DeadTargetRegistry()
    
    async def deliver(
        self,
        content: str,
        targets: List[DeliveryTarget],
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deliver content to all specified targets.
        
        Args:
            content: The message/output to deliver
            targets: List of delivery targets
            job_id: Optional job ID (for cron jobs)
            job_name: Optional job name
            metadata: Additional metadata to include
        
        Returns:
            Dict with delivery results per target
        """
        results = {}
        
        for target in targets:
            # Skip targets we've already proven permanently unreachable
            # (deleted group, blocked/kicked bot, deactivated user). Re-sending
            # to them on every tick wastes a send against flood control and
            # spams logs. Self-healing: a later successful send clears the flag.
            # LOCAL/origin-without-chat targets are never dead-tracked.
            if (
                target.platform != Platform.LOCAL
                and target.chat_id
                and self.dead_targets.is_dead(target.platform.value, target.chat_id)
            ):
                logger.info(
                    "Skipping delivery to known-dead target %s:%s "
                    "(send to it again to clear)",
                    target.platform.value, target.chat_id,
                )
                results[target.to_string()] = {
                    "success": False,
                    "skipped": "dead_target",
                    "error": "target previously confirmed unreachable",
                }
                continue
            try:
                if target.platform == Platform.LOCAL:
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)
                    # Successful platform delivery — clear any stale dead flag.
                    if target.chat_id and not _send_result_failed(result):
                        self.dead_targets.clear(target.platform.value, target.chat_id)
                
                results[target.to_string()] = {
                    "success": True,
                    "result": result
                }
            except Exception as e:
                # A hard failure raises here. If the platform reported a
                # whole-chat death, record it so future deliveries short-circuit.
                if target.platform != Platform.LOCAL and target.chat_id:
                    dead_kind = _classify_dead_from_error_text(str(e))
                    if dead_kind:
                        self.dead_targets.mark_dead(
                            target.platform.value, target.chat_id,
                            reason=f"{dead_kind}: {str(e)[:120]}",
                        )
                results[target.to_string()] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def _deliver_local(
        self,
        content: str,
        job_id: Optional[str],
        job_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Save content to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if job_id:
            output_path = self.output_dir / job_id / f"{timestamp}.md"
        else:
            output_path = self.output_dir / "misc" / f"{timestamp}.md"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the output document
        lines = []
        if job_name:
            lines.append(f"# {job_name}")
        else:
            lines.append("# Delivery Output")
        
        lines.append("")
        lines.append(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if job_id:
            lines.append(f"**Job ID:** {job_id}")
        
        if metadata:
            for key, value in metadata.items():
                lines.append(f"**{key}:** {value}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(content)
        
        output_path.write_text("\n".join(lines))
        
        return {
            "path": str(output_path),
            "timestamp": timestamp
        }
    
    def _save_full_output(self, content: str, job_id: str) -> Path:
        """Save full cron output to disk and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = get_hermes_home() / "cron" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{job_id}_{timestamp}.txt"
        path.write_text(content)
        return path

    def _filter_silence_narration_enabled(self) -> bool:
        """Whether the outbound silence-narration filter is active.

        ``HERMES_FILTER_SILENCE_NARRATION`` env var overrides config when set;
        otherwise the ``gateway.filter_silence_narration`` config flag wins
        (default True).
        """
        env = os.getenv("HERMES_FILTER_SILENCE_NARRATION")
        if env is not None:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return bool(getattr(self.config, "filter_silence_narration", True))

    async def _deliver_to_platform(
        self,
        target: DeliveryTarget,
        content: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deliver content to a messaging platform."""
        transport = resolve_delivery_transport(target.platform, self.config, self.adapters)
        if transport is None:
            raise ValueError(f"No adapter configured for {target.platform.value}")
        adapter = transport.adapter

        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform.value} delivery")
        
        # Guard: handle oversized cron output.
        #
        # Two independent decisions:
        #   1. AUDIT SAVE — when content exceeds MAX_PLATFORM_OUTPUT, the full
        #      output is always written to disk as a recoverable audit trail.
        #      This fires regardless of adapter capability (best-effort).
        #   2. TRUNCATION — for non-chunking adapters, content above the cap is
        #      truncated with a footer pointing to the saved file.  Chunking-
        #      capable adapters (splits_long_messages=True) receive the full
        #      payload and split natively in their send().
        job_id = (metadata or {}).get("job_id", "unknown")
        saved_path: Optional[Path] = None

        if len(content) > MAX_PLATFORM_OUTPUT:
            # Step 1 — audit save (best-effort).  The save is a side-effect
            # audit trail, not essential to delivery.  If it fails (full disk,
            # permissions), delivery proceeds — the content reaches the adapter
            # regardless.
            try:
                saved_path = self._save_full_output(content, job_id)
            except OSError as exc:
                logger.warning(
                    "Audit save failed for cron output (%d chars, job=%s): %s — "
                    "delivery proceeds without audit copy",
                    len(content), job_id, exc,
                )

            # Step 2 — truncation (only for non-chunking adapters).
            if getattr(adapter, "splits_long_messages", False):
                # Adapter chunks natively — deliver full payload.
                if saved_path:
                    logger.info(
                        "Cron output preserved for chunking adapter (%d chars) — "
                        "full output saved to %s",
                        len(content), saved_path,
                    )
            else:
                # Non-chunking adapter — truncate with footer.  The footer
                # needs a valid path, so if the best-effort save above failed,
                # retry it here (a failure now is a real delivery problem).
                if saved_path is None:
                    saved_path = self._save_full_output(content, job_id)
                footer = f"\n\n... [truncated, full output saved to {saved_path}]"
                visible = max(0, MAX_PLATFORM_OUTPUT - len(footer))
                logger.info(
                    "Cron output truncated (%d chars) — full output: %s",
                    len(content), saved_path,
                )
                content = content[:visible] + footer
        
        # Substrate-level anti-loop guard: drop hallucinated "silence narration"
        # (*(silent)*, 🔇, a bare ".", etc.) before it ever reaches the adapter.
        # In bot-to-bot channels these tokens mirror back and forth until a
        # model crashes with "no content after all retries". Behavioral prompt
        # rules drift across providers; this single chokepoint covers every
        # platform adapter regardless of which persona's prompt failed.
        # Local/file delivery (_deliver_local) is a separate path and is never
        # filtered — saved silence has no loop risk.
        if self._filter_silence_narration_enabled() and _is_silence_narration(content):
            logger.warning(
                "Dropped silence-narration outbound to %s (chat=%s): %r",
                target.platform.value,
                target.chat_id,
                content[:40],
            )
            return {
                "success": True,
                "filtered": "silence_narration",
                "delivered": False,
            }

        send_metadata = dict(metadata or {})
        if transport.is_relay:
            home = self.config.get_home_channel(target.platform)
            if home is not None and home.chat_id == target.chat_id:
                if home.user_id:
                    send_metadata["user_id"] = home.user_id
                if home.scope_id:
                    send_metadata["scope_id"] = home.scope_id
        is_named_telegram_private_topic = False
        named_telegram_private_topic_name: Optional[str] = None
        if target.thread_id:
            has_explicit_direct_topic = (
                "direct_messages_topic_id" in send_metadata
                or "telegram_direct_messages_topic_id" in send_metadata
            )
            target_thread_id = target.thread_id
            is_named_telegram_private_topic = (
                target.platform == Platform.TELEGRAM
                and looks_like_telegram_private_chat_id(target.chat_id)
                and not _looks_like_int(target_thread_id)
                and "thread_id" not in send_metadata
                and "message_thread_id" not in send_metadata
                and not has_explicit_direct_topic
            )
            if is_named_telegram_private_topic:
                named_telegram_private_topic_name = target_thread_id
                ensure_dm_topic = getattr(adapter, "ensure_dm_topic", None)
                if ensure_dm_topic is None:
                    raise RuntimeError(
                        "Telegram adapter cannot create named private DM topics"
                    )
                created_thread_id = await ensure_dm_topic(target.chat_id, target_thread_id)
                if not created_thread_id:
                    raise RuntimeError(
                        f"Failed to create Telegram private DM topic '{target_thread_id}'"
                    )
                target_thread_id = str(created_thread_id)
                send_metadata["thread_id"] = target_thread_id
                send_metadata["telegram_dm_topic_created_for_send"] = True
            elif (
                target.platform == Platform.TELEGRAM
                and looks_like_telegram_private_chat_id(target.chat_id)
                and "thread_id" not in send_metadata
                and "message_thread_id" not in send_metadata
                and not has_explicit_direct_topic
            ):
                # Legacy private topic/thread ids that were not created by this
                # send path may still need a reply anchor to stay visible in the
                # requested lane. Named targets are created above via
                # createForumTopic and can use message_thread_id directly.
                reply_anchor = send_metadata.get("telegram_reply_to_message_id")
                if reply_anchor is None:
                    raise RuntimeError(
                        "Telegram private DM topic delivery requires telegram_reply_to_message_id; "
                        "send to the bare chat or provide a reply anchor"
                    )
                send_metadata["thread_id"] = target_thread_id
                send_metadata["telegram_dm_topic_reply_fallback"] = True
            elif "thread_id" not in send_metadata and "message_thread_id" not in send_metadata and not has_explicit_direct_topic:
                send_metadata["thread_id"] = target_thread_id
        result = await transport.send(
            target.platform,
            target.chat_id,
            content,
            metadata=send_metadata or None,
        )
        if _send_result_failed(result):
            if (
                is_named_telegram_private_topic
                and named_telegram_private_topic_name
                and _is_thread_not_found_delivery_error(result)
            ):
                ensure_dm_topic = getattr(adapter, "ensure_dm_topic", None)
                if ensure_dm_topic is None:
                    raise RuntimeError(
                        "Telegram adapter cannot refresh named private DM topics"
                    )
                refreshed_thread_id = await ensure_dm_topic(
                    target.chat_id,
                    named_telegram_private_topic_name,
                    force_create=True,
                )
                if not refreshed_thread_id:
                    raise RuntimeError(
                        f"Failed to refresh Telegram private DM topic '{named_telegram_private_topic_name}'"
                    )
                send_metadata["thread_id"] = str(refreshed_thread_id)
                send_metadata["telegram_dm_topic_created_for_send"] = True
                result = await transport.send(
                    target.platform,
                    target.chat_id,
                    content,
                    metadata=send_metadata or None,
                )
            if _send_result_failed(result):
                raise RuntimeError(_send_result_error(result) or f"{target.platform.value} delivery failed")
        return result




