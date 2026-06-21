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

# Default cap before gateway-level truncation of cron output for platform
# delivery.  Telegram's hard API limit is 4096; the 200-char headroom covers
# the "full output saved to …" footer appended on truncation.  Override via
# the HERMES_DELIVERY_MAX_PLATFORM_OUTPUT env var.  Adapters that split long
# messages natively (BasePlatformAdapter.splits_long_messages) bypass this
# entirely — the adapter chunks in its own send() and the full output is
# preserved.
_DEFAULT_MAX_PLATFORM_OUTPUT = 4000


def _max_platform_output() -> int:
    """Max chars before gateway-level truncation of cron output.

    ``HERMES_DELIVERY_MAX_PLATFORM_OUTPUT`` env var overrides the default
    (4000).  Non-int or negative values fall back to the default with a
    warning.
    """
    env = os.getenv("HERMES_DELIVERY_MAX_PLATFORM_OUTPUT")
    if env is not None:
        try:
            return max(0, int(env.strip()))
        except ValueError:
            logger.warning(
                "HERMES_DELIVERY_MAX_PLATFORM_OUTPUT=%r is not an int; "
                "using default %d",
                env, _DEFAULT_MAX_PLATFORM_OUTPUT,
            )
    return _DEFAULT_MAX_PLATFORM_OUTPUT

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

from .config import Platform, GatewayConfig
from .session import SessionSource


def _looks_like_telegram_private_chat_id(chat_id: Optional[str]) -> bool:
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
    
    def __init__(self, config: GatewayConfig, adapters: Dict[Platform, Any] = None):
        """
        Initialize the delivery router.
        
        Args:
            config: Gateway configuration
            adapters: Dict mapping platforms to their adapter instances
        """
        self.config = config
        self.adapters = adapters or {}
        self.output_dir = get_hermes_home() / "cron" / "output"
    
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
            try:
                if target.platform == Platform.LOCAL:
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)
                
                results[target.to_string()] = {
                    "success": True,
                    "result": result
                }
            except Exception as e:
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
        adapter = self.adapters.get(target.platform)
        
        if not adapter:
            raise ValueError(f"No adapter configured for {target.platform.value}")
        
        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform.value} delivery")
        
        # Guard: handle oversized cron output.
        #
        # Two independent decisions:
        #   1. AUDIT SAVE — when content exceeds the audit threshold (4000
        #      chars, the historical default), the full output is always
        #      written to disk as a recoverable audit trail.  This fires
        #      regardless of truncation setting or adapter capability.
        #   2. TRUNCATION — for non-chunking adapters, content above
        #      max_output is truncated with a footer pointing to the saved
        #      file.  Chunking-capable adapters (splits_long_messages=True)
        #      receive the full payload and split natively in their send().
        #      Setting HERMES_DELIVERY_MAX_PLATFORM_OUTPUT=0 disables
        #      truncation entirely (the user takes responsibility for platform
        #      API limits), but the audit save in step 1 still fires.
        max_output = _max_platform_output()
        job_id = (metadata or {}).get("job_id", "unknown")
        saved_path: Optional[Path] = None

        # Step 1 — audit save (independent of truncation, best-effort).
        # The save is a side-effect audit trail, not essential to delivery.
        # If it fails (full disk, permissions), delivery proceeds — the
        # content reaches the adapter regardless.  The truncation path's
        # fallback save below is NOT best-effort: the footer needs a valid
        # path, so a failure there is a real delivery problem.
        if len(content) > _DEFAULT_MAX_PLATFORM_OUTPUT:
            try:
                saved_path = self._save_full_output(content, job_id)
            except OSError as exc:
                logger.warning(
                    "Audit save failed for cron output (%d chars, job=%s): %s — "
                    "delivery proceeds without audit copy",
                    len(content), job_id, exc,
                )

        # Step 2 — truncation (only for non-chunking adapters).
        if max_output > 0 and len(content) > max_output:
            if adapter and getattr(adapter, "splits_long_messages", False):
                # Adapter chunks natively — deliver full payload.
                if saved_path:
                    logger.info(
                        "Cron output preserved for chunking adapter (%d chars) — "
                        "full output saved to %s",
                        len(content), saved_path,
                    )
            else:
                # Non-chunking adapter — truncate with footer.
                if saved_path is None:
                    # Content exceeded max_output but not the audit threshold
                    # (e.g. HERMES_DELIVERY_MAX_PLATFORM_OUTPUT=200).  Save
                    # anyway since we're about to truncate.
                    saved_path = self._save_full_output(content, job_id)
                footer = f"\n\n... [truncated, full output saved to {saved_path}]"
                visible = max(0, max_output - len(footer))
                logger.info(
                    "Cron output truncated (%d chars) — full output: %s",
                    len(content), saved_path,
                )
                content = content[:visible] + footer
        elif saved_path:
            # Truncation disabled (max_output=0) but content was large enough
            # to warrant an audit copy.
            logger.info(
                "Cron output delivered untruncated (%d chars, truncation "
                "disabled) — audit copy saved to %s",
                len(content), saved_path,
            )
        
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
                and _looks_like_telegram_private_chat_id(target.chat_id)
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
                and _looks_like_telegram_private_chat_id(target.chat_id)
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
        result = await adapter.send(target.chat_id, content, metadata=send_metadata or None)
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
                result = await adapter.send(target.chat_id, content, metadata=send_metadata or None)
            if _send_result_failed(result):
                raise RuntimeError(_send_result_error(result) or f"{target.platform.value} delivery failed")
        return result




