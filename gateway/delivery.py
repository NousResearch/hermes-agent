"""Delivery routing for cron job outputs and agent responses, by target: explicit ("telegram:123456789"),
platform home channel ("telegram"), origin (back to where the job was created), or local (files)."""

import logging
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from hermes_cli.config import get_hermes_home

from .config import Platform, GatewayConfig, PlatformConfig
from .degenerate_output import (
    COLLAPSE_MARKER,
    DEFAULT_MAX_DELIVERY_CHARS,
    collapse_degenerate,
    is_degenerate,
    truncate_for_delivery,
)
from .session import SessionSource
from .dead_targets import DeadTargetRegistry, classify_dead_error

logger = logging.getLogger(__name__)

# Cap before gateway-level truncation of cron output for non-chunking platform delivery. Telegram's hard
# API limit is 4096; the headroom covers the "full output saved to …" footer. Adapters that split long
# messages natively (splits_long_messages) bypass this entirely.
MAX_PLATFORM_OUTPUT = 4000
# Matches strings that are *only* a "silence" narration with optional markdown wrappers (*(silent)*,
# _silent_, 🔇, a bare ".", "…"). Anchored so messages that merely *contain* "silent" never match.
_SILENCE_NARRATION = re.compile(
    r'^[\s*_~`]*\(?\s*(silent|silence|no\s+response|no\s+reply)\s*\.?\)?[\s*_~`]*$'
    r'|^[\s*_~`]*[\U0001F507\.\u2026]+[\s*_~`]*$',
    re.IGNORECASE,
)
_THREAD_ROUTING_KEYS = ("thread_id", "message_thread_id", "direct_messages_topic_id", "telegram_direct_messages_topic_id")


def _is_silence_narration(content: Optional[str]) -> bool:
    """True when ``content`` is *only* a silence-narration token (length-guarded)."""
    stripped = content.strip() if content else ""
    return bool(stripped) and len(stripped) <= 64 and bool(_SILENCE_NARRATION.match(stripped))


def save_full_output_copy(content: str, job_id: str) -> Path:
    """Write the complete original of one delivery to the audit dir and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = get_hermes_home() / "cron" / "output" / f"{job_id}_{timestamp}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def resolve_max_delivery_chars(config: Any) -> int:
    """Hard cap on the characters of a single outbound delivery (``cron.max_delivery_chars``).

    Configured in ``config.yaml`` as ``cron.max_delivery_chars`` (default 20000). The cap is applied
    *before* any adapter-specific handling, including adapters that split long messages natively —
    those previously received the entire payload and fanned a 213 KB degenerate cron response into a
    chat one message at a time (2026-11-02 incident). A value of 0 (or any non-positive number)
    disables the cap, matching the "0 = unlimited" convention of the other cron knobs.

    Only a real integer (or an integer-valued string) is a cap: a bool, a float with a fractional
    part, or any other object is a configuration error, because coercing it blindly would turn a
    non-numeric value into a tiny cap (``int(MagicMock())`` is 1) and truncate every delivery.
    """
    raw = getattr(config, "max_delivery_chars", None)
    if not isinstance(raw, (int, str)) or isinstance(raw, bool):
        if raw is not None:
            logger.warning("Invalid cron.max_delivery_chars=%r — falling back to %d",
                           raw, DEFAULT_MAX_DELIVERY_CHARS)
        return DEFAULT_MAX_DELIVERY_CHARS
    try:
        limit = int(raw.strip(), 10) if isinstance(raw, str) else int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid cron.max_delivery_chars=%r — falling back to %d",
                       raw, DEFAULT_MAX_DELIVERY_CHARS)
        return DEFAULT_MAX_DELIVERY_CHARS
    return max(0, limit)


def guard_degenerate_output(
    content: str, job_id: str, save_full_output: Callable[[str, str], Path],
) -> tuple[str, Optional[Path]]:
    """Detect a degenerate (runaway-repetition) payload and collapse it before any adapter sees it.

    Runs FIRST, on the ORIGINAL payload, before the size cap or adapter-specific handling, and
    returns ``(content, saved_path)``. Order matters: the detector must see the full response. The
    2026-11-02 incident's 213 KB payload was a readable preamble followed by a ~180 KB looping tail,
    so truncating first would have sliced the degenerate tail away *before* detection and let a
    preamble ship with no marker. The detector is statistical (no word blacklist), so ordinary
    reports — including ones that legitimately repeat a line or two — pass through untouched.

    The full original is audit-saved BEFORE it is shortened so the pointer in the delivered message
    always resolves to the complete original. A failed save is best-effort: delivery proceeds with
    the collapsed message and no audit pointer.

    Idempotent: a payload that already carries the collapse marker has been collapsed upstream —
    the cron scheduler collapses the run's final response before composing its delivery, and both
    delivery lanes then re-run this guard on that text.  It is passed through untouched rather
    than re-saved and re-collapsed: the audit copy would otherwise store the *excerpt* as if it
    were the original, and a second collapse would mangle what the marker preserved.
    """
    if not is_degenerate(content) or COLLAPSE_MARKER in content:
        return content, None
    original_len = len(content)
    saved_path: Optional[Path] = None
    try:
        saved_path = save_full_output(content, job_id)
    except OSError as exc:
        logger.warning(
            "Audit save failed for degenerate cron output (%d chars, job=%s): %s — "
            "delivery proceeds with a collapsed message and no audit pointer",
            original_len, job_id, exc)
    collapsed = collapse_degenerate(content)
    logger.warning(
        "Degenerate repetitive output detected before delivery (%d chars, job=%s) — "
        "collapsing to %d chars before adapter send",
        original_len, job_id, len(collapsed))
    if collapsed.strip():
        if saved_path is not None:
            collapsed += (f"\n\n[完整原始输出（{original_len} 字符）已保存到 {saved_path}]")
        return collapsed, saved_path
    # Nothing outside the degenerate span was salvageable: deliver a short notice instead of an
    # empty or looping message.
    notice = f"[输出因模型退化重复已被抑制：本次响应无可读正文。原始长度 {original_len} 字符"
    if saved_path is not None:
        notice += f"；完整原始响应已保存到 {saved_path}"
    return notice + "]", saved_path


def guard_outbound_content(
    content: str, job_id: str, *, limit: int, save_full_output: Callable[[str, str], Path],
    chunking_adapter: bool = False,
) -> tuple[str, Optional[Path]]:
    """The outbound guard chain every delivery lane must run: collapse, then the universal cap.

    Returns ``(content, audit_path)``. The caller's adapter-specific ``MAX_PLATFORM_OUTPUT`` cap
    then runs unconditionally: the two caps answer different questions (flood vs per-message) and a
    caller must never be able to skip the second one (PR 148960 review finding 4).

    ``chunking_adapter`` asks for the audit pointer at the HEAD of the capped body as well as the
    tail, because a ``splits_long_messages`` adapter re-splits what it is handed into messages
    smaller than this cap and may drop the trailing ones entirely — a tail-only footer then never
    reaches the reader (PR 148960 review finding 3). A caller with no adapter concept (the cron
    standalone fallback, whose ``_send_to_platform`` also chunks) keeps the tail footer and passes
    ``False``; local/file delivery never reaches this function.

    Two lanes reach a chat: ``DeliveryRouter._deliver_to_platform`` (live adapters) and the cron
    standalone fallback (``cron/scheduler_delivery.py::_deliver_standalone``), which sends through
    ``tools.send_message_tool._send_to_platform`` directly. Both call THIS function, so a guard can
    never exist on one lane and be missing on the other (PR 148960 review finding 1: the fallback
    lane had neither, and a 72,000-char degenerate payload reached the wire verbatim).
    """
    content, audit_path = guard_degenerate_output(content, job_id, save_full_output)

    # Universal delivery cap. Independent of adapter chunking behaviour: no adapter may ever receive
    # an unbounded payload. Runs after the degeneracy guard so a collapsed payload is capped on its
    # own (post-collapse) size.
    if limit and len(content) > limit:
        if audit_path is None:
            # Save the full payload for audit BEFORE it is shortened, so the footer pointer always
            # resolves to the complete original. Best-effort: a failed save never blocks delivery.
            try:
                audit_path = save_full_output(content, job_id)
            except OSError as exc:
                logger.warning("Audit save failed for oversized cron output (%d chars, job=%s): %s — "
                               "delivery proceeds without audit copy", len(content), job_id, exc)
        logger.warning("Cron output exceeded the delivery cap (%d chars > %d, job=%s) — truncating "
                       "before delivery", len(content), limit, job_id)
        content = truncate_for_delivery(
            content, limit=limit, saved_path=str(audit_path) if audit_path else None,
            pointer_at_start=chunking_adapter)
    return content, audit_path


class PartialDeliveryError(RuntimeError):
    """A split send failed after earlier chunks were delivered (``raw_response["partial_overflow"]``).
    Callers must not fall back to re-sending the whole payload: the recipient already has the head."""

    def __init__(self, message: str, result: Any):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class DeliveryTransport:
    """Resolved live transport for one logical delivery platform."""
    adapter: Any
    config: Optional[PlatformConfig]
    transport_platform: Platform

    @property
    def is_relay(self) -> bool:
        return self.transport_platform == Platform.RELAY

    async def send(self, logical_platform: Platform, chat_id: str, content: str,
                   metadata: Optional[dict[str, Any]]) -> Any:
        """Send through this transport while preserving the logical platform."""
        return await (self.adapter.send_for_platform(logical_platform, chat_id, content, metadata=metadata)
                      if self.is_relay else self.adapter.send(chat_id, content, metadata=metadata))


def resolve_delivery_transport(platform: Platform, config: GatewayConfig,
                               adapters: Optional[dict[Platform, Any]]) -> Optional[DeliveryTransport]:
    """Resolve a logical platform to its live delivery transport. A concrete native adapter always wins;
    Relay is eligible only when its authenticated transport explicitly advertises that it fronts the
    logical platform, so restart-time delivery is independent of per-chat caches without letting Relay
    hijack unrelated platform targets."""
    live_adapters = adapters or {}
    native, native_config = live_adapters.get(platform), config.platforms.get(platform)
    # Explicitly supplied live adapters with no config block are honored, but an
    # explicitly disabled native adapter never shadows an enabled Relay transport.
    if native is not None and (native_config is None or native_config.enabled):
        return DeliveryTransport(native, native_config, platform)
    relay, relay_config = live_adapters.get(Platform.RELAY), config.platforms.get(Platform.RELAY)
    fronts_platform = getattr(relay, "fronts_platform", None)
    if (relay is not None and (relay_config is None or relay_config.enabled)
            and callable(fronts_platform) and fronts_platform(platform)):
        return DeliveryTransport(relay, relay_config, Platform.RELAY)
    return None


def looks_like_telegram_private_chat_id(chat_id: Optional[str]) -> bool:
    """True when ``chat_id`` is a positive int — Telegram's private-chat shape (groups/channels are negative).
    Single source of truth, reused by the handoff seed path in ``gateway/run.py`` so handoff-created DM
    topics key the same way as inbound DM-topic messages."""
    try:
        return int(chat_id) > 0
    except (TypeError, ValueError):
        return False


def _looks_like_int(value: Optional[str]) -> bool:
    try:
        return int(value) is not None
    except (TypeError, ValueError):
        return False


def _send_result_error(result: Any) -> Optional[str]:
    """Error string of a failed SendResult object / plain result dict ("" if none), or None on success."""
    get = result.get if isinstance(result, dict) else (lambda name, default=None: getattr(result, name, default))
    return None if get("success", True) is not False else str(get("error") or "")


@dataclass
class DeliveryTarget:
    """One target: "origin", "local", "telegram" (home channel) or "telegram:123456[:thread]"."""
    platform: Platform
    chat_id: Optional[str] = None  # None means use home channel
    thread_id: Optional[str] = None
    is_origin: bool = False
    is_explicit: bool = False  # True if chat_id was explicitly specified
    # Raw target string when the platform name is unknown. The platform falls
    # back to LOCAL for routing, but the original name is preserved so
    # deliver() can report {success: False, error: unknown_platform} instead
    # of silently misrouting to local files.
    unknown_platform: Optional[str] = None

    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource] = None) -> "DeliveryTarget":
        """Parse "origin" | "local" | "<platform>" | "<platform>:<chat_id>[:<thread_id>]"."""
        target = target.strip()
        if target.lower() == "origin":
            return (cls(platform=origin.platform, chat_id=origin.chat_id, thread_id=origin.thread_id, is_origin=True)
                    if origin else cls(platform=Platform.LOCAL, is_origin=True))
        # Platform names are case-insensitive; chat/thread ids keep case. Unknown platforms ->
        # LOCAL for routing but preserve the raw target so deliver() reports
        # unknown_platform instead of silently saving locally.
        parts = target.split(":", 2)
        try:
            platform = Platform(parts[0].lower())
        except ValueError:
            return cls(platform=Platform.LOCAL, unknown_platform=target)
        return (cls(platform=platform, chat_id=parts[1], thread_id=parts[2] if len(parts) > 2 else None, is_explicit=True)
                if len(parts) > 1 else cls(platform=platform))

    def to_string(self) -> str:
        """Convert back to string format."""
        if self.is_origin:
            return "origin"
        if self.unknown_platform is not None:
            return self.unknown_platform
        if self.platform == Platform.LOCAL:
            return "local"
        parts = [self.platform.value, self.chat_id, self.thread_id if self.chat_id else None]
        return ":".join(p for p in parts if p)


async def _ensure_named_dm_topic(adapter: Any, chat_id: str, name: str, *, refresh: bool) -> str:
    """Create (or force-recreate) a named Telegram private DM topic; return its thread id."""
    verb, ensure_dm_topic = "refresh" if refresh else "create", getattr(adapter, "ensure_dm_topic", None)
    if ensure_dm_topic is None:
        raise RuntimeError(f"Telegram adapter cannot {verb} named private DM topics")
    thread_id = await ensure_dm_topic(chat_id, name, **({"force_create": True} if refresh else {}))
    if not thread_id:
        raise RuntimeError(f"Failed to {verb} Telegram private DM topic '{name}'")
    return str(thread_id)


class DeliveryRouter:
    """Resolves delivery targets and dispatches messages to platform adapters."""

    def __init__(self, config: GatewayConfig, adapters: dict[Platform, Any] | None = None,
                 dead_targets: Optional[DeadTargetRegistry] = None):  # profile-local registry when omitted
        self.config = config
        self.adapters = adapters or {}
        self.output_dir = get_hermes_home() / "cron" / "output"
        self.dead_targets = dead_targets or DeadTargetRegistry()

    async def deliver(self, content: str, targets: list[DeliveryTarget], job_id: Optional[str] = None,
                      job_name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Deliver content to all targets; returns per-target results keyed by target string."""
        results = {}
        for target in targets:
            if target.unknown_platform is not None:
                results[target.to_string()] = {
                    "success": False, "error": f"unknown_platform: {target.unknown_platform}"}
                continue
            # Skip targets proven permanently unreachable (deleted group, blocked bot, deactivated user) —
            # re-sending each tick wastes flood-control budget. Self-healing: a later successful send
            # clears the flag. LOCAL/origin-without-chat targets are never dead-tracked.
            tracked = target.platform != Platform.LOCAL and target.chat_id
            if tracked and self.dead_targets.is_dead(target.platform.value, target.chat_id):
                logger.info("Skipping delivery to known-dead target %s:%s (send to it again to clear)",
                            target.platform.value, target.chat_id)
                results[target.to_string()] = {"success": False, "skipped": "dead_target",
                                               "error": "target previously confirmed unreachable"}
                continue
            try:
                if target.platform == Platform.LOCAL:
                    result = self._deliver_local(content, job_id, job_name, metadata)
                else:
                    result = await self._deliver_to_platform(target, content, metadata)
                    if target.chat_id and _send_result_error(result) is None:
                        self.dead_targets.clear(target.platform.value, target.chat_id)
                results[target.to_string()] = {"success": True, "result": result}
            except Exception as e:
                # Hard failures raise. Record a whole-chat death so future deliveries short-circuit.
                dead_kind = classify_dead_error(str(e)) if tracked else None
                if dead_kind:
                    self.dead_targets.mark_dead(target.platform.value, target.chat_id,
                                                reason=f"{dead_kind}: {str(e)[:120]}")
                results[target.to_string()] = {"success": False, "error": str(e)}
        return results

    def _deliver_local(self, content: str, job_id: Optional[str], job_name: Optional[str],
                       metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Save content to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / (job_id or "misc") / f"{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {job_name}" if job_name else "# Delivery Output", "",
                 f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        lines += [f"**Job ID:** {job_id}"] if job_id else []
        lines += [f"**{key}:** {value}" for key, value in (metadata or {}).items()] + ["", "---", "", content]
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return {"path": str(output_path), "timestamp": timestamp}

    def _save_full_output(self, content: str, job_id: str) -> Path:
        """Save full cron output to disk and return the file path."""
        return save_full_output_copy(content, job_id)

    def _filter_silence_narration_enabled(self) -> bool:
        """filter silence narration based on gateway config without checking process env"""
        return bool(getattr(self.config, "filter_silence_narration", True))

    def _max_delivery_chars(self) -> int:
        """Hard cap on the characters of a single outbound delivery (``cron.max_delivery_chars``)."""
        return resolve_max_delivery_chars(self.config)

    def _cap_oversized_output(self, adapter: Any, content: str, job_id: str,
                              saved_path: Optional[Path] = None) -> str:
        """Audit-save oversized cron output; truncate it for non-chunking adapters. Above MAX_PLATFORM_OUTPUT
        the full output is always written to disk as an audit trail, best-effort — a failed save (full disk,
        permissions) never blocks delivery. Non-chunking adapters then get the content truncated with a
        footer pointing to the saved file; ``splits_long_messages`` adapters receive the full payload.

        ``saved_path`` is an audit copy the caller already wrote (the universal cap saves the ORIGINAL
        before shortening it), passed in so the same payload is never saved twice.

        This cap is per-MESSAGE, not the flood guard: an adapter that cannot split a long message
        enforces its own limit downstream and drops the tail (Google Chat cuts at 4000, ntfy at 4096,
        IRC at 450), so that reader must receive the head *and* the pointer inside the one message it
        can show. It therefore still applies when ``cron.max_delivery_chars`` already shortened the
        payload — the two caps answer different questions (PR 148960 review finding 4).
        """
        if len(content) <= MAX_PLATFORM_OUTPUT:
            return content
        if saved_path is None:
            try:
                saved_path = self._save_full_output(content, job_id)
            except OSError as exc:
                logger.warning("Audit save failed for cron output (%d chars, job=%s): %s — "
                               "delivery proceeds without audit copy", len(content), job_id, exc)
        if getattr(adapter, "splits_long_messages", False):
            if saved_path:
                logger.info("Cron output preserved for chunking adapter (%d chars) — "
                            "full output saved to %s", len(content), saved_path)
            return content
        # The footer needs a valid path: if the best-effort save failed, retry
        # (a failure now is a real delivery problem and propagates).
        saved_path = saved_path or self._save_full_output(content, job_id)
        footer = f"\n\n... [truncated, full output saved to {saved_path}]"
        logger.info("Cron output truncated (%d chars) — full output: %s", len(content), saved_path)
        return content[:max(0, MAX_PLATFORM_OUTPUT - len(footer))] + footer

    async def _deliver_to_platform(self, target: DeliveryTarget, content: str,
                                   metadata: Optional[dict[str, Any]],
                                   transport: Optional[DeliveryTransport] = None,
                                   ) -> dict[str, Any]:
        """Deliver content to a messaging platform.

        ``transport`` carries an already-authorized transport past resolution:
        the cron live lane resolved and authorized it per target (including the
        SharedRouteAdapters satellite grant), and re-resolving from the plain
        adapters dict cannot re-derive that grant under satellite config
        (#115656). Omitted (None) preserves resolution for every other caller.
        """
        if transport is None:
            transport = resolve_delivery_transport(target.platform, self.config, self.adapters)
        if transport is None:
            raise ValueError(f"No adapter configured for {target.platform.value}")
        if not target.chat_id:
            raise ValueError(f"No chat ID for {target.platform.value} delivery")
        adapter = transport.adapter
        job_id = (metadata or {}).get("job_id", "unknown")

        # Steps 0-1 — degeneracy collapse and then the universal size cap, on the ORIGINAL payload
        # before any adapter-specific handling (including native chunking). The detector must see the
        # full response; truncating a preamble-plus-loop response first would hide the looping tail.
        # The per-MESSAGE adapter cap follows unconditionally: an adapter that cannot split a long
        # message cuts the tail itself and would drop the audit pointer with it.
        content, audit_path = guard_outbound_content(
            content, job_id, limit=self._max_delivery_chars(), save_full_output=self._save_full_output,
            chunking_adapter=bool(getattr(adapter, "splits_long_messages", False)))
        # ``audit_path`` (when present) holds the ORIGINAL payload — saved by the universal cap or by
        # the degeneracy collapse — so the adapter cap reuses it instead of writing a second copy.
        content = self._cap_oversized_output(adapter, content, job_id, saved_path=audit_path)

        # Substrate-level anti-loop guard: drop hallucinated "silence narration" (*(silent)*, 🔇, a bare ".")
        # before it reaches any adapter — in bot-to-bot channels these mirror back and forth until a model
        # crashes with "no content after all retries"; prompt rules drift across providers, so this single
        # chokepoint covers every platform. Local/file delivery is never filtered (saved silence has no loop
        # risk). Cron output is an ARTIFACT, not model chatter: a legitimately terse job ("...", a single 🔇)
        # has no mirror loop, and dropping it while returning success is how a cron gets logged as delivered
        # with nothing on the wire. Cron sends carry job_id in metadata; everything else is filtered.
        # See #77763.
        is_cron_artifact = "job_id" in (metadata or {})
        if self._filter_silence_narration_enabled() and not is_cron_artifact and _is_silence_narration(content):
            logger.warning("Dropped silence-narration outbound to %s (chat=%s): %r",
                           target.platform.value, target.chat_id, content[:40])
            return {"success": True, "filtered": "silence_narration", "delivered": False}

        send_metadata = dict(metadata or {})
        home = self.config.get_home_channel(target.platform) if transport.is_relay else None
        if home is not None and home.chat_id == target.chat_id:
            send_metadata.update({k: v for k, v in (("user_id", home.user_id), ("scope_id", home.scope_id)) if v})

        # Caller-supplied thread routing always wins over target.thread_id.
        named_topic: Optional[str] = None  # named Telegram private topic created for this send
        thread_id = target.thread_id
        if thread_id and not any(key in send_metadata for key in _THREAD_ROUTING_KEYS):
            send_metadata["thread_id"] = thread_id
            if target.platform == Platform.TELEGRAM and looks_like_telegram_private_chat_id(target.chat_id):
                if not _looks_like_int(thread_id):
                    # Named topic: create via createForumTopic, use message_thread_id directly.
                    named_topic = thread_id
                    send_metadata["thread_id"] = await _ensure_named_dm_topic(adapter, target.chat_id, thread_id, refresh=False)
                    send_metadata["telegram_dm_topic_created_for_send"] = True
                elif send_metadata.get("telegram_reply_to_message_id") is None:
                    # Legacy numeric private topic ids not created by this send path need a reply
                    # anchor to stay visible in the requested lane.
                    raise RuntimeError(
                        "Telegram private DM topic delivery requires telegram_reply_to_message_id; "
                        "send to the bare chat or provide a reply anchor"
                    )
                else:
                    send_metadata["telegram_dm_topic_reply_fallback"] = True

        for retry in (False, True):
            result = await transport.send(target.platform, target.chat_id, content, metadata=send_metadata or None)
            error = _send_result_error(result)
            if retry or error is None or not named_topic or "thread not found" not in error.lower():
                break
            # The named topic vanished under us: recreate it once and resend.
            send_metadata["thread_id"] = await _ensure_named_dm_topic(adapter, target.chat_id, named_topic, refresh=True)
            send_metadata["telegram_dm_topic_created_for_send"] = True
        if error is not None:
            from gateway.platforms.base import BasePlatformAdapter
            if BasePlatformAdapter._is_partial_delivery(result):
                raise PartialDeliveryError(error, result)
            raise RuntimeError(error or f"{target.platform.value} delivery failed")
        return result
