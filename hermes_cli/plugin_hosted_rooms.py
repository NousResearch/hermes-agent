"""``ctx.hosted_rooms``: the plugin facade between a Telegram platform handler and the gateway's
hosted Group Chat rooms (``HostedRoomService``).

Disabled-first and fail-closed. Every verb re-checks, at call time and in this order: the
``gateway.hosted_room_ingress`` capability (read from the plugin's own profile config), a running
hosted-room worker in this process, and a room protocol this facade was written against. Any miss
returns a typed ``disabled`` result; no verb raises into PTB handler dispatch, so ordinary Telegram
handling continues unchanged. Coordinates are derived from the adapter-delivered ``Update`` here,
never accepted from the plugin. ``publish_outcome`` is coordinator-only and separately gated
(``gateway.hosted_room_publish``): it re-verifies a settled reply against this gateway's own room
log and posts it as the gateway bot to the binding's chat/topic only, never a caller-chosen target.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

FACADE_VERSION = 1
CAPABILITY_ID = "gateway.hosted_room_ingress"
PUBLISH_CAPABILITY_ID = "gateway.hosted_room_publish"
MAX_PUBLICATION_CHARS = 4000
MAX_PUBLICATION_LABEL_CHARS = 64
SUPPORTED_ROOM_PROTOCOLS = frozenset({2})
_GROUP_CHAT_TYPES = frozenset({"group", "supergroup"})
_OUTCOME_KINDS = {"turn.settled": "settled", "turn.failed": "failed",
                  "turn.cancelled": "cancelled", "turn.deferred": "deferred"}

IngressStatus = Literal["accepted", "duplicate", "disabled", "rejected", "failed"]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _publication_label(label: Any, profile: str) -> Optional[str]:
    """Return a safe caller-supplied display label, or the verified profile when omitted."""
    if label is None:
        return profile
    if not isinstance(label, str) or label != label.strip() or not label \
            or len(label) > MAX_PUBLICATION_LABEL_CHARS \
            or any(ord(char) < 32 or char in "[]\x7f" for char in label):
        return None
    return label


@dataclass(frozen=True)
class TelegramRoomBinding:
    """Operator-pinned route: one chat (and topic; ``None`` = non-topic messages) into one room."""

    room_id: str
    chat_id: int
    thread_id: Optional[int]
    sender_ids: frozenset

    def __post_init__(self) -> None:
        if not isinstance(self.room_id, str) or not self.room_id.strip():
            raise ValueError("room_id must be a non-empty string")
        if not _is_int(self.chat_id) or not (self.thread_id is None or _is_int(self.thread_id)):
            raise ValueError("chat_id/thread_id must be integers")
        if not self.sender_ids or not all(_is_int(s) and s > 0 for s in self.sender_ids):
            raise ValueError("sender_ids must be a non-empty set of positive Telegram user ids")

    @property
    def room_thread_id(self) -> str:
        return f"tg:{self.chat_id}:{'main' if self.thread_id is None else self.thread_id}"


@dataclass(frozen=True)
class TelegramIngressEvent:
    """Coordinates the facade derived from the Update itself."""

    update_id: int
    chat_id: int
    thread_id: Optional[int]
    message_id: int
    sender_id: int
    text: str


@dataclass(frozen=True)
class FacadeStatus:
    facade_version: int
    enabled: bool
    reason: str
    room_protocol: Optional[int]
    publication: bool = False
    publication_labels: bool = False


@dataclass(frozen=True)
class IngressResult:
    status: IngressStatus
    reason: str = ""
    event_id: Optional[str] = None
    seq: Optional[int] = None
    event: Optional[TelegramIngressEvent] = None

    @property
    def ok(self) -> bool:
        return self.status in ("accepted", "duplicate")


@dataclass(frozen=True)
class RoomOutcome:
    """One terminal member turn. ``text`` is set only for a settled, non-pass reply."""

    seq: int
    status: Literal["settled", "passed", "failed", "cancelled", "deferred"]
    member_id: str
    profile: str
    source_event_id: str
    text: Optional[str]
    reason: Optional[str]


@dataclass(frozen=True)
class OutcomePage:
    enabled: bool
    reason: str
    outcomes: tuple
    cursor: int



@dataclass(frozen=True)
class PublicationResult:
    status: Literal["published", "duplicate", "disabled", "rejected", "failed"]
    reason: str = ""
    message_id: Optional[str] = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.status in ("published", "duplicate")


class _Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _room_protocol() -> Optional[int]:
    from gateway import hosted_rooms
    value = getattr(hosted_rooms, "PROTOCOL_VERSION", None)
    return value if _is_int(value) and value in SUPPORTED_ROOM_PROTOCOLS else None


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise _Rejected("unsupported_message")
    return text


def _origin_reason(message: Any, sender: Any) -> str:
    """First untrusted-origin marker on a message, or ``""``."""
    if getattr(message, "forward_origin", None) is not None or getattr(message, "forward_date", None):
        return "forwarded"
    if getattr(message, "sender_chat", None) is not None:
        return "sender_chat"
    if getattr(message, "via_bot", None) is not None:
        return "via_bot"
    return "bot_sender" if getattr(sender, "is_bot", True) is not False else ""


def _event_from_update(update: Any, binding: TelegramRoomBinding) -> TelegramIngressEvent:
    if getattr(update, "edited_message", None) is not None:
        raise _Rejected("edit_not_admitted")
    message = getattr(update, "message", None)
    chat, sender = getattr(message, "chat", None), getattr(message, "from_user", None)
    if message is None or chat is None or sender is None:
        raise _Rejected("unsupported_message")
    update_id, message_id = getattr(update, "update_id", None), getattr(message, "message_id", None)
    chat_id, sender_id = getattr(chat, "id", None), getattr(sender, "id", None)
    if not all(_is_int(v) and v != 0 for v in (update_id, message_id, chat_id, sender_id)) or message_id < 0:
        raise _Rejected("invalid_coordinates")
    if getattr(chat, "type", None) not in _GROUP_CHAT_TYPES:
        raise _Rejected("not_group_chat")
    if reason := _origin_reason(message, sender):
        raise _Rejected(reason)
    topic = getattr(message, "message_thread_id", None) if getattr(message, "is_topic_message", False) is True else None
    if chat_id != binding.chat_id:
        raise _Rejected("chat_mismatch")
    if topic != binding.thread_id:
        raise _Rejected("topic_mismatch")
    if sender_id not in binding.sender_ids:
        raise _Rejected("sender_not_allowed")
    return TelegramIngressEvent(update_id, chat_id, topic, message_id, sender_id, _message_text(message))


def _error_reason(exc: Exception) -> Optional[str]:
    """Map a room-layer refusal to a stable reason; ``None`` = unexpected failure."""
    from gateway import hosted_rooms
    from gateway.hosted_room_discussion import DiscussionValidationError
    for cls, reason in ((hosted_rooms.RoomNotFoundError, "room_not_found"),
                        (hosted_rooms.EventConflictError, "event_conflict"),
                        (hosted_rooms.AuthorityConflictError, "not_room_authority"),
                        (DiscussionValidationError, "invalid_event"),
                        (hosted_rooms.HostedRoomError, "invalid_event")):
        if isinstance(exc, cls):
            return reason
    return None


# -- publication ------------------------------------------------------------------------------

# Per gateway process: a restart forgets it, so a relay persists its own outcomes cursor.
_LEDGER_MAX = 4096
_IN_FLIGHT = object()
_UNCERTAIN = object()  # the send was cancelled mid-flight: it may have reached Telegram
_ledger_lock = threading.Lock()
_ledger: OrderedDict = OrderedDict()


def _ledger_claim(key: tuple) -> Optional[PublicationResult]:
    """``None`` = claimed for this caller; else the ``duplicate`` answer for an earlier claim."""
    with _ledger_lock:
        if key in _ledger:
            value = _ledger[key]
            if value is _IN_FLIGHT:
                return PublicationResult("duplicate", "in_flight")
            if value is _UNCERTAIN:
                return PublicationResult("duplicate", "delivery_uncertain")
            return PublicationResult("duplicate", "", value)
        _ledger[key] = _IN_FLIGHT
        # Evict only confirmed deliveries. Live and uncertain claims are retained for the process
        # lifetime: either may already reach Telegram and therefore must never become sendable again.
        evictable = [k for k, v in _ledger.items() if v is not _IN_FLIGHT and v is not _UNCERTAIN]
        for old in evictable[:max(0, len(_ledger) - _LEDGER_MAX)]:
            del _ledger[old]
    return None


def _ledger_settle(key: tuple, result: Optional[PublicationResult]) -> None:
    """Record a published id; a failed send releases the claim so a retry can publish. ``None`` = the
    caller was cancelled mid-send: keep the key so the possibly-delivered post is never repeated."""
    with _ledger_lock:
        if result is None:
            _ledger[key] = _UNCERTAIN
        elif result.status == "published":
            _ledger[key] = result.message_id
        else:
            _ledger.pop(key, None)


def _claim_reason(binding: Any, outcome: Any) -> str:
    if not isinstance(binding, TelegramRoomBinding) or not isinstance(outcome, RoomOutcome) \
            or not (_is_int(outcome.seq) and outcome.seq > 0):
        return "invalid_outcome"
    if outcome.status != "settled" or not isinstance(outcome.text, str) or not outcome.text:
        return "not_settled"
    return ""


def _terminal_payload(db_path: Any, binding: TelegramRoomBinding, seq: int, authority: str) -> dict:
    """Payload of the coordinator's own non-pass ``turn.settled`` at exactly ``seq``."""
    from gateway import hosted_rooms
    rows = hosted_rooms.read_events(db_path, room_id=binding.room_id, since_seq=seq - 1, limit=1).get("events") or []
    event = rows[0] if rows else {}
    payload, actor = event.get("payload") or {}, event.get("actor") or {}
    if event.get("seq") != seq or event.get("kind") != "turn.settled" or payload.get("passed") is not False \
            or (actor.get("kind"), actor.get("id")) != ("gateway", authority):
        raise _Rejected("not_settled")
    if payload.get("thread_id") != binding.room_thread_id:
        raise _Rejected("topic_mismatch")
    return payload


def _scan_log(db_path: Any, room_id: str, stop_seq: int, wanted: set) -> dict[str, dict]:
    """Events before ``stop_seq`` whose id is in ``wanted``; nothing else is kept."""
    from gateway import hosted_rooms
    found: dict[str, dict] = {}
    cursor = 0
    while cursor < stop_seq - 1 and len(found) < len(wanted):
        page = hosted_rooms.read_events(db_path, room_id=room_id, since_seq=cursor, limit=hosted_rooms.MAX_LOG_LIMIT)
        rows = page.get("events") or []
        for row in rows:
            if int(row["seq"]) >= stop_seq:
                return found
            if row.get("event_id") in wanted:
                found[row["event_id"]] = row
        advanced = int(page.get("cursor") or cursor)
        if not rows or advanced <= cursor:
            break
        cursor = advanced
    return found


def _check_source(source: Optional[dict], binding: TelegramRoomBinding) -> None:
    """Loop prevention: only rounds an allowlisted human started in this exact chat/topic publish."""
    source = source or {}
    actor_id = (source.get("actor") or {}).get("id")
    if source.get("kind") != "message.user" or actor_id not in {f"telegram:{s}" for s in binding.sender_ids} \
            or (source.get("payload") or {}).get("thread_id") != binding.room_thread_id:
        raise _Rejected("source_not_admitted")


def _member_text(message: Optional[dict], member_id: str, binding: TelegramRoomBinding) -> str:
    payload = (message or {}).get("payload") or {}
    if (message or {}).get("kind") != "message.member" or payload.get("member_id") != member_id \
            or payload.get("thread_id") != binding.room_thread_id or not isinstance(payload.get("text"), str):
        raise _Rejected("not_settled")
    return payload["text"]


def _verify_publication(db_path: Any, binding: TelegramRoomBinding, outcome: RoomOutcome) -> str:
    """Re-derive ``outcome`` from the room log; returns the member's reply text or raises."""
    from gateway import hosted_rooms
    room = hosted_rooms.room_state(db_path, room_id=binding.room_id)
    authority = room.get("authority_gateway_id")
    if not authority or authority != hosted_rooms.local_authority_gateway_id():
        raise _Rejected("not_room_authority")
    payload = _terminal_payload(db_path, binding, outcome.seq, authority)
    member_id = payload.get("member_id")
    profiles = {m.get("member_id"): m.get("profile") for m in room.get("members") or []}
    if member_id not in profiles or outcome.member_id != member_id or outcome.profile != profiles[member_id]:
        raise _Rejected("roster_mismatch")
    source_id = payload.get("discussion_event_id")
    if not isinstance(source_id, str) or not re.fullmatch(rf"tg:{binding.chat_id}:[1-9][0-9]*", source_id):
        raise _Rejected("chat_mismatch")
    message_id = payload.get("message_event_id")
    found = _scan_log(db_path, binding.room_id, outcome.seq, {source_id, message_id})
    _check_source(found.get(source_id), binding)
    text = _member_text(found.get(message_id), member_id, binding)
    derived = (outcome.seq, "settled", member_id, profiles[member_id], source_id, text)
    if derived != (outcome.seq, outcome.status, outcome.member_id, outcome.profile, outcome.source_event_id,
                   outcome.text):
        raise _Rejected("outcome_mismatch")
    return text


async def _deliver(runner: Any, adapter: Any, binding: TelegramRoomBinding, text: str) -> PublicationResult:
    # The strict-topic flag makes the Telegram adapter fail loudly on "thread not found" instead
    # of retrying without message_thread_id, which would post into the General topic.
    meta = None if binding.thread_id is None else {
        "thread_id": str(binding.thread_id), "telegram_dm_topic_created_for_send": True}
    loop = getattr(runner, "_gateway_loop", None)
    try:
        coro = adapter.send(str(binding.chat_id), text, metadata=meta)
        if loop is not None and loop.is_running() and not loop.is_closed() and loop is not asyncio.get_running_loop():
            result = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))
        else:
            result = await coro
    except Exception:
        logger.warning("hosted_rooms publish send raised for room %s", binding.room_id, exc_info=True)
        return PublicationResult("failed", "adapter_error", retryable=True)
    if not getattr(result, "success", False):
        return PublicationResult("failed", "send_failed", retryable=bool(getattr(result, "retryable", False)))
    message_id = getattr(result, "message_id", None)
    return PublicationResult("published", "", str(message_id) if message_id is not None else None)


class HostedRoomFacade:
    """Per-plugin facade; holds only the plugin id and its profile home."""

    def __init__(self, plugin_id: str, home: Optional[Path] = None) -> None:
        self._plugin_id = plugin_id
        self._home = home

    # -- gate ---------------------------------------------------------------------------------

    def _scope(self) -> Any:
        from hermes_cli.plugins_loader import _plugin_home_scope
        return _plugin_home_scope(self._home) if self._home is not None else nullcontext()

    def _granted(self, capability: str = CAPABILITY_ID) -> bool:
        try:
            from hermes_cli.config import load_config
            from hermes_cli.plugin_capabilities import plugin_capability_granted
            with self._scope():
                cfg = load_config() or {}
            return plugin_capability_granted(self._plugin_id, capability, config=cfg)
        except Exception:
            logger.debug("hosted_rooms capability check failed for %s", self._plugin_id, exc_info=True)
            return False

    def _gate(self) -> tuple[Any, str]:
        """``(service, "")`` when live, else ``(None, reason)``."""
        if not self._granted():
            return None, "capability_not_granted"
        try:
            from tui_gateway import methods_groups
            service = methods_groups.get_hosted_room_service()
        except Exception:  # boundary: a broken worker probe means disabled, never a raise
            logger.debug("hosted_rooms worker probe failed for %s", self._plugin_id, exc_info=True)
            service = None
        if service is None:
            return None, "hosted_rooms_unavailable"
        if _room_protocol() is None:
            return None, "incompatible_gateway"
        return service, ""

    def describe(self) -> FacadeStatus:
        """Version + capability discovery; call before relying on any other verb."""
        _service, reason = self._gate()
        publication = not reason and self._granted(PUBLISH_CAPABILITY_ID)
        return FacadeStatus(FACADE_VERSION, not reason, reason, _room_protocol(), publication, True)

    # -- ingress ------------------------------------------------------------------------------

    def _submit(self, binding: TelegramRoomBinding, update: Any) -> IngressResult:
        service, reason = self._gate()
        if service is None:
            return IngressResult("disabled", reason)
        if not isinstance(binding, TelegramRoomBinding):
            return IngressResult("rejected", "invalid_binding")
        try:
            event = _event_from_update(update, binding)
        except _Rejected as exc:
            return IngressResult("rejected", exc.reason)
        except Exception:
            logger.warning("hosted_rooms update validation failed for plugin %s", self._plugin_id, exc_info=True)
            return IngressResult("failed", "internal_error")
        event_id = f"tg:{event.chat_id}:{event.message_id}"
        try:
            row = service.send(
                room_id=binding.room_id, event_id=event_id, actor_id=f"telegram:{event.sender_id}",
                payload={"text": event.text, "thread_id": binding.room_thread_id})
        except Exception as exc:
            if (code := _error_reason(exc)) is not None:
                return IngressResult("rejected", code, event=event)
            logger.warning("hosted_rooms ingress failed for plugin %s", self._plugin_id, exc_info=True)
            return IngressResult("failed", "internal_error", event=event)
        status: IngressStatus = "duplicate" if row.get("idempotent") else "accepted"
        return IngressResult(status, "", event_id, int(row["seq"]), event)

    async def submit_telegram_update(self, binding: TelegramRoomBinding, update: Any) -> IngressResult:
        """Admit one adapter-delivered PTB ``Update`` into ``binding.room_id``.

        Idempotent per Telegram message: a redelivery (or a retry after the caller was cancelled
        mid-call) returns ``duplicate``. Cancelling the await does not roll back an append that
        already started."""
        return await asyncio.to_thread(self._submit, binding, update)

    # -- read back ----------------------------------------------------------------------------

    def outcomes(self, binding: TelegramRoomBinding, since_seq: int = 0) -> OutcomePage:
        """Terminal member turns in this binding's thread after ``since_seq``, attributed from the
        room roster. Pass the returned ``cursor`` back; it never skips a reply awaiting settlement."""
        service, reason = self._gate()
        if service is None:
            return OutcomePage(False, reason, (), since_seq)
        try:
            return self._read_outcomes(service, binding, since_seq)
        except Exception as exc:
            code = _error_reason(exc)
            if code is None:
                logger.warning("hosted_rooms outcomes failed for plugin %s", self._plugin_id, exc_info=True)
            return OutcomePage(True, code or "internal_error", (), since_seq)

    def _read_outcomes(self, service: Any, binding: TelegramRoomBinding, since_seq: int) -> OutcomePage:
        from gateway import hosted_rooms
        room = hosted_rooms.room_state(service.db_path, room_id=binding.room_id)
        profiles = {str(m.get("member_id")): str(m.get("profile")) for m in room.get("members") or []}
        page = hosted_rooms.read_events(
            service.db_path, room_id=binding.room_id, since_seq=since_seq, limit=hosted_rooms.MAX_LOG_LIMIT)
        pending: dict[str, dict] = {}
        outcomes: list[RoomOutcome] = []
        for row in page.get("events") or []:
            payload = row.get("payload") or {}
            if payload.get("thread_id") != binding.room_thread_id:
                continue
            if row["kind"] == "message.member":
                pending[row["event_id"]] = row
                continue
            status = _OUTCOME_KINDS.get(row["kind"])
            if status is None:
                continue
            reply = pending.pop(payload.get("message_event_id") or "", None)
            if status == "settled" and payload.get("passed"):
                status = "passed"
            member_id = str(payload.get("member_id"))
            outcomes.append(RoomOutcome(
                seq=int(row["seq"]), status=status, member_id=member_id,
                profile=profiles.get(member_id, ""), source_event_id=str(payload.get("discussion_event_id")),
                text=reply["payload"]["text"] if reply else None,
                reason=payload.get("reason") or payload.get("error")))
        cursor = int(page.get("cursor") or since_seq)
        if pending:
            cursor = min(int(row["seq"]) for row in pending.values()) - 1
        return OutcomePage(True, "", tuple(outcomes), cursor)

    # -- publication --------------------------------------------------------------------------

    def _telegram_adapter(self) -> tuple[Any, Any]:
        """``(runner, adapter)`` for this plugin's profile's connected Telegram bot, else
        ``(None, None)``. Profile-aware and fail-closed like ``platform_actions``: no fallback to
        ``runner.adapters``, which would act through the default profile's bot."""
        try:
            from gateway.config import Platform
            from gateway.run import _gateway_runner_ref
            from hermes_cli.profiles import get_active_profile_name
            runner = _gateway_runner_ref()
            resolve = getattr(runner, "_authorization_adapter", None)
            if runner is None or not callable(resolve):
                return None, None
            with self._scope():
                profile = get_active_profile_name()
            adapter = resolve(Platform.TELEGRAM, profile)
            return (runner, adapter) if adapter is not None and bool(adapter.is_connected) else (None, None)
        except Exception:
            logger.debug("hosted_rooms adapter lookup failed for %s", self._plugin_id, exc_info=True)
            return None, None

    async def _publish(self, binding: Any, outcome: Any, facade_version: Any,
                       label: Any = None) -> PublicationResult:
        if not (_is_int(facade_version) and facade_version == FACADE_VERSION):
            return PublicationResult("disabled", "incompatible_facade")
        service, reason = self._gate()
        if service is None:
            return PublicationResult("disabled", reason)
        if not self._granted(PUBLISH_CAPABILITY_ID):
            return PublicationResult("disabled", "capability_not_granted")
        if reason := _claim_reason(binding, outcome):
            return PublicationResult("rejected", reason)
        display_label = _publication_label(label, outcome.profile)
        if display_label is None:
            return PublicationResult("rejected", "invalid_label")
        runner, adapter = self._telegram_adapter()
        if adapter is None:
            return PublicationResult("disabled", "adapter_unavailable")
        try:
            member_text = await asyncio.to_thread(_verify_publication, service.db_path, binding, outcome)
        except _Rejected as exc:
            return PublicationResult("rejected", exc.reason)
        except Exception as exc:
            if (code := _error_reason(exc)) is None:
                raise
            return PublicationResult("rejected", code)
        text = f"[{display_label}] {member_text}"
        if len(text) > MAX_PUBLICATION_CHARS:
            return PublicationResult("rejected", "text_too_long")
        key = (str(service.db_path), binding.room_id, outcome.seq, binding.chat_id, binding.thread_id)
        if (duplicate := _ledger_claim(key)) is not None:
            return duplicate
        try:
            result = await _deliver(runner, adapter, binding, text)
        except BaseException:  # cancelled mid-send (``_deliver`` itself never raises an Exception)
            _ledger_settle(key, None)
            raise
        _ledger_settle(key, result)
        return result

    async def publish_outcome(self, binding: TelegramRoomBinding, outcome: RoomOutcome, *,
                              facade_version: int, label: Optional[str] = None) -> PublicationResult:
        """Post one settled reply to ``binding``'s chat/topic as the gateway bot.

        The default wire form is ``[profile] text``. ``label`` may replace only that bracketed display
        label; it is bounded and control/bracket-free, while attribution is still re-derived from the
        room log using ``outcome.profile``.

        ``outcome`` is only a claim: it is re-verified against this gateway's room log (this gateway
        must be the room's coordinator, the round must have been started by an allowlisted Telegram
        sender in this exact chat/topic). Writes nothing to any room store. Idempotent per
        (room, seq, target) within this gateway process; never raises."""
        try:
            result = await self._publish(binding, outcome, facade_version, label)
        except Exception:
            logger.warning("hosted_rooms publish failed for plugin %s", self._plugin_id, exc_info=True)
            result = PublicationResult("failed", "internal_error")
        logger.info("hosted_rooms publish plugin=%s room=%s seq=%s status=%s reason=%s", self._plugin_id,
                    getattr(binding, "room_id", None), getattr(outcome, "seq", None), result.status, result.reason)
        return result
