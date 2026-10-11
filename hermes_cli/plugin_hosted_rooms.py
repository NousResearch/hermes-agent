"""``ctx.hosted_rooms``: the plugin facade between a Telegram platform handler and the gateway's
hosted Group Chat rooms (``HostedRoomService``).

Disabled-first and fail-closed. Every verb re-checks, at call time and in this order: the
``gateway.hosted_room_ingress`` capability (read from the plugin's own profile config), a running
hosted-room worker in this process, and a room protocol this facade was written against. Any miss
returns a typed ``disabled`` result; no verb raises into PTB handler dispatch, so ordinary Telegram
handling continues unchanged. Coordinates are derived from the adapter-delivered ``Update`` here,
never accepted from the plugin. There is deliberately no publication verb.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

FACADE_VERSION = 1
CAPABILITY_ID = "gateway.hosted_room_ingress"
SUPPORTED_ROOM_PROTOCOLS = frozenset({2})
_GROUP_CHAT_TYPES = frozenset({"group", "supergroup"})
_OUTCOME_KINDS = {"turn.settled": "settled", "turn.failed": "failed",
                  "turn.cancelled": "cancelled", "turn.deferred": "deferred"}

IngressStatus = Literal["accepted", "duplicate", "disabled", "rejected", "failed"]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
class StopResult:
    status: Literal["stopped", "disabled", "rejected", "failed"]
    reason: str = ""
    tasks: int = 0


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


class HostedRoomFacade:
    """Per-plugin facade; holds only the plugin id and its profile home."""

    def __init__(self, plugin_id: str, home: Optional[Path] = None) -> None:
        self._plugin_id = plugin_id
        self._home = home

    # -- gate ---------------------------------------------------------------------------------

    def _granted(self) -> bool:
        try:
            from hermes_cli.config import load_config
            from hermes_cli.plugin_capabilities import plugin_capability_granted
            from hermes_cli.plugins_loader import _plugin_home_scope
            with _plugin_home_scope(self._home) if self._home is not None else nullcontext():
                cfg = load_config() or {}
            return plugin_capability_granted(self._plugin_id, CAPABILITY_ID, config=cfg)
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
        return FacadeStatus(FACADE_VERSION, not reason, reason, _room_protocol())

    # -- ingress ------------------------------------------------------------------------------

    def _submit(self, binding: TelegramRoomBinding, update: Any) -> IngressResult:
        service, reason = self._gate()
        if service is None:
            return IngressResult("disabled", reason)
        try:
            event = _event_from_update(update, binding)
        except _Rejected as exc:
            return IngressResult("rejected", exc.reason)
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

    # -- cancellation -------------------------------------------------------------------------

    def stop(self, binding: TelegramRoomBinding, *, cancel_id: str) -> StopResult:
        """Kill switch: fence the room and cancel its queued/running turns."""
        service, reason = self._gate()
        if service is None:
            return StopResult("disabled", reason)
        try:
            return StopResult("stopped", "", int(service.stop_room(binding.room_id, cancel_id=cancel_id)))
        except Exception as exc:
            if (code := _error_reason(exc)) is not None:
                return StopResult("rejected", code)
            logger.warning("hosted_rooms stop failed for plugin %s", self._plugin_id, exc_info=True)
            return StopResult("failed", "internal_error")
