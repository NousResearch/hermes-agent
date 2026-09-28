"""Create visible Matrix threads through an admitted live session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

from plugins.platforms.matrix.client_events import raw_event
from plugins.platforms.matrix.read_context import (
    MatrixSessionAccess,
    MatrixSessionError,
)

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

# A reverse proxy can return these statuses after forwarding the request, so the
# homeserver may already have accepted the event.
_PROXY_UNCERTAIN_STATUSES = frozenset({502, 503, 504})


@dataclass(frozen=True)
class MatrixThreadDelivery:
    room_id: str
    root_event_id: str | None = None
    initial_reply_event_id: str | None = None
    root_delivered: bool = False
    sending: bool = False

    def failure(
        self, error: str, *, delivery_uncertain: bool = False
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"success": False, "error": error}
        if self.root_event_id:
            result.update(room_id=self.room_id, root_event_id=self.root_event_id)
        if self.root_delivered or self.initial_reply_event_id:
            result["partial"] = True
        if self.initial_reply_event_id:
            result["initial_reply_event_id"] = self.initial_reply_event_id
        if delivery_uncertain:
            result["delivery_uncertain"] = True
        return result


@dataclass
class MatrixThreadCreateProgress:
    room_id: str
    delivery: MatrixThreadDelivery = field(init=False)

    def __post_init__(self) -> None:
        self.delivery = MatrixThreadDelivery(self.room_id)

    def record(
        self,
        root: str | None,
        initial_reply: str | None,
        root_delivered: bool,
        sending: bool = False,
    ) -> None:
        self.delivery = MatrixThreadDelivery(
            self.room_id, root, initial_reply, root_delivered, sending
        )

    def mark_sending(self) -> None:
        self.delivery = replace(self.delivery, sending=True)


class MatrixThreadCreateMixin:
    async def create_matrix_thread(
        self: MatrixAdapter,
        room_id: str,
        message: str,
        *,
        requester: str,
        root_text: str | None,
        root_event_id: str | None,
        interrupted: Callable[[], bool],
        progress: MatrixThreadCreateProgress | None = None,
    ) -> dict[str, Any]:
        """Return confirmed event IDs after posting the first reply to a main-timeline root."""
        root = None
        initial_reply = None
        root_delivered = False
        phase = "root"
        if progress is None:
            progress = MatrixThreadCreateProgress(room_id)
        try:
            access = MatrixSessionAccess.capture(
                self, room_id, requester, interrupted=interrupted
            )
            await access.admit()
            self._check_thread_interrupt(interrupted)
            if root_event_id:
                raw = raw_event(
                    await asyncio.wait_for(
                        access.client.get_event(room_id, root_event_id), timeout=10.0
                    )
                )
                access.check()
                self._check_thread_interrupt(interrupted)
                if (
                    raw.get("event_id") != root_event_id
                    or raw.get("room_id") != room_id
                ):
                    raise MatrixSessionError(
                        "Matrix root is missing or belongs to another room"
                    )
                if "redacted_because" in (raw.get("unsigned") or {}):
                    raise MatrixSessionError("Matrix root was withdrawn")
                event = await access.decrypt(raw)
                self._check_thread_interrupt(interrupted)
                if (
                    event.get("event_id") != root_event_id
                    or event.get("room_id") != room_id
                ):
                    raise MatrixSessionError(
                        "Matrix root is missing or belongs to another room"
                    )
                content = event.get("content") or {}
                relation = content.get("m.relates_to") or {}
                if (
                    event.get("type") != "m.room.message"
                    or not content.get("msgtype")
                    or relation.get("rel_type")
                    or "redacted_because" in (event.get("unsigned") or {})
                ):
                    raise MatrixSessionError(
                        "Matrix root must be an eligible main-timeline message"
                    )
                root = root_event_id
                progress.record(root, initial_reply, root_delivered)
            else:
                if root_text is None:
                    raise MatrixSessionError("root_text is required for a new root")
                formatted = self.format_message(root_text)
                if self.message_len_fn(formatted) > self.max_message_length:
                    raise MatrixSessionError(
                        "root_text exceeds the Matrix message length limit"
                    )
                self._check_thread_interrupt(interrupted)
                access.check()
                root = await self._send_room_message(
                    room_id,
                    self._build_text_message_content(formatted),
                    access=access,
                    before_request=progress.mark_sending,
                )
                root_delivered = True
                progress.record(root, initial_reply, root_delivered)

            phase = "reply"
            for chunk in self.truncate_message(
                self.format_message(message),
                self.max_message_length,
                len_fn=self.message_len_fn,
            ):
                await access.admit()
                self._check_thread_interrupt(interrupted)
                content = self._build_text_message_content(chunk)
                self._apply_relation_metadata(
                    room_id,
                    content,
                    metadata={
                        "thread_id": root,
                        "matrix_thread_fallback_event_id": root
                        if initial_reply is None
                        else "",
                    },
                )
                event_id = await self._send_room_message(
                    room_id,
                    content,
                    access=access,
                    before_request=progress.mark_sending,
                )
                progress.record(root, initial_reply or event_id, root_delivered)
                if initial_reply is not None:
                    continue
                initial_reply = event_id
                access.check(event_id)
                home_token = set_hermes_home_override(str(access.participation_home))
                try:
                    await self._threads.mark_async(root)
                finally:
                    reset_hermes_home_override(home_token)
                access.check(event_id)
            if initial_reply is None:
                raise MatrixSessionError(
                    "Matrix initial thread reply was not delivered"
                )
            return {
                "success": True,
                "room_id": room_id,
                "root_event_id": root,
                "initial_reply_event_id": initial_reply,
            }
        except (Exception, asyncio.CancelledError) as exc:
            sending = progress.delivery.sending
            if isinstance(exc, MatrixSessionError) and exc.event_id:
                if phase == "root":
                    root, root_delivered = exc.event_id, True
                elif initial_reply is None:
                    initial_reply = exc.event_id
                sending = False
            progress.record(root, initial_reply, root_delivered, sending)
            return progress.delivery.failure(
                "Matrix thread creation cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else f"{type(exc).__name__}: {exc}"
                if not isinstance(exc, MatrixSessionError)
                else str(exc),
                delivery_uncertain=sending and _outcome_unknown(exc),
            )

    @staticmethod
    def _check_thread_interrupt(interrupted: Callable[[], bool]) -> None:
        if interrupted():
            raise MatrixSessionError("Matrix thread creation interrupted")


def _outcome_unknown(exc: BaseException) -> bool:
    from mautrix.errors import MatrixConnectionError, MatrixRequestError

    if isinstance(exc, MatrixRequestError):
        return exc.http_status in _PROXY_UNCERTAIN_STATUSES
    return isinstance(
        exc, (asyncio.CancelledError, TimeoutError, OSError, MatrixConnectionError)
    )
