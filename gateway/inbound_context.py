"""Refreshable external context for a new gateway input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from gateway.platforms.event import MessageEvent


class InboundContextSnapshot(Protocol):
    async def refresh(self) -> None: ...

    def prepend_history(self, text: str) -> str: ...

    def reply_event(self, event: MessageEvent) -> MessageEvent: ...

    def reply_image_paths(self) -> list[str]: ...


@dataclass
class PreparedInboundMessage:
    snapshot: InboundContextSnapshot
    event: MessageEvent
    text: str
    channel_context: str | None = None
    quoted_image_text: str = ""
    quoted_image_paths: tuple[str, ...] = ()
    message_text: str | None = None
    persist_user_message: str | None = None
    persist_user_timestamp: float | None = None

    def retained_image_paths(self, paths: list[str]) -> list[str]:
        current = self.snapshot.reply_image_paths()
        return [
            path
            for path in paths
            if path not in self.quoted_image_paths or path in current
        ]

    def render(self, runner: Any, *, timestamps: bool = False) -> str:
        text = self.text
        if (
            self.quoted_image_text
            and list(self.quoted_image_paths) == self.snapshot.reply_image_paths()
        ):
            text = f"{self.quoted_image_text}\n\n{text}"
        text = self.snapshot.prepend_history(text)
        reply = self.snapshot.reply_event(self.event)
        text = runner._prepend_inbound_reply_context(reply, self.event.source, text)
        if self.channel_context:
            text = f"{self.channel_context}\n\n[New message]\n{text}"
        if timestamps:
            text, self.persist_user_message, self.persist_user_timestamp = (
                runner._hmwa_apply_message_timestamp(self.event, text)
            )
        self.message_text = text
        return text
