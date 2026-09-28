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


@dataclass(frozen=True)
class QuotedImageEnrichment:
    path: str
    text: str


@dataclass
class PreparedInboundMessage:
    snapshot: InboundContextSnapshot
    event: MessageEvent
    text: str
    channel_context: str | None = None
    quoted_images: tuple[QuotedImageEnrichment, ...] = ()
    message_text: str | None = None
    persist_user_message: str | None = None
    persist_user_timestamp: float | None = None

    def retained_image_paths(self, paths: list[str]) -> list[str]:
        current = self.snapshot.reply_image_paths()
        authored = self.event.authored_media().media_urls
        quoted = {image.path for image in self.quoted_images}
        return [
            path
            for path in paths
            if path not in quoted or path in current or path in authored
        ]

    def revalidate_native_input(
        self, runner: Any, message: str, paths: list[str]
    ) -> tuple[str, list[str]]:
        previous = self.message_text
        current = self.render(runner, timestamps=True)
        if previous is not None and previous in message:
            current = message.replace(previous, current, 1)
        return current, self.retained_image_paths(paths)

    def render(self, runner: Any, *, timestamps: bool = False) -> str:
        text = self.text
        current = self.snapshot.reply_image_paths()
        descriptions = [
            image.text
            for image in self.quoted_images
            if image.path in current and image.text
        ]
        if descriptions:
            text = "\n\n".join([*descriptions, text])
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
