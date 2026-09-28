"""Refreshable external context for a new gateway input."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from gateway.platforms.event import MessageEvent, TurnContextUpdate

_T = TypeVar("_T")

_LOOP_CALL_TIMEOUT_SECONDS = 10.0


class InboundContextSnapshot(Protocol):
    def use_turn_context(self, update: TurnContextUpdate | None) -> None: ...

    async def refresh(self) -> None: ...

    def prepend_turn_context(self, text: str) -> str: ...

    def reply_event(self, event: MessageEvent) -> MessageEvent: ...

    def reply_image_paths(self) -> list[str]: ...


@dataclass(frozen=True)
class QuotedImageEnrichment:
    path: str
    text: str


@dataclass(frozen=True)
class AuthoredImageEnrichment:
    paths: tuple[str, ...]
    text: str


@dataclass
class PreparedInboundMessage:
    """A new input whose external context is rendered again before model use.

    The snapshot reads platform state that belongs to the event loop that
    prepared the input. Calls from any other thread run on that loop.
    """

    snapshot: InboundContextSnapshot
    event: MessageEvent
    text: str
    redact_pii: bool = False
    quoted_images: tuple[QuotedImageEnrichment, ...] = ()
    authored_images: AuthoredImageEnrichment | None = None
    message_text: str | None = None
    persist_user_message: str | None = None
    persist_user_timestamp: float | None = None
    loop: asyncio.AbstractEventLoop = field(
        default_factory=asyncio.get_running_loop, repr=False, compare=False
    )

    def retained_image_paths(self, paths: list[str]) -> list[str]:
        return self._on_loop(self._retained_image_paths, paths)

    def revalidate_native_input(
        self, runner: Any, message: str, paths: list[str]
    ) -> tuple[str, list[str]]:
        return self._on_loop(self._revalidate_native_input, runner, message, paths)

    def render(self, runner: Any, *, timestamps: bool = False) -> str:
        return self._on_loop(self._render, runner, timestamps)

    def _on_loop(self, call: Callable[..., _T], *args: Any) -> _T:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop or not self.loop.is_running():
            return call(*args)

        async def run() -> _T:
            return call(*args)

        future = asyncio.run_coroutine_threadsafe(run(), self.loop)
        try:
            return future.result(timeout=_LOOP_CALL_TIMEOUT_SECONDS)
        except TimeoutError:
            future.cancel()
            raise

    def _retained_image_paths(self, paths: list[str]) -> list[str]:
        current = self.snapshot.reply_image_paths()
        authored = self.event.authored_media().media_urls
        media_event = getattr(self.snapshot, "media_event", None)
        current_authored = (
            media_event(self.event).authored_media().media_urls
            if callable(media_event)
            else authored
        )
        quoted = {image.path for image in self.quoted_images}
        return [
            path
            for path in paths
            if (path not in quoted or path in current or path in current_authored)
            and (path not in authored or path in current_authored or path in current)
        ]

    def _revalidate_native_input(
        self, runner: Any, message: str, paths: list[str]
    ) -> tuple[str, list[str]]:
        previous = self.message_text
        current = self._render(runner, True)
        if previous is not None and previous in message:
            current = message.replace(previous, current, 1)
        return current, self._retained_image_paths(paths)

    def _render(self, runner: Any, timestamps: bool) -> str:
        text = self.text
        current = self.snapshot.reply_image_paths()
        descriptions = [
            image.text
            for image in self.quoted_images
            if image.path in current and image.text
        ]
        media_event = getattr(self.snapshot, "media_event", None)
        if self.authored_images is not None and callable(media_event):
            authored = media_event(self.event).authored_media().media_urls
            if self.authored_images.text and all(
                path in authored for path in self.authored_images.paths
            ):
                descriptions.insert(0, self.authored_images.text)
        if descriptions:
            text = "\n\n".join([*descriptions, text])
        reply = self.snapshot.reply_event(self.event)
        text = runner._prepend_inbound_reply_context(
            reply, self.event.source, text, redact_pii=self.redact_pii,
        )
        text = self.snapshot.prepend_turn_context(text)
        if timestamps:
            text, self.persist_user_message, self.persist_user_timestamp = (
                runner._hmwa_apply_message_timestamp(self.event, text)
            )
        self.message_text = text
        return text
