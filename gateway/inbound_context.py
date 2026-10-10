"""Refreshable external context for a new gateway input."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from gateway.platforms.event import MessageEvent, TurnContextUpdate
from gateway.session import SessionSource

_T = TypeVar("_T")

_LOOP_CALL_TIMEOUT_SECONDS = 10.0


class InboundContextSnapshot(Protocol):
    def use_turn_context(self, update: TurnContextUpdate | None) -> None: ...

    async def refresh(self) -> None: ...

    def prepend_turn_context(self, text: str) -> str: ...

    def reply_event(self, event: MessageEvent) -> MessageEvent: ...

    def reply_image_paths(self) -> list[str]: ...

    def media_event(self, event: MessageEvent) -> MessageEvent:
        """Return *event* with only the authored attachments that are still current."""
        ...

    def authored_text(self, text: str) -> str:
        """Return the prepared message *text* with each authored contribution replaced by its
        current text."""
        ...


@dataclass(frozen=True)
class ImageEnrichment:
    path: str
    text: str

    @classmethod
    async def enrich_each(
        cls, runner: Any, source: SessionSource, session_key: str, paths: list[str]
    ) -> tuple[ImageEnrichment, ...]:
        """Enrich each image separately so that withdrawing one image removes only its own
        description. The session's native image buffer keeps its paths and adds each path
        routed natively here."""
        native_images = runner._consume_pending_native_image_paths(session_key)
        enrichments = []
        for path in paths:
            text = await runner._enrich_inbound_images(source, session_key, "", [path])
            enrichments.append(cls(path, text))
            native_images.extend(runner._consume_pending_native_image_paths(session_key))
        state = runner._peek_session_state(session_key)
        if state is not None:
            state.persistent.native_image_paths = list(dict.fromkeys(native_images))
        return tuple(enrichments)


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
    quoted_images: tuple[ImageEnrichment, ...] = ()
    authored_images: tuple[ImageEnrichment, ...] = ()
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
        current_authored = self.snapshot.media_event(self.event).media_urls
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
        text = self.snapshot.authored_text(self.text)
        authored = self.snapshot.media_event(self.event).media_urls
        quoted = self.snapshot.reply_image_paths()
        descriptions = [
            *(image.text for image in self.authored_images if image.path in authored and image.text),
            *(image.text for image in self.quoted_images if image.path in quoted and image.text),
        ]
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
