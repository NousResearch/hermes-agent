"""Wyoming `handle` TCP server — Home Assistant conversation-agent endpoint.

Pure protocol layer: knows wyoming events and nothing about Hermes. The
adapter injects ``on_transcript(text, context, respond)``; ``respond`` is
single-use and returns True only for the call that actually wrote a frame,
so callers can race an ack against the final reply safely.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from wyoming.asr import Transcript
from wyoming.event import Event
from wyoming.handle import Handled, NotHandled
from wyoming.info import Attribution, Describe, HandleModel, HandleProgram, Info
from wyoming.server import AsyncEventHandler

logger = logging.getLogger(__name__)

_ATTRIBUTION = Attribution(
    name="Hermes Agent", url="https://github.com/NousResearch/hermes-agent"
)

RespondFn = Callable[[Optional[str]], Awaitable[bool]]
TranscriptCallback = Callable[[str, Dict[str, Any], RespondFn], Awaitable[None]]


def build_info(*, supports_home_control: bool) -> Info:
    """Info advertised on Describe. Empty model languages => HA MATCH_ALL."""
    return Info(
        handle=[
            HandleProgram(
                name="hermes",
                description="Hermes Agent conversation handler",
                attribution=_ATTRIBUTION,
                installed=True,
                version="1.0",
                supports_home_control=supports_home_control,
                models=[
                    HandleModel(
                        name="hermes",
                        description="Hermes Agent",
                        attribution=_ATTRIBUTION,
                        installed=True,
                        version="1.0",
                        languages=[],
                    )
                ],
            )
        ]
    )


class _Handler(AsyncEventHandler):
    def __init__(self, reader, writer, *, info: Info, on_transcript: TranscriptCallback):
        super().__init__(reader, writer)
        self._info = info
        self._on_transcript = on_transcript

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True
        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            responded = False

            async def respond(text: Optional[str]) -> bool:
                nonlocal responded
                if responded:
                    return False
                responded = True
                frame = NotHandled() if text is None else Handled(text=text)
                try:
                    await self.write_event(frame.event())
                except (ConnectionError, OSError):
                    # HA gave up (pipeline timeout/restart): reply undeliverable.
                    return False
                return True

            try:
                await self._on_transcript(
                    transcript.text or "", transcript.context or {}, respond
                )
            finally:
                if not responded:
                    # The callback must always answer; never leave HA hanging.
                    await respond(None)
            return False  # one utterance per connection; HA reconnects per turn
        # Unsupported event types (ping, audio, …): ignore, keep the connection.
        return True


class HandleServer:
    """Owns the TCP listener; one _Handler per inbound HA connection."""

    def __init__(
        self,
        bind_host: str,
        port: int,
        *,
        on_transcript: TranscriptCallback,
        supports_home_control: bool = False,
    ):
        self._bind_host = bind_host
        self._requested_port = int(port)
        self._info = build_info(supports_home_control=supports_home_control)
        self._on_transcript = on_transcript
        self._server: Optional[asyncio.AbstractServer] = None
        self.port: int = self._requested_port

    async def start(self) -> None:
        """Bind and serve. Raises OSError if the port cannot be bound."""

        async def _client_connected(reader, writer):
            handler = _Handler(
                reader, writer, info=self._info, on_transcript=self._on_transcript
            )
            try:
                await handler.run()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass  # client vanished mid-frame; per-connection, nothing to reset
            except Exception:
                logger.exception("[ha_conversation] connection handler failed")
            finally:
                writer.close()

        self._server = await asyncio.start_server(
            _client_connected, self._bind_host, self._requested_port
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
