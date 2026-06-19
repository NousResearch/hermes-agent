"""Bridge WebSocket server — the seam the .NET media worker dials into.

Unlike a typical client/gateway split, here the **worker is the WS client** and
this driver is the **server** (it binds and waits). Each Teams call opens one
connection to ``{path}/{callId}``; the worker authenticates the upgrade with HMAC
headers, sends ``session.start``, then streams inbound media while this side
streams TTS audio + avatar driver cues back.

This module owns transport concerns only: the HMAC handshake, connection caps,
the pre-start timeout, the read/dispatch loop, and ping/pong. Dialogue/perception
logic lives behind a :class:`CallSessionHandler` so the realtime/streaming brain
can be wired in without touching the wire layer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from aiohttp import WSMsgType, web

from . import hmac_auth, protocol
from .config import HEADER_SIGNATURE, HEADER_TIMESTAMP, TeamsVoiceConfig, resolve_config

logger = logging.getLogger(__name__)

_MAX_FRAME_BYTES = 2 * 1024 * 1024  # 2 MB — accommodates a base64 JPEG video.frame


class CallSession:
    """One live Teams call: the WebSocket plus typed send helpers.

    Inbound frames are delivered to the bound :class:`CallSessionHandler`; the
    handler (and the dialogue brain) drives the call back via the ``send_*``
    methods, which serialize the outbound protocol builders.
    """

    def __init__(self, call_id: str, ws: web.WebSocketResponse) -> None:
        self.call_id = call_id
        self._ws = ws
        self.recording_active = False
        self.human_count = 0

    @property
    def closed(self) -> bool:
        return self._ws.closed

    async def _send(self, msg: dict) -> None:
        if self._ws.closed:
            return
        try:
            await self._ws.send_str(protocol.encode(msg))
        except (ConnectionError, RuntimeError) as exc:
            # A send failure means the call is gone; surface it rather than
            # silently "succeeding" on a dead socket.
            logger.warning("[teams_voice] send failed on %s: %s", self.call_id, exc)
            raise

    async def send_audio_frame(self, seq: int, timestamp_ms: int, payload_base64: str) -> None:
        await self._send(protocol.audio_frame(seq, timestamp_ms, payload_base64))

    async def send_expression(self, emotion: str) -> None:
        await self._send(protocol.expression(emotion))

    async def send_speech_marks(self, marks: list[dict[str, int]], ts: int = 0) -> None:
        await self._send(protocol.speech_marks(marks, ts))

    async def send_display_image(self, data_base64: str, mime: str, **kwargs) -> None:
        await self._send(protocol.display_image(data_base64, mime, **kwargs))

    async def send_assistant_cancel(self, turn_id: int) -> None:
        await self._send(protocol.assistant_cancel(turn_id))


class CallSessionHandler:
    """Interface the dialogue/perception brain implements.

    Every method is async and best-effort: a handler exception is logged and the
    call continues (a bad frame must not tear down the socket). The default
    implementation just logs — wire a real handler via :class:`BridgeServer`.
    """

    async def on_session_start(self, session: CallSession, msg: protocol.SessionStart) -> None:
        logger.info(
            "[teams_voice] session.start call=%s thread=%s dir=%s caller=%s",
            msg.call_id, msg.thread_id, msg.direction, msg.caller.display_name,
        )

    async def on_audio_frame(self, session: CallSession, msg: protocol.AudioFrame) -> None:
        ...  # TODO: gate on recording, echo-guard, route to STT/realtime model

    async def on_video_frame(self, session: CallSession, msg: protocol.VideoFrame) -> None:
        ...  # TODO: gate on recording, store in vision ring, ambient push

    async def on_recording_status(self, session: CallSession, msg: protocol.RecordingStatus) -> None:
        session.recording_active = msg.status == "active"
        logger.info("[teams_voice] recording.status call=%s = %s", session.call_id, msg.status)

    async def on_participants(self, session: CallSession, msg: protocol.Participants) -> None:
        session.human_count = msg.count

    async def on_dtmf(self, session: CallSession, msg: protocol.Dtmf) -> None:
        ...  # TODO: surface keypress to the realtime model (recording-gated)

    async def on_session_end(self, session: CallSession, msg: protocol.SessionEnd) -> None:
        logger.info("[teams_voice] session.end call=%s reason=%s", session.call_id, msg.reason)


HandlerFactory = Callable[[], CallSessionHandler]


class BridgeServer:
    """Hosts the HMAC-authenticated WebSocket the media worker connects to."""

    def __init__(
        self,
        config: Optional[TeamsVoiceConfig] = None,
        handler_factory: Optional[HandlerFactory] = None,
    ) -> None:
        self.config = config or resolve_config()
        self._handler_factory = handler_factory or CallSessionHandler
        self._replay = hmac_auth.ReplayGuard(window_ms=self.config.hmac_window_ms)
        self._runner: Optional[web.AppRunner] = None
        self._conn_count = 0
        self._conn_by_ip: dict[str, int] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bind and start serving. Raises if no shared secret is configured."""
        if not self.config.configured:
            raise RuntimeError(
                "teams_voice bridge has no shared secret "
                "(set TEAMS_VOICE_SHARED_SECRET or config.extra.shared_secret)"
            )
        app = web.Application()
        route = f"{self.config.path.rstrip('/')}/{{call_id}}"
        app.router.add_get(route, self._handle_ws)
        app.router.add_get("/health", lambda _req: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()
        logger.info(
            "[teams_voice] bridge listening host=%s port=%s path=%s",
            self.config.host, self.config.port, route,
        )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # ── connection handling ──────────────────────────────────────────────────

    async def _handle_ws(self, request: web.Request) -> web.StreamResponse:
        call_id = request.match_info.get("call_id", "").strip()
        if not call_id:
            return web.Response(status=400, text="missing callId")

        ok, reason = hmac_auth.verify_upgrade(
            secret=self.config.shared_secret,
            call_id=call_id,
            timestamp_header=request.headers.get(HEADER_TIMESTAMP),
            signature_header=request.headers.get(HEADER_SIGNATURE),
            window_ms=self.config.hmac_window_ms,
            replay_guard=self._replay,
        )
        if not ok:
            logger.warning("[teams_voice] upgrade rejected call=%s: %s", call_id, reason)
            return web.Response(status=401, text="unauthorized")

        peer_ip = request.remote or "?"
        if self._conn_count >= self.config.max_connections:
            return web.Response(status=503, text="too many connections")
        if self._conn_by_ip.get(peer_ip, 0) >= self.config.max_connections_per_ip:
            return web.Response(status=503, text="too many connections")

        ws = web.WebSocketResponse(max_msg_size=_MAX_FRAME_BYTES, heartbeat=None)
        await ws.prepare(request)

        self._conn_count += 1
        self._conn_by_ip[peer_ip] = self._conn_by_ip.get(peer_ip, 0) + 1
        session = CallSession(call_id, ws)
        handler = self._handler_factory()
        try:
            await self._read_loop(session, handler)
        finally:
            self._conn_count -= 1
            self._conn_by_ip[peer_ip] = max(0, self._conn_by_ip.get(peer_ip, 1) - 1)
            logger.debug("[teams_voice] connection closed %s", call_id)
        return ws

    async def _read_loop(self, session: CallSession, handler: CallSessionHandler) -> None:
        started = False
        while not session.closed:
            timeout = None if started else self.config.pre_start_timeout_s
            try:
                msg = await asyncio.wait_for(session._ws.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("[teams_voice] no session.start within %ss; closing %s",
                               timeout, session.call_id)
                await session._ws.close()
                return

            if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                return
            if msg.type is not WSMsgType.TEXT:
                continue

            try:
                parsed = protocol.decode(msg.data)
            except protocol.ProtocolError as exc:
                logger.warning("[teams_voice] bad frame on %s: %s", session.call_id, exc)
                continue

            started = await self._dispatch(session, handler, parsed) or started

    async def _dispatch(
        self,
        session: CallSession,
        handler: CallSessionHandler,
        parsed: protocol.InboundMessage,
    ) -> bool:
        """Route one parsed frame to the handler. Returns True once started."""
        try:
            if isinstance(parsed, protocol.Ping):
                await session._send(protocol.pong(parsed.ts))
                return False
            if isinstance(parsed, protocol.SessionStart):
                session.recording_active = parsed.recording_status == "active"
                await handler.on_session_start(session, parsed)
                return True
            if isinstance(parsed, protocol.AudioFrame):
                await handler.on_audio_frame(session, parsed)
            elif isinstance(parsed, protocol.VideoFrame):
                await handler.on_video_frame(session, parsed)
            elif isinstance(parsed, protocol.RecordingStatus):
                await handler.on_recording_status(session, parsed)
            elif isinstance(parsed, protocol.Participants):
                await handler.on_participants(session, parsed)
            elif isinstance(parsed, protocol.Dtmf):
                await handler.on_dtmf(session, parsed)
            elif isinstance(parsed, protocol.SessionEnd):
                await handler.on_session_end(session, parsed)
                await session._ws.close()
        except Exception:  # noqa: BLE001 — a handler fault must not kill the call
            logger.error(
                "[teams_voice] handler error on %s frame=%s",
                session.call_id, getattr(parsed, "type", "?"), exc_info=True,
            )
        return False


async def _amain() -> None:
    logging.basicConfig(level=logging.INFO)
    server = BridgeServer()
    await server.start()
    try:
        await asyncio.Future()  # run until cancelled
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_amain())
