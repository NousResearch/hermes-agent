"""OpenAI-style realtime voice over WebSocket: ``GET /v1/audio/converse``.

The gateway counterpart of the dashboard ``WS /api/audio/converse`` router
(:mod:`hermes_cli.web_routers.audio`), hosted on the aiohttp api_server. The
client streams mic PCM16 @16 kHz as binary WS frames; the server does VAD → STT
→ a real agent turn → streaming TTS and sends PCM16 back as binary frames. JSON
text frames carry control; barge-in is supported.

Authentication uses the profile's ``API_SERVER_KEY`` (``_expected_api_key``),
NOT the dashboard token, and never a ``?token=`` query param. The client's FIRST
frame is a single ``{"type":"start", ...}`` frame carrying auth + all session
config (rates, idle_interval, name, profile); the key is presented one of two
ways (both validated constant-time), so the single handler flow supports both:

(A) Sec-WebSocket-Protocol (browser clients, e.g. Caduceus): the client offers
    ``hermes-voice-v1`` plus ``hermes-key.<API_KEY>``. The key subprotocol is
    validated constant-time BEFORE ``prepare`` and — on success — the socket is
    accepted selecting ONLY the base ``hermes-voice-v1`` protocol (the key-bearing
    one is never echoed back). A mismatch rejects the upgrade with 401. With the
    subprotocol key present, ``start.key`` is optional/ignored.
(B) start.key (non-browser devices, e.g. ESP32): no subprotocol is offered, the
    socket is accepted, and the ``start`` frame's ``key`` authenticates
    constant-time. A non-``start`` first frame → ``{"error":"bad_start"}`` close
    4400; a ``start`` with a bad/missing key → ``{"error":"unauthorized"}`` close
    4401. No audio/control frame is processed until a valid ``start`` succeeds.

Per-connection sample rates: ``start.input_rate`` (default 16000) sets the capture
rate and ``start.output_rate`` (default 24000) the reply-PCM rate, so a single-clock
device can make input == output. Rates are clamped to [8000, 48000].

The framework-agnostic VAD/STT/mic core AND the per-turn incremental-TTS driver
(:func:`tools.voice_converse_loop.drive_converse_turns`, shared with the dashboard)
live in :mod:`tools.voice_converse_loop`; this module owns only the aiohttp handler
plus a thin ``_run_turn`` adapter that wraps :meth:`_run_agent` onto the driver. It
follows the same modular pattern as :mod:`gateway.platforms.api_server_runs`:
:func:`_http_routes` returns the route table and the handler is bound onto the
adapter.

Protocol:
  client → ``{"type":"start", "key":?, "input_rate":?, "output_rate":?,
             "idle_interval":?, "name":?, "profile":?}`` (FIRST frame; all but
             auth optional), then binary PCM16 mono frames at ``input_rate``
             (30 ms blocks preferred),
           ``{"stop": true}`` to end, ``{"commit": true}`` to force endpoint
  server → ``{"type": "ready", "input": {...}, "output": {...}}``,
           ``{"type": "transcript", "text": ...}``,
           ``{"type": "speaking"}`` then binary PCM frames,
           ``{"type": "interrupted"}`` on barge-in,
           ``{"type": "turn_done"}`` after each reply,
           ``{"type": "error", "error": ...}`` on failure.
"""

import asyncio
import contextlib
import hmac
import json
import logging
import uuid
from typing import Dict, List, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


logger = logging.getLogger("gateway.platforms.api_server")

# Subprotocols: clients offer the base + ``hermes-key.<API_KEY>``; only the base
# is ever selected on the accepted socket (the key is never echoed back).
_VOICE_WS_PROTOCOL = "hermes-voice-v1"
_VOICE_KEY_PROTOCOL_PREFIX = "hermes-key."
# Start-frame deadline: the client must send its ``{"type":"start", ...}`` frame within this
# window before any audio/control frame or session start.
_FIRST_FRAME_TIMEOUT = 5.0


def _key_ok(candidate: str, expected: str) -> bool:
    """Constant-time compare of a presented key to the profile's expected key.

    False when either side is empty (an unconfigured key must never admit a
    request) — compare as bytes so a non-ASCII candidate 401s rather than 500s.
    """
    if not candidate or not expected:
        return False
    return hmac.compare_digest(candidate.encode(), expected.encode())


def _offered_key_protocol(request: "web.Request") -> Optional[str]:
    """Return the single ``hermes-key.<KEY>`` value the client offered, or None.

    ``None`` means no key subprotocol was offered at all (-> mechanism B). An
    empty string means one was offered but malformed/empty (-> reject in A).
    """
    offered = [
        value.strip()
        for value in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        if value.strip()]
    key_protocols = [v for v in offered if v.startswith(_VOICE_KEY_PROTOCOL_PREFIX)]
    if not key_protocols:
        return None
    if len(key_protocols) != 1:
        return ""  # ambiguous — treat as malformed
    return key_protocols[0][len(_VOICE_KEY_PROTOCOL_PREFIX):]


def _http_routes(self) -> list:
    """(method, path, handler) rows for the converse endpoint.

    Same shape as ``api_server_runs._http_routes`` / ``room_grants._http_routes``
    so ``api_server._http_route_table`` can ``routes.extend(...)`` it.
    """
    return [("GET", "/v1/audio/converse", self._handle_converse_ws)]


def _converse_stt_model(self, profile: Optional[str]) -> Optional[str]:
    """STT model override for the converse loop (mirrors the dashboard resolver).

    Local provider prefers ``stt.local.model`` (default ``base``); every other
    provider uses ``stt.model`` (or the provider default when unset). Resolved
    under the request's profile scope.
    """
    with self._profile_scope(profile):
        from hermes_cli.config import load_config

        stt = (load_config().get("stt") or {})
        if str(stt.get("provider") or "").strip().lower() == "local":
            local = stt.get("local") if isinstance(stt.get("local"), dict) else {}
            return (local or {}).get("model") or "base"
        return stt.get("model")


def _resolve_converse_session(self, profile: Optional[str], input_rate: int, output_rate: int):
    """Resolve ``(synth, cap, session)`` under the profile scope for the given rates.

    Blocking config/provider resolution — runs off the event loop. ``synth`` is a
    converse synthesizer (streaming when the provider has a chunked API, else the
    one-shot fallback — NEVER ``None``, so any provider incl. edge works) wrapped to
    emit int16 PCM at *output_rate*, ``cap`` its per-request max text length,
    ``session`` a fresh :class:`~tools.voice_converse_loop.ConverseSession` that
    captures at *input_rate*.
    """
    import numpy as np
    from tools.tts_tool import _get_provider, _load_tts_config, _resolve_max_text_length
    from tools.voice_converse_loop import (
        ConverseSession, resample_synth, resolve_converse_synthesizer)

    stt_model = _converse_stt_model(self, profile)
    with self._profile_scope(profile):
        cfg = _load_tts_config()
        synth = resample_synth(resolve_converse_synthesizer(cfg), output_rate)
        cap = _resolve_max_text_length(_get_provider(cfg), cfg)
    return synth, cap, ConverseSession(np, stt_model=stt_model, input_rate=input_rate)


async def _await_start_frame(
    ws: "web.WebSocketResponse", expected_key: str, *, subprotocol_authed: bool,
) -> Optional[dict]:
    """Require ``{"type":"start", ...}`` as the client's FIRST frame within 5s.

    Returns the parsed frame dict on success. On failure sends an error frame, closes and
    returns ``None``: a non-JSON / non-``start`` first frame → ``bad_start`` + close 4400;
    a valid ``start`` that fails auth (no subprotocol key AND bad/missing ``start.key``) →
    ``unauthorized`` + close 4401. No audio/control frame is processed until this succeeds.
    """
    async def _fail(error: str, code: int) -> None:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": error})
            await ws.close(code=code)
        return None

    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=_FIRST_FRAME_TIMEOUT)
    except asyncio.TimeoutError:
        return await _fail("bad_start", 4400)
    if msg.type != web.WSMsgType.TEXT:
        return await _fail("bad_start", 4400)
    try:
        frame = json.loads(msg.data)
    except (ValueError, TypeError):
        return await _fail("bad_start", 4400)
    if not isinstance(frame, dict) or frame.get("type") != "start":
        return await _fail("bad_start", 4400)
    if not subprotocol_authed and not _key_ok(str(frame.get("key") or ""), expected_key):
        return await _fail("unauthorized", 4401)
    return frame


async def _handle_converse_ws(self, request: "web.Request") -> "web.WebSocketResponse":
    """GET /v1/audio/converse — off-device realtime voice over one WebSocket.

    Auth uses the profile's ``API_SERVER_KEY`` (``_expected_api_key``), NOT the
    dashboard token and never a ``?token=`` param. Two mechanisms, one flow:
    (A) a ``hermes-key.<KEY>`` subprotocol is validated BEFORE ``prepare`` (a
    mismatch rejects the upgrade with 401; success accepts selecting only the base
    ``hermes-voice-v1`` protocol); (B) if no key subprotocol is offered, the socket
    is accepted and the ``start`` frame's ``key`` authenticates. The client's FIRST
    frame is a single ``{"type":"start", ...}`` carrying auth + all session config
    (input_rate/output_rate/idle_interval/name/profile). Only after a valid start do
    we resolve providers and run the client pump + turn driver.

    BARGE-IN (v1 limitation): the shared turn driver
    (:func:`tools.voice_converse_loop.drive_converse_turns`) stops TTS PLAYBACK on a
    VAD trip (tts_stop + mark_speech_interrupted) but the in-flight agent turn
    (``_run_agent``) is NOT cancelled in v1 — it runs to completion, so a barged turn
    may still finish and fire tools, and the next utterance queues behind it.
    Cancelling the in-flight turn is a deliberate follow-up.
    """
    from gateway.platforms.api_server import _api_request_profile

    expected_key = self._expected_api_key()

    # (A) Subprotocol key: validated pre-prepare so a bad key rejects the upgrade. When
    # present, the socket is subprotocol-authed and start.key is optional/ignored.
    offered_key = _offered_key_protocol(request)
    if offered_key is not None:
        if not _key_ok(offered_key, expected_key):
            logger.warning("converse WS rejected invalid subprotocol API key")
            raise web.HTTPUnauthorized()
        # Accept selecting ONLY the base protocol — never echo the key-bearing one.
        ws = web.WebSocketResponse(heartbeat=30.0, protocols=(_VOICE_WS_PROTOCOL,))
        await ws.prepare(request)
        subprotocol_authed = True
    else:
        # (B) No subprotocol: accept, then require a start frame whose start.key authenticates.
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        subprotocol_authed = False

    # The client's FIRST frame is the single start frame carrying auth + all session config.
    frame = await _await_start_frame(ws, expected_key, subprotocol_authed=subprotocol_authed)
    if frame is None:
        return ws
    from tools.voice_converse_loop import parse_start_config
    input_rate, output_rate, idle_interval, name, start_profile = parse_start_config(frame)
    # Profile: the start frame wins; else fall back to the request's own scope.
    profile = start_profile if start_profile is not None else _api_request_profile.get()

    loop = asyncio.get_running_loop()
    try:
        synth, cap, session = await loop.run_in_executor(
            None, lambda: _resolve_converse_session(self, profile, input_rate, output_rate))
    except Exception:
        logger.exception("converse setup failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "converse setup failed"})
            await ws.close()
        return ws

    await ws.send_json({
        "type": "ready",
        "input": {"sample_rate": input_rate, "format": "pcm16", "block_ms": 30},
        "output": {"sample_rate": output_rate, "format": "pcm16"},
    })

    session.start()
    # Stable per-connection identity + spoken-conversation history persisted for the
    # life of the socket (each turn appends the user + assistant message so context
    # carries across turns, exactly like the dashboard's ephemeral session).
    session_id = f"voice_{uuid.uuid4().hex}"
    conversation_history: List[Dict[str, str]] = []
    # Pre-create the session row as source="voice" (a chat sub-kind the dashboard files
    # under Chats/"Voice", not "Automations"). The agent's later create_session upsert
    # PRESERVES an existing row's source (it is absent from ON CONFLICT DO UPDATE), so
    # this sticks — while the agent PLATFORM stays "api_server", leaving HA toolset
    # resolution untouched. Off-loop: create_session is a blocking SQLite write.
    def _precreate_voice_session() -> None:
        db = self._ensure_session_db()
        if db is not None:
            db.create_session(session_id=session_id, source="voice")

    with contextlib.suppress(Exception):
        await loop.run_in_executor(None, _precreate_voice_session)

    async def _pump_client() -> None:
        # Binary frames feed the mic shim; {"stop"}/disconnect ends; {"commit"}
        # forces the current utterance to endpoint.
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    session.stream.feed(msg.data)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        frame = json.loads(msg.data)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(frame, dict) and frame.get("stop"):
                        break
                    if isinstance(frame, dict) and frame.get("commit"):
                        session.commit()
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except Exception:
            logger.debug("converse client pump ended", exc_info=True)
        session.stop()

    async def _run_turn(transcript, on_delta, *, interrupted: bool):
        # Gateway turn adapter for drive_converse_turns: run the real agent turn on
        # the main loop (preserving the request's profile scope); deltas stream out via
        # on_delta as they land. Returns (reply_text, err): the reply is the agent's
        # final response (fallback: joined deltas, computed by the driver), and err is
        # set from result["failed"] or an exception. The session row was pre-created
        # source="voice"; the agent's create_session upsert preserves that source.
        #
        # NOTE: `interrupted` is accepted for the shared driver's signature but the
        # gateway does not plumb a barge-in note into _run_agent (dashboard parity is
        # dashboard-only), so it is intentionally unused here.
        from tools.voice_converse_loop import voice_system_prompt
        result, _usage = await self._run_agent(
            user_message=transcript, conversation_history=list(conversation_history),
            ephemeral_system_prompt=voice_system_prompt(name),
            stream_delta_callback=on_delta, session_id=session_id)
        if isinstance(result, dict) and result.get("failed"):
            return "", str(result.get("error") or "agent run failed")
        if isinstance(result, dict):
            return str(result.get("final_response") or ""), None
        return "", None

    async def _drive_turns() -> None:
        from tools.voice_converse_loop import drive_converse_turns

        await drive_converse_turns(
            session=session, synth=synth, cap=cap, loop=loop,
            send_json=ws.send_json, send_bytes=ws.send_bytes,
            run_turn=_run_turn, history=conversation_history,
            idle_interval=idle_interval)

    pump = asyncio.ensure_future(_pump_client())
    driver = asyncio.ensure_future(_drive_turns())
    try:
        await driver
    except (asyncio.CancelledError, RuntimeError):
        pass
    except Exception:
        logger.exception("converse loop crashed")
    finally:
        session.stop()
        pump.cancel()
        driver.cancel()
        with contextlib.suppress(Exception):
            await ws.close()
    return ws
