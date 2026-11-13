"""OpenAI-style realtime voice over WebSocket: ``GET /v1/audio/converse``.

Hosted on the aiohttp api_server, built on the framework-agnostic core in
:mod:`tools.voice_converse_loop`. The client streams mic PCM16 @16 kHz as
binary WS frames; the server does VAD → STT
→ a real agent turn → streaming TTS and sends PCM16 back as binary frames. JSON
text frames carry control; barge-in is supported.

Authentication uses the profile's ``API_SERVER_KEY`` (``_expected_api_key``),
NOT the dashboard token, and never a ``?token=`` query param. The client's FIRST
frame is a single ``{"type":"start", ...}`` frame carrying auth + all session
config (rates, quiet_interval, name, profile); the key is presented one of two
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
             "quiet_interval":?, "name":?, "profile":?}`` (FIRST frame; all but
             auth optional), then binary PCM16 mono frames at ``input_rate``
             (30 ms blocks preferred),
           ``{"stop": true}`` to end, ``{"commit": true}`` to force endpoint,
           ``{"type":"drained","turn_id":?}`` when the reply's last audio has left the speaker
             (ends the server's playback-tail: resumes quiet/VAD from real playback end, not from
             turn_done which fires mid-playback)
  server → ``{"type": "ready", "input": {...}, "output": {...}, "session_id": ...,
             "server_recv_unix_ms": ..., "server_send_unix_ms": ...}`` (session_id identifies this
             socket for client/server log correlation; the two timestamps + the client's
             start.t0_unix_ms close an NTP-style clock-skew exchange),
           ``{"type": "transcript", "text": ..., "turn_id": ..., "traceparent"?: ...}``,
           ``{"type": "thinking"}`` while the agent turn runs (show "thinking", not "listening"),
           ``{"type": "speaking", "turn_id": ..., "traceparent"?: ...}`` then binary PCM frames,
           ``{"type": "interrupted", "turn_id": ...}`` on barge-in,
           ``{"type": "turn_done", "turn_id": ..., "expects_more": ?}`` after each reply (session
             mode sets expects_more=false when the agent signed off → sleep, true on a trailing
             question → keep listening; absent otherwise),
           ``{"type": "error", "error": ..., "turn_id": ...}`` on failure.
           turn_id ties a turn's frames together so a client (or log) can correlate them.
           traceparent (W3C) is present only when a tracing backend is plugged into
           :mod:`tools.voice_tracing`; a client can nest its own spans under the server's turn.
"""

import asyncio
import contextlib
import functools
import hmac
import json
import logging
import time
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


def is_voice_key_ws_upgrade(request) -> bool:
    """True for a ``/v1/audio/converse`` WebSocket upgrade carrying a
    ``hermes-key.<KEY>`` subprotocol.

    Such handshakes authenticate with an explicit bearer key in
    ``Sec-WebSocket-Protocol`` (validated constant-time by the handler), not an
    ambient cookie. The Origin/CSRF guard exists to protect credentials the
    browser attaches automatically — a hostile page can set any subprotocol but
    cannot produce a key it lacks — so it adds nothing on top of the key here,
    while it *does* block legitimate browser-context clients (Electron renderers
    send an unsuppressible ``Origin: null``). The converse route accepts no
    cookie auth, so relaxing Origin for key-bearing upgrades is safe.

    PRESENCE, NOT VALIDITY (on purpose): the exemption keys off the mere presence
    of a ``hermes-key.`` subprotocol, not whether the key is correct. Validating
    the key here would duplicate auth into the wrong layer — two constant-time
    paths to keep in lockstep, and a wrong key would 403-as-origin instead of
    401-as-auth. The only cost of presence-based matching is that a hostile page
    can reach the handler's constant-time key check and burn a 401 — but any
    non-browser client can already do that by omitting Origin entirely, so it
    concedes nothing.

    INVARIANT: this holds ONLY while ``/v1/audio/converse`` authenticates with
    the explicit key alone (no cookie/session fallback). If ambient auth is ever
    added to that handler, this exemption silently becomes a Cross-Site
    WebSocket Hijacking hole — ``test_cookie_is_not_a_credential`` guards it."""
    if request.headers.get("Upgrade", "").lower() != "websocket":
        return False
    if "upgrade" not in request.headers.get("Connection", "").lower():
        return False
    protocols = request.headers.get("Sec-WebSocket-Protocol", "")
    return any(p.strip().startswith("hermes-key.") for p in protocols.split(","))


def _http_routes(self) -> list:
    """(method, path, handler) rows for the converse endpoint.

    Same shape as ``api_server_runs._http_routes`` / ``room_grants._http_routes``
    so ``api_server._http_route_table`` can ``routes.extend(...)`` it.
    """
    return [("GET", "/v1/audio/converse", functools.partial(_handle_converse_ws, self))]


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


def _resolve_converse_session(self, profile: Optional[str], input_rate: int, output_rate: int,
                              quiet_interval: float = 0.0):
    """Resolve ``(synth, cap, session, route)`` under the profile scope for the given rates.
    ``route`` is the optional voice-only ``(model, provider)`` override (``voice.model`` /
    ``voice.provider``; ``None`` each = normal resolution), read once per connection.

    Blocking config/provider resolution — runs off the event loop. ``synth`` is a
    converse synthesizer (streaming when the provider has a chunked API, else the
    one-shot fallback — NEVER ``None``, so any provider incl. edge works) wrapped to
    emit int16 PCM at *output_rate*, ``cap`` its per-request max text length,
    ``session`` a fresh :class:`~tools.voice_converse_loop.ConverseSession` that
    captures at *input_rate*.
    """
    import numpy as np
    from hermes_cli.config import load_config
    from tools.tts_tool import _get_provider, _load_tts_config, _resolve_max_text_length
    from tools.voice_converse_loop import (
        ConverseSession, parse_smart_turn_config, resample_synth, resolve_converse_synthesizer)
    from tools.voice_converse_vad import parse_vad_config

    stt_model = _converse_stt_model(self, profile)
    with self._profile_scope(profile):
        cfg = _load_tts_config()
        synth = resample_synth(resolve_converse_synthesizer(cfg), output_rate)
        cap = _resolve_max_text_length(_get_provider(cfg), cfg)
        full_cfg = load_config()
        endpoint_model, endpoint_threshold = parse_smart_turn_config(full_cfg)
        vad_model = parse_vad_config(full_cfg)
        voice_cfg = full_cfg.get("voice") or {}
        route = (voice_cfg.get("model") or None, voice_cfg.get("provider") or None)
        # Inside the scope: the Silero weights cache under the profile's home on first use.
        session = ConverseSession(
            np, stt_model=stt_model, input_rate=input_rate, quiet_interval=quiet_interval,
            endpoint_model=endpoint_model, endpoint_threshold=endpoint_threshold,
            vad_model=vad_model)
    return synth, cap, session, route


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

    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=_FIRST_FRAME_TIMEOUT)
    except TimeoutError:
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
    (input_rate/output_rate/quiet_interval/name/profile). Only after a valid start do
    we resolve providers and run the client pump + turn driver.

    BARGE-IN: a VAD trip over the reply stops TTS playback (tts_stop +
    mark_speech_interrupted) AND cancels the in-flight agent turn — the driver calls the
    ``interrupt_turn`` hook, which cooperatively interrupts the live ``_run_agent`` via
    ``agent_ref[0].interrupt(hard_cancel=True)`` (thread-safe; the turn loop settles cleanly,
    preserving prompt-cache + role-alternation invariants). So the agent stops generating /
    firing tools and the barge-in utterance is handled at once instead of queuing behind it.
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
    server_recv_unix_ms = int(time.time() * 1000)  # when the start frame arrived (clock-skew ref)
    from tools.voice_converse_loop import parse_start_config
    input_rate, output_rate, quiet_interval, name, start_profile = parse_start_config(frame)
    # Profile: the start frame wins; else fall back to the request's own scope.
    profile = start_profile if start_profile is not None else _api_request_profile.get()

    loop = asyncio.get_running_loop()
    try:
        synth, cap, session, (voice_model, voice_provider) = await loop.run_in_executor(
            None, lambda: _resolve_converse_session(
                self, profile, input_rate, output_rate, quiet_interval))
    except Exception:
        logger.exception("converse setup failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "converse setup failed"})
            await ws.close()
        return ws

    # Correlation fields: session_id joins this socket across client/server traces; the two
    # timestamps close an NTP-style skew exchange with the client's start.t0_unix_ms (recv = when
    # the start frame arrived, send = now).
    await ws.send_json({
        "type": "ready",
        "input": {"sample_rate": input_rate, "format": "pcm16", "block_ms": 30},
        "output": {"sample_rate": output_rate, "format": "pcm16"},
        "session_id": session.session_id,
        "server_recv_unix_ms": server_recv_unix_ms,
        "server_send_unix_ms": int(time.time() * 1000),
    })

    session.start()
    # Stable per-connection identity + spoken-conversation history persisted for the
    # life of the socket (each turn appends the user + assistant message so context
    # carries across turns, exactly like the dashboard's ephemeral session).
    session_id = f"voice_{uuid.uuid4().hex}"
    conversation_history: list[dict[str, str]] = []
    # The session row is created by the API turn door on the first turn (source api_server); a row
    # pre-created under another source would be refused there as a foreign surface.

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
                    if isinstance(frame, dict) and frame.get("type") == "drained":
                        # Client finished PLAYING the reply → end the playback tail (resume
                        # quiet/VAD from now, not from turn_done which was mid-playback).
                        session.notify_drained(frame.get("turn_id"))
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        except Exception:
            logger.debug("converse client pump ended", exc_info=True)
        session.stop()

    # Holds the live AIAgent for the CURRENT turn so a barge-in can interrupt it. _run_agent
    # writes agent_box[0] = agent, so the slot MUST pre-exist — a bare [] makes that assignment
    # raise "list assignment index out of range" and fails every turn. _run_turn nulls the slot
    # around each turn so no stale agent is interrupted between turns.
    agent_box: list = [None]

    def _interrupt_turn() -> None:
        # Barge-in: cooperatively stop the in-flight agent turn (thread-safe; the turn loop honors
        # the flag and settles cleanly, preserving prompt-cache + role-alternation invariants).
        ag = agent_box[0] if agent_box else None
        if ag is not None:
            with contextlib.suppress(Exception):
                ag.interrupt(hard_cancel=True)

    async def _run_turn(transcript, on_delta, *, interrupted: bool):
        # Gateway turn adapter for drive_converse_turns: run the real agent turn on
        # the main loop (preserving the request's profile scope); deltas stream out via
        # on_delta as they land. Returns (reply_text, err): the reply is the agent's
        # final response (fallback: joined deltas, computed by the driver), and err is
        # set from result["failed"] or an exception.
        #
        # NOTE: `interrupted` is accepted for the shared driver's signature but the
        # gateway does not plumb a barge-in note into _run_agent (dashboard parity is
        # dashboard-only), so it is intentionally unused here.
        from tools.voice_converse_loop import voice_system_prompt
        agent_box[0] = None
        try:
            result, _usage = await self._run_agent(
                user_message=transcript, conversation_history=list(conversation_history),
                ephemeral_system_prompt=voice_system_prompt(name, allow_signoff=quiet_interval > 0),
                stream_delta_callback=on_delta, session_id=session_id,
                requested_model=voice_model, requested_provider=voice_provider,
                agent_ref=agent_box)
        finally:
            agent_box[0] = None  # no stale agent to interrupt between turns
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
            quiet_interval=quiet_interval, interrupt_turn=_interrupt_turn)

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
