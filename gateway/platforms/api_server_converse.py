"""OpenAI-style realtime voice over WebSocket: ``GET /v1/audio/converse``.

The gateway counterpart of the dashboard ``WS /api/audio/converse`` router
(:mod:`hermes_cli.web_routers.audio`), hosted on the aiohttp api_server. The
client streams mic PCM16 @16 kHz as binary WS frames; the server does VAD → STT
→ a real agent turn → streaming TTS and sends PCM16 back as binary frames. JSON
text frames carry control; barge-in is supported.

Authentication uses the profile's ``API_SERVER_KEY`` (``_expected_api_key``),
NOT the dashboard token, and never a ``?token=`` query param. Browser WebSocket
clients can't set request headers, so the ``Authorization: Bearer`` path is out;
instead the key is presented one of two ways (both validated constant-time), and
the single handler flow supports both:

(A) Sec-WebSocket-Protocol (preferred): the client offers ``hermes-voice-v1``
    plus ``hermes-key.<API_KEY>``. The key subprotocol is extracted from the
    request header exactly like ``_handle_browser_control_ws`` reads its ticket,
    compared constant-time, and — on success — the socket is accepted selecting
    ONLY the base ``hermes-voice-v1`` protocol (the key-bearing one is never
    echoed back). A mismatch rejects the upgrade with 401 before ``prepare``.
(B) First-message auth (fallback, only when no ``hermes-key.`` subprotocol is
    offered): the socket is accepted, then the client's FIRST frame must be
    ``{"type":"auth","key":"<API_KEY>"}`` within 5 s. Missing/invalid/timed-out
    → ``{"type":"error","error":"unauthorized"}`` and close 4401. No audio or
    control frame is processed and no session starts until auth succeeds.

The framework-agnostic VAD/STT/mic core lives in
:mod:`tools.voice_converse_loop`; this module owns only the aiohttp handler and
the per-turn incremental-TTS driver (mirroring the dashboard ``_drive_turns`` /
``_synthesize`` loop). It follows the same modular pattern as
:mod:`gateway.platforms.api_server_runs`: :func:`_http_routes` returns the route
table and the handler is bound onto the adapter.

Protocol:
  client → binary PCM16 @16 kHz mono frames (30 ms blocks preferred),
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
import queue
import threading
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
# First-message auth deadline (mechanism B): the client must send its auth frame
# within this window before any audio/control frame or session start.
_FIRST_MESSAGE_AUTH_TIMEOUT = 5.0


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


def _resolve_converse_session(self, profile: Optional[str]):
    """Resolve ``(synth, cap, session)`` under the profile scope.

    Blocking config/provider resolution — runs off the event loop. ``synth`` is a
    converse synthesizer (streaming when the provider has a chunked API, else the
    one-shot fallback — NEVER ``None``, so any provider incl. edge works), ``cap``
    its per-request max text length, ``session`` a fresh
    :class:`~tools.voice_converse_loop.ConverseSession`.
    """
    import numpy as np
    from tools.tts_tool import _get_provider, _load_tts_config, _resolve_max_text_length
    from tools.voice_converse_loop import ConverseSession, resolve_converse_synthesizer

    stt_model = _converse_stt_model(self, profile)
    with self._profile_scope(profile):
        cfg = _load_tts_config()
        synth = resolve_converse_synthesizer(cfg)
        cap = _resolve_max_text_length(_get_provider(cfg), cfg)
    return synth, cap, ConverseSession(np, stt_model=stt_model)


async def _await_first_message_auth(ws: "web.WebSocketResponse", expected_key: str) -> bool:
    """Mechanism B: require ``{"type":"auth","key":...}`` as the first frame within 5s.

    Returns True on success (the socket is authed and ready for audio/control),
    else sends ``{"type":"error","error":"unauthorized"}``, closes 4401 and returns
    False. No audio/control frame is processed here — the caller starts the session
    only after this returns True.
    """
    async def _reject() -> bool:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "unauthorized"})
            await ws.close(code=4401)
        return False

    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=_FIRST_MESSAGE_AUTH_TIMEOUT)
    except asyncio.TimeoutError:
        return await _reject()
    if msg.type != web.WSMsgType.TEXT:
        return await _reject()
    try:
        frame = json.loads(msg.data)
    except (ValueError, TypeError):
        return await _reject()
    if (not isinstance(frame, dict) or frame.get("type") != "auth"
            or not _key_ok(str(frame.get("key") or ""), expected_key)):
        return await _reject()
    return True


async def _handle_converse_ws(self, request: "web.Request") -> "web.WebSocketResponse":
    """GET /v1/audio/converse — off-device realtime voice over one WebSocket.

    Auth uses the profile's ``API_SERVER_KEY`` (``_expected_api_key``), NOT the
    dashboard token and never a ``?token=`` param. Two mechanisms, one flow:
    (A) a ``hermes-key.<KEY>`` subprotocol is validated BEFORE ``prepare`` (a
    mismatch rejects the upgrade with 401; success accepts selecting only the base
    ``hermes-voice-v1`` protocol); (B) if no key subprotocol is offered, the socket
    is accepted and the first frame must be ``{"type":"auth","key":...}`` within 5s.
    Only after auth do we resolve providers and run the client pump + turn driver.
    """
    from gateway.platforms.api_server import _api_request_profile

    profile = _api_request_profile.get()
    expected_key = self._expected_api_key()

    # (A) Subprotocol key: validated pre-prepare so a bad key rejects the upgrade.
    offered_key = _offered_key_protocol(request)
    if offered_key is not None:
        if not _key_ok(offered_key, expected_key):
            logger.warning("converse WS rejected invalid subprotocol API key")
            raise web.HTTPUnauthorized()
        # Accept selecting ONLY the base protocol — never echo the key-bearing one.
        ws = web.WebSocketResponse(heartbeat=30.0, protocols=(_VOICE_WS_PROTOCOL,))
        await ws.prepare(request)
    else:
        # (B) First-message auth: accept, then require an auth frame within 5s.
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        if not await _await_first_message_auth(ws, expected_key):
            return ws

    loop = asyncio.get_running_loop()
    try:
        synth, cap, session = await loop.run_in_executor(
            None, lambda: _resolve_converse_session(self, profile))
    except Exception:
        logger.exception("converse setup failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "error": "converse setup failed"})
            await ws.close()
        return ws

    await ws.send_json({
        "type": "ready",
        "input": {"sample_rate": 16000, "format": "pcm16", "block_ms": 30},
        "output": {"sample_rate": synth.sample_rate, "format": "pcm16"},
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
            pass
        session.stop()

    async def _drive_turns() -> None:
        # One turn per transcript: announce it, then run the real agent and speak its
        # reply INCREMENTALLY — each sentence is synthesized and streamed the moment it
        # is ready (mirrors the dashboard _drive_turns/_synthesize loop), so the user
        # hears sentence 1 while sentence 2 is still being generated.
        from tools.tts_streaming import SentenceChunker
        from tools.tts_text_normalize import _strip_markdown_for_tts
        from tools.voice_converse_loop import split_text_for_tts_stream

        while not session.stopped:
            transcript = await loop.run_in_executor(None, session.transcripts.get)
            if transcript is None:  # shutdown sentinel
                break
            if not transcript:
                continue
            await ws.send_json({"type": "transcript", "text": transcript})

            session.take_interrupted()  # clear any stale barge-in latch before this turn
            text_q: "queue.Queue[Optional[str]]" = queue.Queue()  # deltas; None = turn done
            pcm_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()  # PCM out; None = done
            tts_stop = threading.Event()
            reply_parts: List[str] = []
            turn_err: dict = {}

            def _on_delta(delta: str) -> None:
                # Called from _run_agent's executor thread; text_q is thread-safe.
                if delta:
                    reply_parts.append(delta)
                    text_q.put(delta)

            async def _run_turn(t=transcript, history=list(conversation_history)) -> None:
                # The real agent turn on the main loop (preserving the request's profile
                # scope); deltas stream out via _on_delta as they land, then the None
                # sentinel closes the synthesis pipeline when the turn ends. The session
                # row was pre-created source="voice"; the agent appends turns and its
                # create_session upsert preserves that source.
                try:
                    result, _usage = await self._run_agent(
                        user_message=t, conversation_history=history,
                        stream_delta_callback=_on_delta, session_id=session_id)
                    if isinstance(result, dict) and result.get("failed"):
                        turn_err["err"] = str(result.get("error") or "agent run failed")
                    elif isinstance(result, dict):
                        turn_err["final"] = str(result.get("final_response") or "")
                except Exception as exc:  # noqa: BLE001 - surface, don't wedge the loop
                    turn_err["err"] = f"voice turn failed: {exc}"
                finally:
                    text_q.put(None)

            def _produce() -> None:
                # Cut streaming deltas into sentences and synthesize each as it lands
                # (mirrors the dashboard _produce; text source is the live turn).
                chunker = SentenceChunker()
                idle_poll_seconds = 0.5
                idle_polls_before_force_flush = 4  # ~2s of silence -> speak the tail

                def _sentences():
                    idle_polls = 0
                    while not (tts_stop.is_set() or session.stopped):
                        try:
                            delta = text_q.get(timeout=idle_poll_seconds)
                        except queue.Empty:
                            idle_polls += 1
                            buffered = chunker.buf.strip()
                            if not buffered or (
                                    "<think" in chunker.buf and "</think>" not in chunker.buf):
                                continue
                            if buffered.endswith((".", "!", "?", "…", ":")) or (
                                    idle_polls >= idle_polls_before_force_flush):
                                yield from chunker.flush()
                            continue
                        idle_polls = 0
                        if delta is None:
                            yield from chunker.flush()
                            return
                        yield from chunker.feed(delta)

                try:
                    for sentence in _sentences():
                        cleaned = _strip_markdown_for_tts(sentence)
                        if not cleaned:
                            continue
                        for piece in split_text_for_tts_stream(cleaned, cap):
                            for chunk in synth.synth(piece):
                                if tts_stop.is_set() or session.stopped:
                                    return
                                loop.call_soon_threadsafe(pcm_q.put_nowait, chunk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("converse synthesis failed: %s", exc)
                finally:
                    loop.call_soon_threadsafe(pcm_q.put_nowait, None)

            turn_task = asyncio.ensure_future(_run_turn())
            threading.Thread(target=_produce, name="converse-tts", daemon=True).start()

            # Consumer: stream PCM out; flip `playing` on only when real audio starts
            # (kept off during generation so a mid-thought interjection stays VAD-sensitive).
            speaking = False
            while True:
                chunk = await pcm_q.get()
                if chunk is None:
                    break
                if not speaking:
                    session.set_playing(True, tts_stop=tts_stop)
                    await ws.send_json({"type": "speaking"})
                    speaking = True
                await ws.send_bytes(chunk)
            if speaking:
                session.set_playing(False)

            # The turn task set the None sentinel that ended synthesis, so it is
            # effectively done; await it to surface errors and settle turn_err.
            with contextlib.suppress(Exception):
                await turn_task

            # Persist the turn so history carries across the connection. Prefer the
            # agent's final response; fall back to the streamed deltas.
            reply = turn_err.get("final") or "".join(reply_parts)
            conversation_history.append({"role": "user", "content": transcript})
            if reply:
                conversation_history.append({"role": "assistant", "content": reply})

            if session.take_interrupted() or tts_stop.is_set():
                await ws.send_json({"type": "interrupted"})
            elif turn_err.get("err"):
                await ws.send_json({"type": "error", "error": turn_err["err"]})
            await ws.send_json({"type": "turn_done"})

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
