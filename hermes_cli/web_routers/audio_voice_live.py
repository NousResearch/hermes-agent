"""The authenticated server-side Azure realtime voice-live relay socket: ``/api/audio/voice-live-stream``.

Browsers and the desktop cannot hold the Azure realtime URL + resource key, so this WebSocket is
the client's whole vendor session: the client sends ``{"type": "session.start", "history": [...]}``
and the backend opens the Azure realtime WebSocket itself (preview URL, ``api-key`` header),
runs the ``session.created`` → ``session.update`` → ``session.updated`` handshake (the session
config carries the ``delegate_to_hermes`` function tool), then emits a synthetic
``session.started`` and pumps vendor events through ``tools/voice_live_relay``. Delegation is
function calling: ``response.function_call_arguments.done`` is forwarded (the desktop turns its
``call_id`` into the existing ``onDelegation``/``prompt.submit`` flow), browser
``session.commentary.append`` frames accumulate server-side per delegation and
``session.commentary.complete`` answers the call with ONE ``conversation.item.create``
function_call_output + ``response.create`` (``session.thinking.append`` is the fallback text).
``session.close`` closes everything. The Azure URL/key never appear in client events or logs.

Mounted via ``web_routers/audio.py`` (web_server mounts that router). Same trust boundary as the
other audio sockets: ``_ws_auth_ok`` with this route's own socket ticket, plus the host/origin
guard; voice config and the key resolve under the requesting profile's scope (query ``profile``
or the native grant's home), never the launch profile's.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from hermes_cli.dashboard_auth.native_http import grant_home_scope
from hermes_cli.web_deps import late
from hermes_cli.web_server_chat import _ws_auth_ok, _ws_close_reason, _ws_request_is_allowed
from hermes_cli.web_session_sockets import VOICE_LIVE_STREAM_SOCKET
from tools.voice_live_relay import (
    RELAY_CLOSE,
    RELAY_ERROR,
    RELAY_SESSION_CLOSED,
    RELAY_SESSION_STARTED,
    VoiceLiveRelay,
    redact_secrets,
)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")

#: How long the vendor open handshake (connect → session.created → session.update →
#: session.updated) may take.
OPEN_TIMEOUT_SECONDS = 30.0

#: How long the route waits for the relay pump thread to wind down during cleanup.
PUMP_JOIN_TIMEOUT_SECONDS = 5.0


@router.websocket("/api/audio/voice-live-stream")
async def voice_live_stream_ws(ws: WebSocket) -> None:
    """Relay one Azure realtime voice-live session for an authenticated client. See module docstring."""
    # A socket ticket opens this stream only when minted for it (``/api/webapp/socket-ticket``).
    if not _ws_auth_ok(ws, socket=VOICE_LIVE_STREAM_SOCKET):
        await ws.close(code=4401)
        return
    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return
    await ws.accept(subprotocol=getattr(ws, "_hermes_ws_subprotocol", None))

    profile = (ws.query_params.get("profile") or "").strip() or None
    grant = getattr(ws, "_hermes_native_grant", None)
    loop = asyncio.get_running_loop()

    async def _fail(message: str, code: int = 4400) -> None:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": RELAY_ERROR, "message": message})
        with contextlib.suppress(Exception):
            await ws.close(code=code, reason=_ws_close_reason(message))

    # Resolve the requesting profile's voice config + credentials off-loop, under that profile's
    # scope (query ?profile= or the native grant's home) — never the launch profile's.
    def _scoped_resolve():
        from tools import voice_live
        with grant_home_scope(grant), _config_profile_scope(profile):
            live = voice_live._live_section()
            api_key, base_url = voice_live._resolve_credentials(live)
            return live, api_key, base_url

    try:
        live, api_key, base_url = await asyncio.to_thread(_scoped_resolve)
    except HTTPException as exc:  # unknown ?profile= is the scope's 404, not a relay failure
        await _fail(str(exc.detail))
        return
    except Exception:  # health: allow BLE001 -- request boundary: answer the client, traceback logged below
        _log.exception("voice-live-stream credential resolution failed")
        await _fail("voice-live-stream is unavailable on this profile", 1011)
        return

    from tools import voice_live

    if voice_live.live_backend(base_url) != voice_live.BACKEND_AZURE:
        await _fail("voice-live-stream relays Azure realtime endpoints only; on this base_url use POST /api/audio/voice-live/session")
        return
    deployment = str(live.get("model") or voice_live.DEFAULT_LIVE_MODEL)
    ws_url = voice_live.azure_realtime_ws_url(base_url, deployment)
    secrets = (api_key, ws_url)  # redacted from anything that could reach the client or the logs
    if not api_key:
        await _fail("GPT-Live on an Azure endpoint needs an API key (voice.gpt_live.api_key, VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)")
        return

    try:
        first = json.loads(await ws.receive_text())
    except WebSocketDisconnect:
        return
    except (ValueError, RuntimeError):  # undecodable frame or dead socket: reject, don't crash the route
        await _fail("the first frame must be a JSON object", 1003)
        return
    if not isinstance(first, dict) or first.get("type") != "session.start":
        await _fail('the first frame must be {"type": "session.start", "history": [...]}')
        return
    history = first.get("history")
    if history is not None and not isinstance(history, list):
        await _fail("history must be a list of conversation items")
        return

    def _open():
        with grant_home_scope(grant), _config_profile_scope(profile):
            return voice_live.open_azure_realtime_session(
                history=history, timeout=OPEN_TIMEOUT_SECONDS,
                api_key=api_key, base_url=base_url)

    try:
        conn = await asyncio.to_thread(_open)
    except HTTPException as exc:
        await _fail(str(exc.detail))
        return
    except ValueError as exc:
        await _fail(redact_secrets(str(exc), secrets))
        return
    except RuntimeError as exc:
        _log.debug("voice-live-stream azure open rejected: %s", type(exc).__name__)
        await _fail(redact_secrets(str(exc), secrets), 1011)
        return
    except Exception:  # health: allow BLE001 -- request boundary: answer the client, traceback logged below
        _log.exception("voice-live-stream azure open failed")
        await _fail("the Azure realtime session could not be opened", 1011)
        return

    out_q: asyncio.Queue = asyncio.Queue()
    client_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def _to_browser(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(out_q.put_nowait, event)

    relay = VoiceLiveRelay(
        conn.ws, conn.session_id,
        send_to_browser=_to_browser,
        secrets=secrets,
    )
    from agent.memory_provider import spawn_context_thread  # keeps the request's profile scope on the thread
    pump = spawn_context_thread(
        relay.run, name=f"voice-live-relay-{conn.session_id}", args=(client_q, stop))
    pump.start()

    async def _pump_client() -> None:
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    frame = json.loads(raw)
                except ValueError:
                    out_q.put_nowait({"type": RELAY_ERROR, "message": "frames must be JSON"})
                    continue
                client_q.put(frame)
                if isinstance(frame, dict) and frame.get("type") == "session.close":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            client_q.put(RELAY_CLOSE)
            stop.set()

    client_task = asyncio.ensure_future(_pump_client())
    try:
        # Only after the vendor session is configured does the client get its session.started;
        # the pump is already running, but out_q is FIFO so started still lands first on the wire.
        await ws.send_json({"type": RELAY_SESSION_STARTED, "session": {"id": conn.session_id}})
        while True:
            event = await out_q.get()
            await ws.send_json(event)
            if event.get("type") == RELAY_SESSION_CLOSED:
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        client_q.put(RELAY_CLOSE)
        client_task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(pump.join, PUMP_JOIN_TIMEOUT_SECONDS)
        await asyncio.to_thread(relay.close)
        with contextlib.suppress(Exception):
            await ws.close()
