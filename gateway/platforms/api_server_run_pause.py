"""Pause / resume a live run over the Runs API.

- ``POST /v1/runs/{run_id}/pause``   ask the run to pause at its next safe boundary
- ``POST /v1/runs/{run_id}/resume``  let a pausing or paused run continue

Status ``running`` -> ``pausing`` (asked) -> ``paused`` (blocked at the boundary) -> ``running``.
409 when the run is not in a state that allows the call, 404 when the run is unknown. Works for
``POST /v1/runs`` runs and for ``POST /api/sessions/{id}/chat/stream`` turns: the agent is reached
exactly as ``/steer`` reaches it (the session authority's adopted agent, else
``_active_run_agents``). ``run.paused`` / ``run.resumed`` go to the run's ``/v1/runs/{id}/events``
stream and to the session chat stream; both SSE loops keep sending keepalives while paused.

A pause is live-process state on the agent (``RunPauseMixin``), never persisted: GET
``/v1/runs/{id}`` overlays it on the stored ``running`` status. The max-pause
guard (``gateway.api_server.max_pause_seconds`` / ``API_SERVER_MAX_PAUSE_SECONDS``, default
1800, 0 = off) stops a run left paused, so a forgotten pause does not hold a run slot.

Kept in its own module so ``api_server.py`` only wires it in.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import weakref
from contextlib import suppress
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

logger = logging.getLogger("gateway.platforms.api_server")

DEFAULT_MAX_PAUSE_SECONDS = 1800.0
_PAUSABLE_STATUSES = ("running",)

# Spliced into ``GET /v1/capabilities`` by ``APIServerAdapter._handle_capabilities``.
FEATURE_FLAGS = {"run_pause": True}
CAPABILITY_ENDPOINTS = (
    ("run_pause", ("POST", "/v1/runs/{run_id}/pause")),
    ("run_resume", ("POST", "/v1/runs/{run_id}/resume")),
)


def max_pause_seconds() -> float:
    """``API_SERVER_MAX_PAUSE_SECONDS``, else ``gateway.api_server.max_pause_seconds``, else 1800;
    0 turns the auto-stop off, negatives clamp to 0, an unreadable value falls back to the default."""
    raw = os.getenv("API_SERVER_MAX_PAUSE_SECONDS")
    try:
        if raw is None or raw == "":
            from hermes_cli.config import cfg_get, load_config
            raw = cfg_get(load_config(), "gateway", "api_server", "max_pause_seconds",
                          default=DEFAULT_MAX_PAUSE_SECONDS)
        return max(0.0, float(raw))
    except Exception:  # health: allow BLE001 -- a bad config value must not break /pause; logged with traceback
        logger.warning(
            "max_pause_seconds %r (API_SERVER_MAX_PAUSE_SECONDS or gateway.api_server.max_pause_seconds) is not a"
            " number of seconds; set one (0 turns the auto-stop off). Using %ss.", raw, DEFAULT_MAX_PAUSE_SECONDS,
            exc_info=True)
        return DEFAULT_MAX_PAUSE_SECONDS


def _paused_agents(adapter) -> dict[str, tuple[Any, Any]]:
    """run_id -> (weakref to the agent a pause was accepted on, that pause's turn id). Weak, so a
    finished agent is never kept alive; turn-scoped, so a cached agent paused later for another run
    is never reached through this run's id."""
    return vars(adapter).setdefault("_run_pause_agents", {})


def _paused_agent(adapter, run_id: str) -> Optional[Any]:
    ref, turn = _paused_agents(adapter).get(run_id, (None, None))
    agent = ref() if ref is not None else None
    if agent is None or agent.pause_state() is None or vars(agent).get("_run_pause_turn") != turn:
        _paused_agents(adapter).pop(run_id, None)
        return None
    return agent


def _prune(adapter) -> None:
    for run_id in list(_paused_agents(adapter)):
        _paused_agent(adapter, run_id)


def _stream_sinks(adapter) -> weakref.WeakValueDictionary[str, Any]:
    """run_id -> the session chat stream's event queue; weak, so it goes with the request."""
    return vars(adapter).setdefault("_run_pause_stream_sinks", weakref.WeakValueDictionary())


def watch_session_stream(adapter, events: Any) -> Any:
    """Register a session chat stream's event queue (keyed by its ``run_id``) so ``run.paused`` /
    ``run.resumed`` reach it; returns *events* so the hook wraps the constructor call in place."""
    _stream_sinks(adapter)[events.run_id] = events
    return events


def _api_server():
    return sys.modules["gateway.platforms.api_server"]


def _api_runs():
    from gateway.platforms import api_server_runs
    return api_server_runs


def _publish(adapter, run_id: str, name: str, fields: dict) -> None:
    """On the loop: one control event to the run's SSE stream and the session chat stream."""
    runs = _api_runs()
    current = adapter._run_statuses.get(run_id)
    if current is not None:
        adapter._set_run_status(run_id, str(current.get("status") or "running"), last_event=name)
    stream = adapter._run_streams.get(run_id)
    if stream is not None:
        with suppress(Exception):
            stream.put_nowait(runs._run_event(run_id, name, **fields))
    sink = _stream_sinks(adapter).get(run_id)
    if sink is not None:
        with suppress(Exception):
            sink.enqueue(name, dict(fields))


def _listener(adapter, run_id: str, loop: asyncio.AbstractEventLoop):
    def _on_change(name: str, fields: dict) -> None:
        # Called from the turn's thread (run.paused) or the loop (run.resumed): always hop.
        with suppress(RuntimeError):  # loop closed: nobody left to tell
            loop.call_soon_threadsafe(_publish, adapter, run_id, name, fields)
    return _on_change


def _live_agent(adapter, run_id: str) -> Optional[Any]:
    """The agent a pause for *run_id* reaches: the same lookup ``/steer`` uses."""
    if run_id in adapter._stopping_run_ids:
        return None
    if getattr(adapter.gateway_runner, "session_authority", None) is None:
        return adapter._active_run_agents.get(run_id)
    from gateway.platforms.api_server_authority_runs import run_admission
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError
    owned = run_admission(adapter, run_id)
    if owned is None:
        return None
    authority, row = owned
    if row["status"] != "started":
        return None
    sid = row["target_session_id"]
    try:
        with authority.sessions[sid].event_stream.lock:
            authority.check_approval_generation(sid, row["generation"])
            agent, _ = authority.steer_agent(SessionRef(authority.profile_id, sid), row["generation"], None)
    except (KeyError, RuntimeStoreError):
        return None
    return agent


def overlay_status(adapter, run_id: str, status: dict) -> dict:
    """*status* with ``pausing`` / ``paused`` in place of ``running`` while the run's agent is paused."""
    agent = _paused_agent(adapter, run_id)
    if agent is None or not isinstance(status, dict) or status.get("status") not in _PAUSABLE_STATUSES:
        return status
    state = agent.pause_state()
    return {**status, "status": state} if state else status


def _refused(run_id: str, message: str, code: str) -> web.Response:
    from gateway.platforms.api_server_room_grants import _json_error
    return _json_error(_api_server()._openai_error, f"{message}: {run_id}", code=code, status=409)


def _load(adapter, request):
    return _api_runs()._load_owned_run(
        adapter, request, _api_server=_api_server(), permission=None, active_fallback=False)


async def handle_pause(adapter, request: web.Request) -> web.Response:
    """POST /v1/runs/{run_id}/pause — pause the run at its next safe boundary."""
    run_id, status, _, _, err = _load(adapter, request)
    if err is not None:
        return err
    agent = _live_agent(adapter, run_id)
    if status.get("status") not in _PAUSABLE_STATUSES or agent is None or not callable(
            getattr(type(agent), "pause", None)):
        return _refused(run_id, "Run is not currently pausable", "run_not_pausable")
    limit = max_pause_seconds()
    if not agent.pause(max_seconds=limit, listener=_listener(adapter, run_id, asyncio.get_running_loop())):
        return _refused(run_id, "Run is already pausing or paused, or is stopping", "run_not_pausable")
    _prune(adapter)
    _paused_agents(adapter)[run_id] = (weakref.ref(agent), vars(agent).get("_run_pause_turn"))
    return web.json_response({
        "object": "hermes.run.pause", "run_id": run_id, "status": "pausing",
        "max_pause_seconds": limit or None})


async def handle_resume(adapter, request: web.Request) -> web.Response:
    """POST /v1/runs/{run_id}/resume — let a pausing or paused run continue."""
    run_id, status, _, _, err = _load(adapter, request)
    if err is not None:
        return err
    if status.get("status") not in _PAUSABLE_STATUSES:
        return _refused(run_id, "Run is not pausing or paused", "run_not_paused")
    agent = _paused_agent(adapter, run_id) or _live_agent(adapter, run_id)
    if agent is None or not callable(getattr(type(agent), "resume", None)) or not agent.resume():
        return _refused(run_id, "Run is not pausing or paused", "run_not_paused")
    _paused_agents(adapter).pop(run_id, None)
    return web.json_response({"object": "hermes.run.resume", "run_id": run_id, "status": "running"})


_HANDLERS = (
    ("/v1/runs/{run_id}/pause", "handle_pause"),
    ("/v1/runs/{run_id}/resume", "handle_resume"),
)


def _bind(adapter, name: str):
    # Looked up at call time so test patches on this module take effect.
    async def _handler(request: web.Request) -> web.Response:
        return await globals()[name](adapter, request)

    _handler.__name__ = name
    return _handler


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    """(method, path, handler) rows for ``APIServerAdapter._http_route_table``."""
    return [("POST", path, _bind(adapter, name)) for path, name in _HANDLERS]
