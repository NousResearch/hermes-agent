"""The desktop window bridges for a local viewer's turn on the session authority.

Main's TUI sidecar wired two things per session (``tui_gateway/session_notifications.py::
_wire_desktop_sinks`` and ``tui_gateway/agent_callbacks.py::_agent_cbs``):

* the ``tools.desktop_ui`` emitter, so ``desktop_preview`` open/close, ``focus_pane``,
  ``close_terminal`` and the other window events reach the renderer as ``event`` frames;
* the GUI callbacks (``read_preview_callback``, ``drive_preview_callback``,
  ``read_terminal_callback``, ``read_window_below_callback``, ``theme_callback``, ``tour_callback``,
  ``setup_beat_callback``), answered by the renderer through ``preview.read`` / ``preview.act`` /
  ``terminal.read`` / ``window.read`` / ``theme.apply`` / ``theme.pick`` / ``tour`` /
  ``setup_beat`` server requests;
* the process-registry sinks (``on_output`` / ``on_close``) that stream a background process into
  its read-only terminal tab (``agent.terminal.output``) and let ``close_terminal`` drop the tab
  (``terminal.close``).

On the session authority none was installed, so every ``desktop_ui`` tool the session policy
still offers a GUI session failed with "only available in the Hermes desktop app". Here the
emitter and the process sinks are installed once per process; the emitter routes each event to the
turn that raised it (by the turn's bound ``HERMES_SESSION_ID``) through the generation-fenced
execution stream, and the process sinks route by the process's ``session_key`` to the owning
session's event stream (a background process outlives the turn that started it). The callbacks
are installed per turn on the cached agent and restored with the other turn sinks.

Not bridged here: ``connection_callback`` (the manage_connections card). Its card answers through
the ``connection.respond`` RPC, which the session authority does not serve yet, so installing the
callback would show a card whose buttons go nowhere; the tool keeps its off-desktop result.
"""
from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Agent attributes ``wire`` installs for one local-viewer turn (restored by ``local_viewer_events``).
CALLBACK_ATTRS = ("read_terminal_callback", "read_preview_callback", "drive_preview_callback",
                  "read_window_below_callback", "theme_callback", "tour_callback", "setup_beat_callback")

# main: tui_gateway/agent_callbacks.py ``_read_block`` budgets and tui_gateway/server.py's preview.act ladder.
_TERMINAL_READ_TIMEOUT_S = 30
_PREVIEW_READ_TIMEOUT_S = 45
_WINDOW_READ_TIMEOUT_S = 30
_PREVIEW_ACTION_TIMEOUT_S = 45
_PREVIEW_ACTION_PROBE_TIMEOUT_S = 10
_PREVIEW_ACTION_REPROBE_COOLDOWN_S = 30
_PREVIEW_ACTION_BRIDGE_UNAVAILABLE = json.dumps({
    "success": False,
    "error": ("No Hermes Desktop window answered the preview action request. The drive_preview / "
              "annotate_preview bridge is served by the desktop app's renderer, which updates "
              "separately from this backend, so an app build older than the tool has nothing "
              "listening. Update the Hermes Desktop app, open a page with open_preview, and try "
              "again in this session after a short cooldown.")})

# main: tui_gateway/theme_bridge.py and tui_gateway/server.py's tour ladder.
_THEME_TIMEOUT_S = 15
_THEME_PICK_TIMEOUT_S = 300
_THEME_UNANSWERED = {"success": False, "error": "No Hermes Desktop window answered the theme request."}
_TOUR_TIMEOUT_S = 45
_TOUR_PROBE_TIMEOUT_S = 10
_TOUR_BRIDGE_UNAVAILABLE = json.dumps({
    "success": False,
    "error": ("No Hermes Desktop window answered the tour request. The tour is driven by the desktop app's "
              "renderer, which updates separately from this backend, so an app build older than the tour tool "
              "has nothing listening. Update the Hermes Desktop app and start a new session. Do not retry tour "
              "in this session.")})

_lock = threading.Lock()
# turn key (the turn's bound HERMES_SESSION_ID) -> publish(event_type, payload) of the live turn
_turns: dict[str, object] = {}
# canonical session id -> {"state": "answered"|"unanswered", "retry_at": monotonic}
_preview_bridge: dict[str, dict] = {}
# canonical session id -> "answered" | "unanswered" (main: the session record's ``tour_bridge``)
_tour_bridge: dict[str, str] = {}
# process ``session_key`` -> (authority, canonical session id) of the session that last ran a turn with it
_process_routes: dict[str, tuple] = {}
_emitter_installed = False
_process_sinks_installed = False


def _emit(sid, event, payload) -> None:
    """``desktop_ui`` sink: publish *event* on the stream of the turn that raised it."""
    from gateway.session_context import get_session_env
    keys = [str(sid or ""), get_session_env("HERMES_SESSION_ID", "")]
    with _lock:
        publish = next((_turns[key] for key in keys if key and key in _turns), None)
    if publish is None:
        # Raised outside a local-viewer turn (a messaging route, a background thread): no window.
        raise RuntimeError("no Hermes desktop window is attached to this turn")
    publish(event, dict(payload or {}))


def _install_emitter() -> None:
    global _emitter_installed
    if _emitter_installed:
        return
    from tools import desktop_ui
    desktop_ui.set_emitter(_emit)
    _emitter_installed = True


def _publish_process(session, event_type, payload) -> None:
    """Publish a process event on the stream of the session that owns the process."""
    key = str(getattr(session, "session_key", "") or "") if session is not None else ""
    with _lock:
        route = _process_routes.get(key) if key else None
    if route is None:
        raise RuntimeError("no Hermes desktop window owns this process")
    authority, session_id = route
    live = authority.sessions.get(session_id)
    if live is None:
        raise RuntimeError("the session that owns this process is no longer live")
    live.event_stream.publish(session_id, payload, event_type=event_type)


def _install_process_sinks() -> None:
    """main's ``_wire_desktop_sinks``: live output and tab close for background processes (once)."""
    global _process_sinks_installed
    if _process_sinks_installed:
        return
    from tools.process_registry import process_registry
    if getattr(process_registry, "on_output", None) is None:
        process_registry.on_output = lambda session, chunk: _publish_process(
            session, "agent.terminal.output", {"process_id": session.id, "chunk": chunk})
    if getattr(process_registry, "on_close", None) is None:
        process_registry.on_close = lambda session, pid: _publish_process(
            session, "terminal.close", {"process_id": pid})
    _process_sinks_installed = True


def _ask_result(turn, agent, method, params, timeout):
    """One structured server request; the viewer's ``result`` dict, or None when none answered."""
    from gateway.session_sudo_prompt import ask_viewer_result
    authority, session_id, _generation = turn._approval_owner
    live = authority.sessions.get(session_id)
    if live is None:
        return None
    return ask_viewer_result(live.event_stream, session_id, method, dict(params), timeout=timeout,
                             cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)))


def _theme(turn, agent, payload) -> dict:
    """main's ``theme_request``: ``theme.apply`` for values, or a ``theme.pick`` card that waits."""
    if "picker" not in payload:
        answer = _ask(turn, agent, "theme.apply", payload, _THEME_TIMEOUT_S)
        if not answer:
            return dict(_THEME_UNANSWERED)
        try:
            return json.loads(answer)
        except ValueError:
            return {"success": True, "text": answer}
    started = time.monotonic()
    result = _ask_result(turn, agent, "theme.pick", payload, _THEME_PICK_TIMEOUT_S)
    if result is None:
        return {"timed_out": True} if time.monotonic() - started >= _THEME_PICK_TIMEOUT_S else {"cancelled": True}
    if isinstance(result.get("picked"), dict):
        return {"success": True, "picked": result["picked"]}
    if isinstance(result.get("said"), str) and result["said"].strip():
        return {"said": result["said"].strip()}
    return {"cancelled": True}


def _tour(turn, agent, payload) -> str:
    """main's ``_tour_request``: probe once per session, then the full deadline or fail fast."""
    session_id = turn._approval_owner[1]
    with _lock:
        state = _tour_bridge.get(session_id)
    if state == "unanswered":
        return _TOUR_BRIDGE_UNAVAILABLE
    answer = _ask(turn, agent, "tour", payload, _TOUR_TIMEOUT_S if state == "answered" else _TOUR_PROBE_TIMEOUT_S)
    with _lock:
        if answer:
            _tour_bridge[session_id] = "answered"
        elif state != "answered" and not getattr(agent, "_interrupt_requested", False):
            _tour_bridge[session_id] = "unanswered"
    return answer or _TOUR_BRIDGE_UNAVAILABLE


def _setup_beat(turn, agent, payload):
    from agent.onboarding_flow import CARD_TIMEOUT_S
    return _ask_result(turn, agent, "setup_beat", payload, CARD_TIMEOUT_S)


def _ask(turn, agent, method, params, timeout):
    """One ``ValueResult`` server request to the session's attached viewers; "" when none answered."""
    from gateway.session_sudo_prompt import ask_viewer
    authority, session_id, _generation = turn._approval_owner
    live = authority.sessions.get(session_id)
    if live is None:
        return ""
    value = ask_viewer(live.event_stream, session_id, method, dict(params), timeout=timeout,
                       cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _read_block(turn, agent, method, timeout):
    return lambda start=None, count=None: _ask(
        turn, agent, method, {k: v for k, v in (("start", start), ("count", count)) if v is not None}, timeout)


def _preview_action(turn, agent, payload) -> str:
    """main's ``_preview_action_request`` ladder: a short probe until the renderer has answered once,
    then the full deadline; an unanswered probe fails fast for a cooldown instead of stacking waits."""
    session_id = turn._approval_owner[1]
    now = time.monotonic()
    with _lock:
        bridge = _preview_bridge.setdefault(session_id, {})
        state = bridge.get("state")
        if state == "unanswered" and now < bridge.get("retry_at", 0):
            return _PREVIEW_ACTION_BRIDGE_UNAVAILABLE
        if state == "unanswered":
            bridge["retry_at"] = now + _PREVIEW_ACTION_REPROBE_COOLDOWN_S
    answer = _ask(turn, agent, "preview.act", payload,
                  _PREVIEW_ACTION_TIMEOUT_S if state == "answered" else _PREVIEW_ACTION_PROBE_TIMEOUT_S)
    with _lock:
        if answer:
            bridge["state"] = "answered"
        elif bridge.get("state") != "answered" and not getattr(agent, "_interrupt_requested", False):
            bridge["state"] = "unanswered"
            bridge["retry_at"] = time.monotonic() + _PREVIEW_ACTION_REPROBE_COOLDOWN_S
    return answer or _PREVIEW_ACTION_BRIDGE_UNAVAILABLE


def wire(turn, agent):
    """Install the window bridges for one local-viewer turn; returns the registry key to release."""
    owner = getattr(turn, "_approval_owner", None)
    if owner is None:
        return None
    try:
        _install_emitter()
        _install_process_sinks()
    except Exception:  # health: allow BLE001 -- the window bridges are optional; the turn must run
        logger.warning("Desktop UI sink install failed", exc_info=True)
    key = str(getattr(turn._ctx, "session_id", "") or owner[1])
    publish = turn._publish_execution
    process_key = str(getattr(turn._ctx, "session_key", "") or "")
    with _lock:
        _turns[key] = publish
        if process_key:
            _process_routes[process_key] = (owner[0], owner[1])
    agent.read_terminal_callback = _read_block(turn, agent, "terminal.read", _TERMINAL_READ_TIMEOUT_S)
    agent.read_preview_callback = _read_block(turn, agent, "preview.read", _PREVIEW_READ_TIMEOUT_S)
    agent.drive_preview_callback = lambda payload: _preview_action(turn, agent, dict(payload or {}))
    agent.read_window_below_callback = lambda: _ask(turn, agent, "window.read", {}, _WINDOW_READ_TIMEOUT_S)
    agent.theme_callback = lambda payload: _theme(turn, agent, dict(payload or {}))
    agent.tour_callback = lambda payload: _tour(turn, agent, dict(payload or {}))
    agent.setup_beat_callback = lambda payload: _setup_beat(turn, agent, dict(payload or {}))
    return key, publish


def release(token) -> None:
    """Drop this turn's routing entry, unless a newer turn of the same session already replaced it."""
    if not token:
        return
    key, publish = token
    with _lock:
        if _turns.get(key) is publish:
            _turns.pop(key, None)
