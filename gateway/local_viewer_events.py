"""Turn events a local viewer (the classic CLI, TUI, Desktop) reads from the canonical stream.

The classic CLI rendered every agent callback in process: the tool-generating line, the
kawaii thinking text, reasoning, MoA references, hearts, and every status line the agent printed
through ``_print_fn``. On the session authority the agent runs in the gateway, so each of those
callbacks is projected as an execution event (generation-fenced through
``TurnRunner._publish_execution``) and only for a local viewer's turn: a messaging route never
installs them, so its agent behaves exactly as before.

The cached agent is shared across turns. ``wire`` installs the sinks for one turn and returns the
previous values; ``restore`` puts them back when the turn's ``run_conversation`` returns.
"""
from __future__ import annotations

import logging
import time

from gateway import local_desktop_ui

logger = logging.getLogger(__name__)

# Attributes ``wire`` may overwrite on the cached agent for one local-viewer turn.
_TURN_ATTRS = ("tool_gen_callback", "_print_fn", "thinking_callback", "reaction_callback", "reasoning_callback",
               *local_desktop_ui.CALLBACK_ATTRS)
# Private key in ``wire``'s return value: the desktop window bridge registration ``restore`` releases.
_DESKTOP_UI_TOKEN = "__desktop_ui_token__"
# MoA facade progress (agent/moa_loop._RELAY_EVENTS) -> the payload tui_gateway/tool_progress emits.
_MOA_KEYS = {"moa.reference": {"moa_index": "index", "moa_count": "count"},
             "moa.progress": {"moa_refs_done": "refs_done", "moa_refs_total": "refs_total"},
             "moa.phase": {"moa_phase": "phase", "moa_refs_done": "refs_done", "moa_refs_total": "refs_total"},
             "moa.aggregating": {}}


def wire(turn, agent) -> dict:
    """Install this turn's local-viewer sinks on ``agent``; returns what ``restore`` needs."""
    previous = {name: getattr(agent, name, None) for name in _TURN_ATTRS}
    publish = turn._publish_execution
    agent.tool_gen_callback = lambda name: publish("tool.generating", {"name": str(name or "")})

    def print_fn(*args, **kwargs):
        publish("agent.print", {"text": str(kwargs.get("sep", " ")).join(str(a) for a in args)})
    agent._print_fn = print_fn
    # Kawaii spinner text ("" clears it) and hearts; neither changes what the provider is sent.
    agent.thinking_callback = lambda text: publish("thinking.delta", {"text": str(text or "")})
    agent.reaction_callback = lambda kind: publish("reaction", {"kind": str(kind or "")})
    owner = getattr(turn, "_approval_owner", None)
    if owner is not None:
        from gateway.session_usage import seed_compression_count
        seed_compression_count(owner[0].sessions.get(owner[1]), agent)
    if show_reasoning(turn):
        # Only on request: a reasoning sink makes the agent ask the provider for reasoning.
        agent.reasoning_callback = lambda text: publish("reasoning.delta", {"text": str(text or "")})
    # The desktop window bridges (preview/terminal/window reads, preview actions, desktop_ui events).
    previous[_DESKTOP_UI_TOKEN] = local_desktop_ui.wire(turn, agent)
    return previous


def show_reasoning(turn) -> bool:
    """The session's ``show_reasoning`` display flag (``config.set``), off until a viewer sets it."""
    owner = getattr(turn, "_approval_owner", None)
    live = owner[0].sessions.get(owner[1]) if owner is not None else None
    return getattr(live, "show_reasoning", None) is True


def _is_active_session_subagent(turn, session_id: str, payload: dict) -> bool:
    subagent_id = payload.get("subagent_id")
    if not isinstance(subagent_id, str) or not subagent_id:
        return False
    import sys
    registry = sys.modules.get("tools.delegate_tool_registry")
    if registry is None:
        return False
    ctx = getattr(turn, "_ctx", None)
    holder = getattr(ctx, "agent_holder", None)
    turn_agent = holder[0] if isinstance(holder, list) and holder else None
    target_sid = getattr(ctx, "session_id", None) or session_id
    with registry._active_subagents_lock:
        record = registry._active_subagents.get(subagent_id)
        if record is None:
            return False
        if turn_agent is not None and registry._is_descendant_of(record.get("agent"), turn_agent):
            return True
        return bool(target_sid and record.get("owner_agent_session_id") == target_sid)


def _publish_subagent(turn, event_type, payload) -> None:
    owner = getattr(turn, "_approval_owner", None)
    if owner is not None:
        authority, session_id, generation = owner
        executing = getattr(authority, "executing", None)
        if isinstance(executing, dict) and executing.get(session_id) != generation:
            if not _is_active_session_subagent(turn, session_id, payload):
                return
            # A background delegate_task outlives the spawning turn's running claim; publish without borrowing a later turn's execution identity.
            live = authority.sessions.get(session_id)
            stream = getattr(live, "event_stream", None)
            if stream is not None:
                import contextlib
                lock = getattr(stream, "lock", None)
                with lock if lock is not None else contextlib.nullcontext():
                    saved = getattr(stream, "execution", None)
                    if isinstance(saved, dict):
                        stream.execution = {}
                    try:
                        stream.publish(session_id, payload, event_type=event_type)
                    finally:
                        if isinstance(saved, dict):
                            stream.execution = saved
            return
    turn._publish_execution(event_type, payload)


def publish_progress(turn, event_type, name, preview, kwargs) -> None:
    """``subagent.*``, ``reasoning.available`` and the MoA facade events from the agent's progress callback; a classic CLI renders MoA references whatever its reasoning display (main's ``_on_tool_progress``), while other viewers get them only with reasoning shown."""
    if isinstance(event_type, str) and event_type.startswith("subagent."):
        if event_type == "subagent.text":
            return
        from tui_gateway.tool_progress import _subagent_payload
        _publish_subagent(turn, event_type, _subagent_payload(
            event_type, name, preview, kwargs, show_reasoning=show_reasoning(turn)))
        return
    if event_type != "reasoning.available" and event_type not in _MOA_KEYS:
        return
    from gateway.session_cli_turn import classic_cli_turn
    if not (show_reasoning(turn) or (event_type != "reasoning.available" and classic_cli_turn(turn))):
        return
    if event_type == "reasoning.available":
        payload = {"text": str(preview or "")}
    elif event_type == "moa.reference":
        payload = {"label": str(name or ""), "text": str(preview or "")}
    elif event_type in ("moa.aggregating", "moa.phase"):
        payload = {"aggregator": str(name or "")} if name or event_type == "moa.aggregating" else {}
    else:
        payload = {"label": str(name or "")}
    payload.update({out: kwargs[key] for key, out in _MOA_KEYS.get(event_type, {}).items()
                    if kwargs.get(key) is not None})
    if event_type == "moa.progress" and not {"refs_done", "refs_total"} <= set(payload):
        return
    if event_type == "moa.phase" and not payload.get("phase"):
        return
    turn._publish_execution(event_type, payload)


def restore(agent, previous: dict | None) -> None:
    if not previous:
        return
    token = previous.pop(_DESKTOP_UI_TOKEN, None)
    if token is not None:
        local_desktop_ui.release(token)
    for name, value in previous.items():
        setattr(agent, name, value)


def note_tool_start(turn, call_id, tool_name=None, args=None) -> None:
    """Monotonic start time of one call (``tool.complete.duration_s``) and, for a write-capable
    tool, the before-state of its local files: the agent fires the start callback BEFORE the tool
    writes, so the gateway owns the snapshot the classic CLI used to take in process."""
    key = str(call_id or "")
    turn.__dict__.setdefault("_local_tool_starts", {})[key] = time.monotonic()
    if tool_name is None:
        return
    try:
        from agent.display import capture_local_edit_snapshot
        snapshot = capture_local_edit_snapshot(tool_name, args if isinstance(args, dict) else {},
                                               task_id=turn._ctx.session_id)
    except Exception:  # health: allow BLE001 -- display-only preview; a failed snapshot only drops the diff
        logger.debug("Edit snapshot capture failed for %s", tool_name, exc_info=True)
        snapshot = None
    if snapshot is not None:
        turn.__dict__.setdefault("_local_edit_snapshots", {})[key] = snapshot


def complete_fields(turn, call_id, tool_name=None, args=None, result=None) -> dict:
    """``duration_s``, ``inline_diff`` (the rendered edit diff main's CLI printed after a write,
    ``None`` when there is none) and, for the todo tool, the full ``todos`` snapshot."""
    key = str(call_id or "")
    started = turn.__dict__.setdefault("_local_tool_starts", {}).pop(key, None)
    fields = {"duration_s": (time.monotonic() - started) if started is not None else None}
    if tool_name is None:
        return fields
    snapshot = turn.__dict__.setdefault("_local_edit_snapshots", {}).pop(key, None)
    rendered: list[str] = []
    try:
        from agent.display import render_edit_diff_with_delta
        render_edit_diff_with_delta(tool_name, result, function_args=args if isinstance(args, dict) else {},
                                    snapshot=snapshot, print_fn=rendered.append)
    except Exception:  # health: allow BLE001 -- display-only preview, as main's CLI _on_tool_complete
        logger.debug("Edit diff preview failed for %s", tool_name, exc_info=True)
        rendered = []
    fields["inline_diff"] = "\n".join(rendered) if rendered else None
    todo = todo_state(tool_name, result)
    if todo is not None:
        fields.update(todo)
    return fields


def todo_state(tool_name, result) -> dict | None:
    """The client-safe ``{todos, revision}`` snapshot a todo tool's result carries, else None."""
    from tools.todo_tool import is_todo_tool_name
    if not is_todo_tool_name(tool_name):
        return None
    import json
    from tui_gateway.tool_progress import _normalize_todo_state
    try:
        return _normalize_todo_state(json.loads(result) if isinstance(result, str) else result)
    except (TypeError, ValueError):
        return None


def prompt_timeout_s(turn, kind) -> int | None:
    """``timeout_s`` of a local viewer's approval / clarify prompt: the window the gateway's own wait
    enforces (``approvals.timeout`` / the clarify timeout), so the classic CLI's modal shows the
    same countdown and gives up when the gateway does. ``None`` on a messaging turn (frame
    unchanged) and for an unlimited clarify wait."""
    from gateway.run_turn_progress import local_viewer_turn
    if not local_viewer_turn(turn):
        return None
    # Both readers fall back to their defaults themselves; they never raise.
    if kind == "approval":
        from tools.approval_context import _get_approval_timeout
        seconds = int(_get_approval_timeout())
    else:
        from tools.clarify_gateway import get_clarify_timeout
        seconds = int(get_clarify_timeout())
    return seconds if seconds > 0 else None
