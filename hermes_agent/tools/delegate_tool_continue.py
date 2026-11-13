"""delegate_task(action='continue'): run a FINISHED child again on its own conversation.

The child is rebuilt from what its registry record retained when it finished (``_resume_spec``: route incl. profile
home, toolsets, the goal/context its system prompt came from, model, session id, and the provider and endpoint it ran
on, which the rebuild must resolve to again), its history is loaded from its saved transcript, and ``message`` is its
next user turn. It keeps its subagent_id AND its session id (the transcript
appends, like ``--resume``), and runs as a new one-task call through the normal delegate_task dispatch, so delivery,
the stale monitor and restart recovery are the spawn path's. After a restart or crash the registry is empty and the
durable record (``delegate_tool_resumable``) is the source: the route is re-resolved from the task's own routing
request and must land on the same endpoint. Either way the child resumes in the cwd and worktree it stopped in.
A continue is a normal dispatch: closed-chat refusal, capacity, depth and the spawn pause all apply."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_agent.tools.registry import tool_error

logger = logging.getLogger("hermes_agent.tools.delegate_tool")  # log-record parity with the origin module

_NOT_FOUND = (
    "No finished subagent '{sid}' in this conversation's spawn tree can be continued (unknown id, not yours, or no "
    "longer retained). Spawn a new task instead."
)


def _claim_resume(sid: str, parent_agent: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Take the finished child's resume spec out of its record, so a second continue can never run the same
    conversation twice at once (the attribution stub stays). ``(spec, None)`` or ``(None, error)``."""
    from hermes_agent.tools.delegate_tool_registry import (
        _active_subagents, _active_subagents_lock, _owns_subagent_record, _recent_subagents,
    )
    with _active_subagents_lock:
        live, record = _active_subagents.get(sid), _recent_subagents.get(sid)
    if live is not None and _owns_subagent_record(live, parent_agent):
        return None, f"Subagent '{sid}' is still running: use action='steer' to send it this message."
    if record is None or not _owns_subagent_record(record, parent_agent):
        return None, _NOT_FOUND.format(sid=sid)
    with _active_subagents_lock:
        if _recent_subagents.get(sid) is not record:
            return None, _NOT_FOUND.format(sid=sid)
        if record.get("state") != "resumable":
            return None, (f"Subagent '{sid}' is still stopping: its previous run has not exited yet. "
                          "Try again shortly; if this persists, spawn a new task.")
        spec = record.pop("resume", None)
    return (spec, None) if spec else (None, _NOT_FOUND.format(sid=sid))


def _release_resume(sid: str, spec: dict[str, Any]) -> None:
    """Hand an unused spec back (the continue was refused before anything ran)."""
    from hermes_agent.tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        record = _recent_subagents.get(sid)
        if record is not None:
            record.setdefault("resume", spec)


@contextlib.contextmanager
def _transcript_store(parent_agent: Any) -> Iterator[Any]:
    """A handle on the db FILE the parent's children write to (``_open_child_session_db``), or None. Never the
    parent's own object: a gateway parent holds an async wrapper whose methods are coroutines."""
    db_path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    if not isinstance(db_path, (str, Path)):
        yield None
        return
    from hermes_agent.hermes_state_registry import acquire, release_or_close
    db = acquire(db_path)
    try:
        yield db
    finally:
        release_or_close(db)


def _refusal(parent_agent: Any) -> Optional[str]:
    from hermes_agent.agent.oneshot_footprint import agent_is_oneshot
    from hermes_agent.agent.runtime_session_store import RuntimeSessionStore, is_worker_process
    from hermes_agent.tools.delegate_tool_registry import is_spawn_paused
    if agent_is_oneshot(parent_agent):
        # The same frozen fact that strips live control from its schema: its children are joined inside the call.
        return "action='continue' is unavailable in one-shot sessions. Spawn a new task instead."
    if is_worker_process() or isinstance(getattr(parent_agent, "_session_db", None), RuntimeSessionStore):
        # A managed worker reserves every child's row through its owner; resuming one is not supported there.
        return "action='continue' is unavailable in managed worker sessions. Spawn a new task instead."
    if is_spawn_paused():
        return ("Delegation spawning is paused. Clear the pause via the TUI (`p` in /agents) or the "
                "`delegation.pause` RPC before retrying.")
    from hermes_agent.tools.delegate_tool import _get_max_spawn_depth
    depth, max_spawn = getattr(parent_agent, "_delegate_depth", 0), _get_max_spawn_depth()
    if isinstance(depth, int) and depth >= max_spawn:  # the same limit a spawn from here hits
        return f"Delegation depth limit reached (depth={depth}, max_spawn_depth={max_spawn})."
    return None


def _durable_spec(sid: str, record: dict[str, Any], parent_agent: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """A resume spec from the durable record: its route re-resolved from the task's own routing request."""
    from hermes_agent.tools.delegate_tool import _load_config
    from hermes_agent.tools.delegate_tool_routes import resolve_task_routes
    request = record.get("route_request")
    if not isinstance(request, dict) or not record.get("session_id"):
        return None, (f"Subagent '{sid}' ran on a route that cannot be rebuilt after a restart. "
                      "Spawn a new task instead.")
    routes, err = resolve_task_routes([{"goal": record.get("goal"), **request}], _load_config(), parent_agent)
    if err or not routes or routes[0].error:
        return None, err or (routes[0].error if routes else "no route")
    return {**record, "route": routes[0], "endpoint": tuple(record.get("endpoint") or ()) or None}, None


def _claim(sid: str, parent_agent: Any, auto: bool = False,
           ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[str]]:
    """``(spec, durable_record, error)``: the in-process record while it is retained, else the durable one. The
    durable claim is the single gate for both: a refusal there refuses the continue even when this caller holds the
    in-process spec (a second continue that fell through to the row, or one from another process, owns it). *auto*:
    the scheduler already claimed the row (``delegate_tool_auto_resume._claim``); this takes that claim over."""
    from hermes_agent.tools.delegate_tool_resumable import claim as claim_durable
    spec, err = _claim_resume(sid, parent_agent)
    if spec is None and err != _NOT_FOUND.format(sid=sid):
        return None, None, err
    if auto:
        from hermes_agent.tools.delegate_tool_auto_resume import CANCELLED, take_claim
        from hermes_agent.tools.delegate_tool_resumable import _db_path_for
        record = take_claim(_db_path_for(parent_agent), sid)
        durable_err = None if record is not None else CANCELLED
    else:
        record, durable_err = claim_durable(parent_agent, sid)
    if durable_err:  # the durable row is the one gate: another continue holds it (here or in another process)
        if spec is not None:
            _release_resume(sid, spec)
        return None, None, durable_err
    if spec is None and record is not None:
        spec, err = _durable_spec(sid, record, parent_agent)
        if spec is None:
            from hermes_agent.tools.delegate_tool_resumable import release
            release(parent_agent, sid, record)
            return None, None, err
    return (spec, record, None) if spec else (None, None, err)


def _segment_tip(db: Any, sid: str, session_id: str) -> str:
    """The newest segment of the child's conversation: with ``compression.in_place: false`` a rotation after the
    record was written moved the child on from *session_id* (``get_compression_tip`` skips delegate children)."""
    if db is None:
        return session_id
    from hermes_agent.tools.delegate_tool_sessions import stored_child_session
    return stored_child_session(db, sid, [session_id]) or session_id


def _start(
    sid: str, message: str, spec: dict[str, Any], parent_agent: Any, background: bool,
    record: Optional[dict[str, Any]] = None, run_origin: str = "continue",
) -> tuple[Optional[str], Optional[str]]:
    """Rebuild, hydrate and dispatch the continued child: ``(tool result JSON, None)`` or ``(None, error)`` when
    it was refused before anything ran."""
    from hermes_agent.tools.delegate_tool import (
        DEFAULT_MAX_ITERATIONS, _build_children, _get_max_concurrent_children, _load_config, _oneshot_spawn_budget,
        _parent_live_home,
    )
    from hermes_agent.tools.delegate_tool_dispatch import _Batch, _capture_origin, _run_batch
    from hermes_agent.tools.delegation_live_log import create_live_transcripts

    route = spec["route"]
    if run_origin == "auto":  # the scheduler has no turn to join a result into: it only ever detaches
        from hermes_agent.tools.delegate_tool_dispatch import _resolve_async_wake_sid
        origin = _capture_origin()
        if _resolve_async_wake_sid(origin[0], origin[4]) is None:
            return None, "Not continued: this session cannot receive a detached result."
    with _transcript_store(parent_agent) as db:
        session_id = _segment_tip(db, sid, spec["session_id"])
        history = db.get_messages_as_conversation(session_id, repair_alternation=True) if db is not None else []
        if not history:
            return None, f"Subagent '{sid}' has no saved transcript to continue from. Spawn a new task instead."
        err = _oneshot_spawn_budget(parent_agent, 1)
        if err:
            return None, err
        task_list = [{"goal": message}]  # the run's user turn; the system prompt is rebuilt from the original goal
        live_home = _parent_live_home(parent_agent)
        live_id, writers, paths = create_live_transcripts(
            task_list, None, model=spec.get("model"), provider=route.creds.get("provider"), home=live_home,
        )
        origin = _capture_origin()  # before the build: child construction clobbers the session-id binding
        children, err = _build_children(
            [{"goal": spec["goal"], "context": spec.get("context")}], [None], [route], top_role="leaf",
            max_iterations=_load_config().get("max_iterations", DEFAULT_MAX_ITERATIONS), parent_agent=parent_agent,
            live_deleg_id=live_id, live_writers=writers,
            rebuild=[{"toolsets": spec.get("toolsets") or None, "model": spec.get("model") or route.creds["model"],
                      "subagent_id": sid, "session_id": session_id, "expect_endpoint": spec.get("endpoint")}],
        )
        if err:
            return None, err
        db.reopen_session(session_id)  # its close() ended it; the continued child's close() ends it again
        children[0][2]._delegate_continue_history = history
        children[0][2]._delegate_run_origin = run_origin  # a parent's own continue ends automatic resumes of it
        from hermes_agent.tools.delegate_tool_resumable import hold_claim, last_workspace
        children[0][2]._delegate_continue_workspace = last_workspace(parent_agent, sid)
        if record is not None:  # a run that fails before it records itself hands the claim back
            hold_claim(children[0][2], parent_agent, sid, record)
        if run_origin == "auto":  # the last word is the parent's: a stop, /stop, /new or replacement since the claim
            from hermes_agent.tools.delegate_tool_auto_resume import confirm_start
            from hermes_agent.tools.delegate_tool_resumable import _db_path_for
            if cancelled := confirm_start(_db_path_for(parent_agent), sid):
                from hermes_agent.tools.delegate_tool_child_run import _close_child, _detach_child
                children[0][2]._delegate_claim = None
                _detach_child(parent_agent, children[0][2])
                _close_child(children[0][2], "Failed to close a cancelled continue's child")
                return None, cancelled
    batch = _Batch(
        task_list, [(0, task_list[0], children[0][2])], parent_agent, route.creds, None, "leaf",
        _get_max_concurrent_children(), live_id, writers, paths, *origin, time.monotonic(), live_home=live_home,
        detached_only=run_origin == "auto",
    )
    return _run_batch(batch, background), None


def continue_subagent(subagent_id: Optional[str], message: Optional[str], parent_agent: Any, *, background: bool,
                      auto: bool = False) -> str:
    """delegate_task(action='continue'): validate ownership and finished state, then run ``message`` as the
    child's next turn. Returns the usual dispatch handle (new delegation_id, same subagent_id) or, where the
    session joins children inline, the joined result. ``auto``: the ``delegation.auto_resume`` scheduler's
    continue (``delegate_tool_auto_resume``), the same claim and dispatch on a stand-in for the parent."""
    sid, text = (subagent_id or "").strip(), (message or "").strip()
    if not sid or not text:
        return tool_error("action='continue' requires subagent_id (a finished child of yours) and a non-empty "
                          "'message' (its next instruction).")
    refusal = _refusal(parent_agent)
    if refusal:
        return tool_error(refusal)
    spec, record, err = _claim(sid, parent_agent, auto)
    if spec is None:
        return tool_error(err or _NOT_FOUND.format(sid=sid))
    try:
        result, err = _start(sid, text, spec, parent_agent, background, record, "auto" if auto else "continue")
    except Exception as exc:
        logger.warning("delegate_task continue of %s failed before dispatch", sid, exc_info=True)
        result, err = None, f"Could not continue subagent '{sid}': {exc}"
    if result is None:
        _release_resume(sid, spec)
        if record is not None:
            from hermes_agent.tools.delegate_tool_resumable import release
            release(parent_agent, sid, record)
        return tool_error(err)
    return result

