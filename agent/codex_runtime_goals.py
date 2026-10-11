"""Native Goal orchestration at the Codex runtime boundary, not a second agent loop."""
from __future__ import annotations


def run_app_server_work(agent, user_message, *, messages, **wire_options):
    from hermes_cli.config import load_config
    from hermes_cli.goals import load_goal
    from agent.codex_runtime import (_persist_projected_messages, _record_codex_app_server_usage,
                                    _record_codex_app_server_compaction, _store_codex_thread_id)
    cfg = load_config() or {}
    options = cfg.get("agent") or {}
    total = options.get("codex_turn_timeout", 600)
    idle = options.get("codex_idle_timeout", 1800)
    if (isinstance(total, bool) or isinstance(idle, bool) or not isinstance(total, (int, float))
            or not isinstance(idle, (int, float)) or total < 0 or idle <= 0):
        raise ValueError("agent.codex_turn_timeout must be >= 0; agent.codex_idle_timeout must be > 0")
    wire_options.update(turn_timeout=total, idle_timeout=idle)
    sid = getattr(agent, "session_id", None)
    state = load_goal(sid) if sid else None
    initial_turn = None
    if state is None or state.runtime != "codex" or state.status != "active":
        initial_turn = agent._codex_session.run_turn(user_input=user_message, **wire_options)
        if (not sid or (cfg.get("goals") or {}).get("runtime") != "codex"
                or initial_turn.error or initial_turn.interrupted):
            return initial_turn
        from agent.transports.codex_app_server_goals import _native_request
        from hermes_cli.codex_goals import adopt_native_goal
        try:
            native = _native_request(agent._codex_session, "thread/goal/get")
            if native is None:
                return initial_turn
            previous_native = state.native_goal if state is not None else None
            if (native.get("status") != "active" and previous_native
                    and previous_native.get("threadId") == native.get("threadId")
                    and previous_native.get("objective") == native.get("objective")):
                return initial_turn  # Ordinary questions do not revive a terminal Goal.
            state = adopt_native_goal(sid, native, thread_id=initial_turn.thread_id, previous=state)
        except Exception:
            # A native scheduler may already be running. Fail closed rather than
            # returning an ordinary success and leaving invisible background work.
            agent._codex_session.close()
            raise
    from agent.transports.codex_app_server_goals import run_native_goal

    def commit_turn(turn, continuing):
        if not turn.projected_messages:
            return  # No rows to persist; native terminal/error classification owns this outcome.
        if not _persist_projected_messages(agent, turn, messages):
            raise RuntimeError("Native Goal turn was not durably mirrored; pausing instead of continuing")
        _store_codex_thread_id(agent, turn.thread_id)
        turn.projected_messages = []  # Already appended/flushed; aggregate must not insert them twice.
        if continuing:
            # Native Goal turns bypass the ordinary finalizer. Consume their real compaction
            # boundary before anchoring usage, rather than retaining a stale mirror estimate.
            _record_codex_app_server_compaction(agent, turn)
            turn.compacted = False  # Recorded here; the aggregate finalizer must not record it again.
            turn.usage_result = _record_codex_app_server_usage(agent, turn, messages=messages)
            turn.usage_recorded = True

    try:
        return run_native_goal(
            agent._codex_session, user_message, session_id=sid, state=state, on_turn=commit_turn,
            initial_turn=initial_turn,
            interrupt_requested=lambda: bool(getattr(agent, "_interrupt_requested", False)), **wire_options,
        )
    except Exception:
        from hermes_cli.codex_goals import pause_native_goal
        pause_native_goal(sid, state.goal_id, "Native Goal setup/transport failed; no automatic replay")
        agent._codex_session.close()
        raise


def native_resume_thread(agent):
    """A resumed native Goal stays on its budget-bearing thread even after /stop."""
    from hermes_cli.goals import load_goal
    sid = getattr(agent, "session_id", None)
    state = load_goal(sid) if sid else None
    if state is not None and state.runtime == "codex" and state.status == "active":
        return (state.native_goal or {}).get("threadId")
    return None
