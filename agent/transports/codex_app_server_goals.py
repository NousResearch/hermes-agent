"""Consume Codex-owned automatic turns; never synthesize a continuation or judge reply."""
from __future__ import annotations

import time
from contextlib import suppress

from hermes_cli.codex_goals import native_objective, pause_native_goal, record_native_goal, native_goal_resource_limited
from hermes_cli.goals import load_goal


def _native_request(session, method, **params):
    result = session._client.request(method, {"threadId": session._thread_id, **params}, timeout=15)
    goal = result.get("goal")
    if method == "thread/goal/get" and goal is None:
        return None
    if method != "thread/goal/clear" and (not isinstance(goal, dict) or goal.get("threadId") != session._thread_id):
        raise RuntimeError(f"{method} returned no matching native Goal")
    return goal


def _merge_result(aggregate, turn):
    aggregate.final_text = turn.final_text
    aggregate.turn_id, aggregate.thread_id = turn.turn_id, turn.thread_id
    aggregate.projected_messages.extend(turn.projected_messages)
    aggregate.tool_iterations += turn.tool_iterations
    aggregate.interrupted, aggregate.error = turn.interrupted, turn.error
    aggregate.should_retire = turn.should_retire
    aggregate.compacted = aggregate.compacted or turn.compacted
    aggregate.model_context_window = turn.model_context_window or aggregate.model_context_window
    aggregate.token_usage_last = turn.token_usage_last
    aggregate.usage_recorded, aggregate.usage_result = turn.usage_recorded, turn.usage_result
    aggregate.native_turns += 1


def _next_turn(session, result, *, idle_timeout, control):
    """Wait only for a native turn/started; no second turn/start request is ever sent."""
    deadline = time.monotonic() + idle_timeout
    while time.monotonic() < deadline:
        control()
        if session._interrupt_event.is_set():
            result.interrupted = result.should_retire = True
            return None
        if session._subprocess_died(result, session._client):
            return None
        request = session._client.take_server_request(timeout=0)
        if request is not None:
            session._handle_server_request(request)
        note = session._client.take_notification(timeout=.25)
        if note is None:
            continue
        params = note.get("params") or {}
        if params.get("threadId") != session._thread_id:
            continue
        if note.get("method") == "turn/started":
            return {"turn": params.get("turn") or {}}
        if note.get("method") == "thread/goal/updated" and (params.get("goal") or {}).get("status") != "active":
            return None
    result.error = f"Codex native Goal inactive for {idle_timeout}s awaiting automatic continuation; task incomplete"
    result.interrupted = result.should_retire = True
    return None


def _attach_native_goal(session, state, objective, *, initial_turn):
    """Resume/revise a bound ledger, or explicitly replace a finished/cleared Goal."""
    existing = _native_request(session, "thread/goal/get")
    if initial_turn is not None:
        native = existing  # Already started by the model: no duplicate kickoff or set.
    elif state.native_goal is not None:
        previous = state.native_goal
        if existing is None or existing.get("objective") != previous.get("objective"):
            raise RuntimeError("Bound native Goal changed or disappeared; refusing to reset its ledger")
        if existing.get("tokenBudget") != state.token_budget:
            raise RuntimeError("Native Goal budget changed; refusing an implicit budget override")
        if native_goal_resource_limited(existing):
            return existing
        params = {"status": "active"}
        if existing["objective"] != objective:
            params["objective"] = objective  # Text revision only; native usage stays cumulative.
        native = _native_request(session, "thread/goal/set", **params)
    else:
        if existing is not None:
            if existing.get("status") == "active":
                raise RuntimeError("Another native Goal is active; refusing to replace it")
            _native_request(session, "thread/goal/clear")
        native = _native_request(session, "thread/goal/set", status="active",
                                 objective=objective, tokenBudget=state.token_budget)
    if native is None or native.get("objective") != objective:
        raise RuntimeError("Native Goal changed before its continuation was attached")
    return native


def run_native_goal(session, user_input, *, session_id, state, on_turn=None, interrupt_requested=None,
                    initial_turn=None, **options):
    from agent.transports.codex_app_server_session import TurnResult
    aggregate = TurnResult(thread_id=session.ensure_started())
    goal_id = state.goal_id
    objective = native_objective(state)
    if state.native_goal and state.native_goal.get("threadId") != aggregate.thread_id:
        raise RuntimeError("Native Goal belongs to a different thread; refusing to reset progress or budget")
    native = _attach_native_goal(session, state, objective, initial_turn=initial_turn)
    budget = native.get("tokenBudget")
    if initial_turn is None and native_goal_resource_limited(native):
        reason = (f"Codex Goal {native['status']}: tokens {native.get('tokensUsed', 0)}/{budget}; "
                  "no work started and no budget reset; explicit budget authorization is required")
        pause_native_goal(session_id, goal_id, reason)
        record_native_goal(session_id, goal_id, native)
        aggregate.error, aggregate.interrupted, aggregate.should_retire = reason, True, True
        return aggregate
    record_native_goal(session_id, goal_id, native)

    def control():
        current = load_goal(session_id)
        changed = (current is None or current.goal_id != goal_id or current.status != "active"
                   or native_objective(current) != objective)
        if changed or session._interrupt_event.is_set() or (interrupt_requested and interrupt_requested()):
            with suppress(Exception):
                _native_request(session, "thread/goal/set", status="paused")
            session.request_interrupt()
        return False

    session._native_goal_control = control
    session._native_goal_running = True
    try:
        turn = initial_turn if initial_turn is not None else session.run_turn(user_input, **options)
        aggregate.submitted_user_text = turn.submitted_user_text
        while True:
            native = _native_request(session, "thread/goal/get")
            if not turn.projected_messages and not turn.error and not turn.interrupted:
                turn.error = "Codex native Goal returned no transcript; task incomplete"
            if on_turn is not None:
                continuing = native["status"] == "active" and bool(turn.tool_iterations) and not turn.error and not turn.interrupted
                on_turn(turn, continuing)
            _merge_result(aggregate, turn)
            if turn.error or turn.interrupted:
                pause_native_goal(session_id, goal_id, turn.error or "Codex turn interrupted; task incomplete")
            mirror = record_native_goal(session_id, goal_id, native, completed_turn=not turn.error and not turn.interrupted)
            if turn.error or turn.interrupted or mirror is None or mirror.status != "active":
                break
            if not turn.tool_iterations:
                pause_native_goal(session_id, goal_id, "Codex returned without tool work; not forcing an empty continuation")
                break
            ts = _next_turn(session, aggregate, idle_timeout=options["idle_timeout"], control=control)
            if ts is None:
                break
            turn = TurnResult(thread_id=aggregate.thread_id)
            session._run_started_turn(turn, ts, options["turn_timeout"], .25, 90,
                                      idle_timeout=options["idle_timeout"])
        current = load_goal(session_id)
        if aggregate.error or aggregate.interrupted or current is None or current.status != "done":
            with suppress(Exception):
                _native_request(session, "thread/goal/set", status="paused")
            reason = aggregate.error or (current.paused_reason if current is not None else "Goal cleared/replaced") or "Goal interrupted"
            pause_native_goal(session_id, goal_id, reason)
            aggregate.error = aggregate.error or f"Codex native Goal paused, not completed: {reason}"
            # Native scheduling may already have accepted its next turn. Retiring the
            # client stops that turn too instead of leaving invisible background work.
            aggregate.interrupted = aggregate.should_retire = True
        return aggregate
    except Exception:
        pause_native_goal(session_id, goal_id, "Native Goal transport failed; no automatic replay")
        with suppress(Exception):
            _native_request(session, "thread/goal/set", status="paused")
        raise
    finally:
        session._native_goal_control = None
        session._native_goal_running = False
        session._interrupt_event.clear()
