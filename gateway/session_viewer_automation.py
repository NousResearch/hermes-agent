"""``session.automation``: the classic CLI's /goal, /subgoal, /loop and /heartbeat managers, run here.

Main's HermesCLI held a ``GoalManager`` / ``LoopManager`` / ``HeartbeatManager`` in its own process:
they wrote ``state_meta`` through a read-write SessionDB and called the goal judge (an auxiliary
model). Over the session authority the gateway is the single writer and runs every model call, so
the viewer's managers are thin proxies (``hermes_cli/gateway_automation.py``) and each mutation or
judgement is one call of this verb, executed against the session's physical row under the owning
profile's runtime scope and the session's own workspace (quality gates run in its cwd).

The viewer keeps main's timing: it calls ``evaluate_after_turn`` / ``complete_tick`` after its turn
returns and queues the continuation itself. A session whose viewer drives its automation this way is
recorded in ``authority.viewer_automation``; the gateway's own post-turn goal/loop hooks skip it, so
a turn is never judged twice or continued twice.
"""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

_CALLS = {
    "goal": frozenset({"state", "set", "pause", "resume", "clear", "mark_done", "add_subgoal", "remove_subgoal",
                       "clear_subgoals", "add_gate", "remove_gate", "clear_gates", "wait_on", "wait_on_session",
                       "wait_for_seconds", "stop_waiting", "is_waiting", "evaluate_after_turn", "command"}),
    "loop": frozenset({"state", "set", "pause", "resume", "clear", "fire_tick", "abandon_tick", "complete_tick",
                       "goal_blocks_tick"}),
    "heartbeat": frozenset({"state", "set", "pause", "resume", "clear", "due_prompt", "abandon_fire"}),
}
_FIELDS = frozenset({"session_id", "profile", "kind", "call", "args", "kwargs"})
_ERRORS = (RuntimeError, ValueError, IndexError)


def viewer_driven(authority, session_id) -> bool:
    """Whether a local viewer drives this session's goal/loop judging (see the module doc): only while
    that viewer's connection still holds its subscription, so a TUI or Desktop attached after it left
    gets the gateway's own post-turn hooks again."""
    connection = getattr(authority, "viewer_automation", {}).get(session_id) if authority is not None else None
    return bool(connection is not None and connection.subscriptions.get(session_id))


def _max_turns(runner):
    configured = getattr(runner, "_goal_max_turns_from_config", None)
    if configured is not None:
        return configured()
    from hermes_cli.goals import DEFAULT_MAX_TURNS
    return DEFAULT_MAX_TURNS


def _manager(runner, kind, target):
    if kind == "goal":
        from hermes_cli.goals import GoalManager
        return GoalManager(session_id=target, default_max_turns=_max_turns(runner))
    if kind == "loop":
        from hermes_cli.loops import LoopManager
        return LoopManager(session_id=target)
    from hermes_cli.heartbeat import HeartbeatManager
    return HeartbeatManager(session_id=target)


def _wire(value):
    """A manager return value as JSON: states and gates keep their own serialisation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_json"):
        return {"__state__": value.to_json()}
    if hasattr(value, "to_dict"):
        return {"__gate__": value.to_dict()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    return str(value)


def _special(mgr, kind, target, call, args, kwargs):
    """Calls that are more than one manager method; ``None`` when ``call`` is a plain method."""
    if kind == "goal" and call == "command":
        # The whole /goal command (main's ``dispatch_goal_command``): ``/goal draft`` drafts its
        # contract with the auxiliary model, which runs here, never in the viewer.
        from hermes_cli.goal_command import dispatch_goal_command
        arg, last_user = (list(args) + ["", None])[:2]
        return dispatch_goal_command(mgr, str(arg or ""), authorize_gate=lambda: None, last_user_message=last_user)
    if kind == "goal" and call == "evaluate_after_turn":
        # Background processes and delegations live in this process now (the turn ran here).
        from hermes_cli.goals import count_active_delegations, gather_background_processes
        try:
            processes = gather_background_processes(owner_task_id=target)
            delegations = count_active_delegations(target)
        except Exception:  # health: allow BLE001 -- the judge degrades to no background block, as main's hook did
            processes, delegations = None, 0
        return mgr.evaluate_after_turn(str((args or [""])[0] or ""), user_initiated=True,
                                       background_processes=processes, active_delegations=delegations)
    if kind == "goal" and call == "set" and isinstance(kwargs.get("contract"), dict):
        from hermes_cli.goals import GoalContract
        kwargs = {**kwargs, "contract": GoalContract.from_dict(kwargs["contract"])}
        return mgr.set(*args, **kwargs)
    if kind == "loop" and call == "goal_blocks_tick":
        from hermes_cli.loops import goal_blocks_loop_tick
        return goal_blocks_loop_tick(target)
    if call == "state":
        return None
    return _NOT_SPECIAL


_NOT_SPECIAL = object()


def invoke(runner, kind, target, call, args, kwargs) -> dict:
    """Run one call against ``target``'s persisted state: ``{state, value}`` or ``{state, error}``."""
    mgr = _manager(runner, kind, target)
    out: dict
    try:
        value = _special(mgr, kind, target, call, args, kwargs)
        if value is _NOT_SPECIAL:
            value = getattr(mgr, call)(*args, **kwargs)
        out = {"value": _wire(value)}
    except _ERRORS as exc:
        out = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    state = mgr.state
    out["state"] = state.to_json() if state is not None else None
    return out


def _parse(params):
    if set(params) - _FIELDS:
        raise RuntimeStoreError("invalid_params")
    kind, call = params.get("kind"), params.get("call")
    args, kwargs = params.get("args") or [], params.get("kwargs") or {}
    if (kind not in _CALLS or call not in _CALLS[kind] or not isinstance(args, list)
            or not isinstance(kwargs, dict) or not all(isinstance(k, str) and not k.startswith("_") for k in kwargs)):
        raise RuntimeStoreError("invalid_params")
    return kind, call, args, kwargs


async def automation(connection, ref, params):
    kind, call, args, kwargs = _parse(params)
    if not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError("invalid_params")
    from gateway.session_busy_controls import authorize
    authorize(connection, ref, params, "session:control")
    authority = connection.authority
    live = authority.sessions[ref.session_id]
    from gateway.session_ancillary import physical_target
    target = physical_target(authority, ref)
    if not hasattr(authority, "viewer_automation"):
        authority.viewer_automation = {}
    authority.viewer_automation[ref.session_id] = connection
    runner = authority.runner
    home = Path(authority.profile_id)

    def run():
        from gateway.run import _profile_runtime_scope
        from gateway.session_commands import _session_workspace
        with _profile_runtime_scope(home), _session_workspace(runner, live.source):
            return invoke(runner, kind, target, call, args, kwargs)
    # The judge and gates block for seconds: off the loop, with the caller's context (profile scope).
    return await asyncio.to_thread(run)
