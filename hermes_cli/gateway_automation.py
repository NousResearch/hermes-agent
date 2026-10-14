"""The classic CLI's /goal, /subgoal, /loop and /heartbeat managers, backed by the gateway session.

Main's ``HermesCLI`` built ``GoalManager`` / ``LoopManager`` / ``HeartbeatManager`` in-process: each
mutation wrote ``state_meta`` through a read-write SessionDB and ``evaluate_after_turn`` called the goal
judge model from the CLI pid. Rewired, the gateway is the single writer and runs every model call, so
these are thin proxies (the GatewayAgent / GatewaySessionStore seam): every mutation and judgement is
one ``session.automation`` call, and the persisted state the gateway returns is cached here so the
status bar, the dock and main's read-only helpers (``status_line``, ``is_due``, ``render_gates``…) keep
running main's own code, unchanged, against the same state object.
"""
from __future__ import annotations

import logging

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)

_ERRORS = {"RuntimeError": RuntimeError, "ValueError": ValueError, "IndexError": IndexError}


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__state__"}:
            return value["__state__"]
        if set(value) == {"__gate__"}:
            from hermes_cli.goals import GoalGate
            return GoalGate.from_dict(value["__gate__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


class _Remote:
    """Shared plumbing: ``self._state`` mirrors the gateway's row; ``_rpc`` runs one call there."""

    kind = ""
    _state_cls: type = type(None)

    def __init__(self, session_id, connection, *, load=True):
        self.session_id = session_id
        self._connection = connection
        self._state = None
        if load:
            self.refresh()

    @property
    def state(self):
        return self._state

    def _adopt(self, raw):
        state = self._state_cls.from_json(raw) if raw else None
        self._state = None if state is None or state.status == "cleared" else state
        return self._state

    def _rpc(self, call, *args, **kwargs):
        try:
            reply = self._connection.call("session.automation", session_id=self.session_id, kind=self.kind,
                                          call=call, args=list(args), kwargs=kwargs)
        except GatewayClientError as exc:
            # Main's managers swallowed a dropped write (debug log, no crash); the REPL stays up.
            logger.debug("session.automation %s.%s failed: %s", self.kind, call, exc)
            return None
        self._adopt(reply.get("state"))
        error = reply.get("error")
        if error:
            raise _ERRORS.get(error.get("type"), RuntimeError)(error.get("message") or "")
        value = _decode(reply.get("value"))
        return self._state if isinstance(value, str) and value == reply.get("state") and value else value

    def refresh(self):
        self._rpc("state")


class GatewayGoalManager(_Remote):
    """``hermes_cli.goals.GoalManager`` over the gateway (same method names and returns)."""

    kind = "goal"

    def __init__(self, session_id, connection, *, default_max_turns=None, load=True):
        from hermes_cli.goals import GoalState
        self._state_cls = GoalState
        self.default_max_turns = default_max_turns
        super().__init__(session_id, connection, load=load)

    # Main's pure reads, on the mirrored state (``GoalManager`` methods bound to this object).
    from hermes_cli.goals import GoalManager as _Local
    is_active = _Local.is_active
    has_goal = _Local.has_goal
    has_contract = _Local.has_contract
    status_line = _Local.status_line
    render_subgoals = _Local.render_subgoals
    render_gates = _Local.render_gates
    render_contract = _Local.render_contract
    next_continuation_prompt = _Local.next_continuation_prompt
    del _Local

    def set(self, goal, *, max_turns=None, contract=None):
        kwargs = {"max_turns": max_turns} if max_turns is not None else {}
        if contract is not None:
            kwargs["contract"] = contract.to_dict()
        return self._rpc("set", goal, **kwargs)

    def pause(self, reason="user-paused"):
        return self._rpc("pause", reason=reason)

    def resume(self, *, reset_budget=True):
        return self._rpc("resume", reset_budget=reset_budget)

    def clear(self):
        self._rpc("clear")

    def mark_done(self, reason):
        self._rpc("mark_done", reason)

    def add_subgoal(self, text):
        return self._rpc("add_subgoal", text)

    def remove_subgoal(self, index_1based):
        return self._rpc("remove_subgoal", int(index_1based))

    def clear_subgoals(self):
        return self._rpc("clear_subgoals")

    def add_gate(self, command, *, timeout_seconds=None, max_retries=None):
        kwargs = {k: v for k, v in (("timeout_seconds", timeout_seconds), ("max_retries", max_retries)) if v}
        return self._rpc("add_gate", command, **kwargs)

    def remove_gate(self, index_1based):
        return self._rpc("remove_gate", int(index_1based))

    def clear_gates(self):
        return self._rpc("clear_gates")

    def wait_on(self, pid, reason=""):
        return self._rpc("wait_on", int(pid), reason=reason)

    def wait_on_session(self, session_id, reason=""):
        return self._rpc("wait_on_session", session_id, reason=reason)

    def wait_for_seconds(self, seconds, reason="", *, on_delegations=0):
        return self._rpc("wait_for_seconds", int(seconds), reason=reason, on_delegations=on_delegations)

    def stop_waiting(self):
        return bool(self._rpc("stop_waiting"))

    def is_waiting(self):
        """Barrier check (and lazy clear) where the barrier lives: the pid / process session is the
        gateway's. A state with no barrier answers locally, so the idle poll costs no round trip."""
        s = self._state
        if s is None or (s.waiting_on_pid is None and s.waiting_on_session is None and not s.waiting_until):
            return False
        return bool(self._rpc("is_waiting"))

    def command(self, arg, last_user_message=None):
        """The whole ``/goal <arg>`` run by the gateway (``/goal draft``: its contract drafter is a model call)."""
        from hermes_cli.goal_command import GoalCommandResult
        result = self._rpc("command", arg, last_user_message)
        if not isinstance(result, dict):
            return GoalCommandResult("Goal unavailable: the gateway did not answer.", error=True)
        return GoalCommandResult(**{k: v for k, v in result.items() if k in GoalCommandResult.__dataclass_fields__})

    def evaluate_after_turn(self, last_response, *, user_initiated=True, background_processes=None,
                            active_delegations=0):
        """The judge runs in the gateway (``auxiliary.goal_judge``), with the gateway's background
        processes and delegations; the decision dict is main's."""
        decision = self._rpc("evaluate_after_turn", last_response or "")
        if not isinstance(decision, dict):
            return {"status": None, "should_continue": False, "continuation_prompt": None,
                    "verdict": "inactive", "reason": "gateway unavailable", "message": ""}
        return decision


class GatewayLoopManager(_Remote):
    """``hermes_cli.loops.LoopManager`` over the gateway."""

    kind = "loop"

    def __init__(self, session_id, connection, *, load=True):
        from hermes_cli.loops import LoopState
        self._state_cls = LoopState
        super().__init__(session_id, connection, load=load)

    from hermes_cli.loops import LoopManager as _Local
    is_active = _Local.is_active
    has_loop = _Local.has_loop
    status_line = _Local.status_line
    is_due = _Local.is_due
    del _Local

    def set(self, prompt, *, interval_seconds=None, times=0, until="", route=None):
        return self._rpc("set", prompt, interval_seconds=interval_seconds, times=times, until=until)

    def pause(self, reason="user-paused"):
        return self._rpc("pause", reason=reason)

    def resume(self):
        return self._rpc("resume")

    def clear(self):
        return bool(self._rpc("clear"))

    def fire_tick(self):
        return self._rpc("fire_tick")

    def abandon_tick(self):
        self._rpc("abandon_tick")

    def complete_tick(self, last_response):
        decision = self._rpc("complete_tick", last_response or "")
        return decision if isinstance(decision, dict) else {"status": None, "stopped": False,
                                                             "reason": "gateway unavailable", "message": ""}

    def goal_blocks_tick(self):
        return bool(self._rpc("goal_blocks_tick"))


class GatewayHeartbeatManager(_Remote):
    """``hermes_cli.heartbeat.HeartbeatManager`` over the gateway."""

    kind = "heartbeat"

    def __init__(self, session_id, connection, *, load=True):
        from hermes_cli.heartbeat import HeartbeatState
        self._state_cls = HeartbeatState
        super().__init__(session_id, connection, load=load)

    from hermes_cli.heartbeat import HeartbeatManager as _Local
    has_heartbeat = _Local.has_heartbeat
    is_active = _Local.is_active
    status_line = _Local.status_line
    del _Local

    def set(self, prompt, interval_seconds):
        return self._rpc("set", prompt, int(interval_seconds))

    def pause(self):
        return self._rpc("pause")

    def resume(self):
        return self._rpc("resume")

    def clear(self):
        return bool(self._rpc("clear"))

    def due_prompt(self, now=None):
        s = self._state
        if s is None or not s.is_due(now):
            return None  # main's cheap local check first: the idle poll costs no round trip
        return self._rpc("due_prompt")

    def abandon_fire(self):
        return bool(self._rpc("abandon_fire"))


def dispatch_goal_command(mgr, arg, *, authorize_gate, last_user_message=None, progress=None):
    """``goal_command.dispatch_goal_command`` for the classic CLI: on a gateway manager ``/goal draft``
    (its contract drafter is an auxiliary-model call) runs whole on the gateway; every other form is
    main's dispatcher over the proxy, so its output strings stay main's."""
    from hermes_cli import goal_command
    if isinstance(mgr, GatewayGoalManager) and (arg or "").split(None, 1)[:1] == ["draft"]:
        if progress is not None:
            progress("Drafting completion contract…")
        return mgr.command(arg, last_user_message)
    return goal_command.dispatch_goal_command(mgr, arg, authorize_gate=authorize_gate,
                                              last_user_message=last_user_message, progress=progress)


def gateway_connection(cli):
    """The CLI's gateway connection when its session store is the gateway's, else None (main's local
    managers: tests and the TUI slash worker's shadow CLI)."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    store = getattr(cli, "_session_db", None)
    if isinstance(store, GatewaySessionStore) and getattr(cli, "_gateway_session_open", False):
        return store.connection
    return None
