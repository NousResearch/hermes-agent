"""Gateway Goal front-end for Codex's persisted, evidence-driven native Goal."""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager

from hermes_cli.goals import DEFAULT_MAX_TURNS, GoalManager, load_goal, save_goal

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def goal_lock(session_id):
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(session_id, threading.RLock())
    with lock:
        yield


def native_objective(state):
    parts = [state.goal]
    if state.has_contract():
        parts.append(state.contract.render_block())
    if state.subgoals:
        parts.append(state.render_subgoals_block())
    text = "\n\n".join(parts)
    if len(text) > 4000:
        raise ValueError("Codex Goal including contract/subgoals exceeds 4000 characters")
    return text


def adopt_native_goal(session_id, native, *, thread_id, previous):
    """Mirror an already authorized native Goal; never set/reset its budget or steal a user control."""
    from hermes_cli.goals import GoalState
    if native.get("threadId") != thread_id or not native.get("objective"):
        raise RuntimeError("Cannot adopt a Goal from a different thread")
    with goal_lock(session_id):
        current = load_goal(session_id)
        # A pause/clear/replacement during the ordinary turn takes precedence.
        if (current.to_json() if current else None) != (previous.to_json() if previous else None):
            raise RuntimeError("Goal changed while discovering a model-created native Goal")
        if current is not None:
            old_native = current.native_goal or {}
            if current.runtime != "codex":
                raise RuntimeError("A Hermes-owned Goal cannot be replaced by native discovery")
            same_goal = (old_native.get("threadId") == thread_id
                         and old_native.get("objective") == native.get("objective"))
            if same_goal or current.status in {"active", "cleared"}:
                raise RuntimeError("Native discovery cannot undo a paused/cleared or active Goal")
        state = GoalState(goal=native["objective"], runtime="codex", goal_id=str(uuid.uuid4()),
                          token_budget=native.get("tokenBudget"), native_goal=native)
        save_goal(session_id, state)
        persisted = load_goal(session_id)
        if persisted is None or persisted.goal_id != state.goal_id:
            raise RuntimeError("Discovered native Goal was not durably mirrored")
        return persisted


def native_goal_resource_limited(native):
    """A budgetLimited label can remain sticky after an authorized ceiling correction."""
    budget = native.get("tokenBudget")
    exhausted = budget is not None and native.get("tokensUsed", 0) >= budget
    return (native.get("status") == "usageLimited" or exhausted
            or (native.get("status") == "budgetLimited" and budget is None))


class CodexGoalManager(GoalManager):
    """Mirror lifecycle for UI/controls only; never run Hermes' aux judge or FIFO loop."""
    runtime_name = "codex"

    def __init__(self, session_id, *, token_budget=None, default_max_turns=DEFAULT_MAX_TURNS):
        super().__init__(session_id, default_max_turns=default_max_turns)
        self.token_budget = token_budget

    def set(self, goal, *, max_turns=None, contract=None):
        if len(goal) > 4000:
            raise ValueError("Codex Goal objective must be at most 4000 characters")
        with goal_lock(self.session_id):
            previous = load_goal(self.session_id)
            state = super().set(goal, contract=contract)
            state.runtime = "codex"
            state.goal_id = str(uuid.uuid4())
            # Editing an unfinished Goal is not a fresh spending authorization.
            # Preserve its thread and cumulative ledger despite a new profile default.
            if (previous is not None and previous.runtime == "codex" and previous.native_goal
                    and previous.status not in {"done", "cleared"}):
                state.token_budget = previous.token_budget
                state.native_goal = dict(previous.native_goal)
                state.turns_used = previous.turns_used
            else:
                state.token_budget = self.token_budget
            return self._save()

    def resume(self, *, reset_budget=True):
        # Native usage must survive a resume; the model cannot award itself more budget.
        with goal_lock(self.session_id):
            self._state = load_goal(self.session_id)
            if self._state is None or self._state.status in {"done", "cleared"}:
                return None
            native = self._state.native_goal or {}
            if native.get("status") == "complete" or native_goal_resource_limited(native):
                raise ValueError("Native Goal is terminal or resource-limited; set a new authorized goal/budget instead")
            return super().resume(reset_budget=False)

    def _mutate(self, method, *args, **kwargs):
        with goal_lock(self.session_id):
            self._state = load_goal(self.session_id)
            return getattr(super(), method)(*args, **kwargs)

    def pause(self, reason="user-paused"):
        return self._mutate("pause", reason=reason)

    def clear(self):
        return self._mutate("clear")

    def add_subgoal(self, text):
        return self._mutate("add_subgoal", text)

    def remove_subgoal(self, index_1based):
        return self._mutate("remove_subgoal", index_1based)

    def clear_subgoals(self):
        return self._mutate("clear_subgoals")

    def set_contract(self, contract):
        return self._mutate("set_contract", contract)

    def add_gate(self, command, **kwargs):
        return self._mutate("add_gate", command, **kwargs)

    def remove_gate(self, index_1based):
        return self._mutate("remove_gate", index_1based)

    def clear_gates(self):
        return self._mutate("clear_gates")

    def wait_on(self, *args, **kwargs):
        raise ValueError("Hermes wait barriers are not supported for native Codex Goals")

    def status_line(self):
        state = self.state
        if state is None or state.status == "cleared":
            return "No active goal. Set one with /goal <text>."
        view = state.native_goal or {}
        budget = str(state.token_budget) if state.token_budget is not None else "unlimited"
        from agent.codex_runtime_goals import public_codex_failure
        suffix = f"; {public_codex_failure(state.paused_reason)}" if state.paused_reason else ""
        return (f"Codex Goal ({state.status}, {state.turns_used} turns, "
                f"tokens {view.get('tokensUsed', 0)}/{budget}{suffix}): {state.goal}")


def goal_manager_for_session(session_id, *, default_max_turns=DEFAULT_MAX_TURNS):
    """Keep ownership of existing goals; select native only for an opted-in Codex route."""
    from hermes_cli.config import load_config
    from hermes_cli.goals import _get_session_db
    state = load_goal(session_id)
    cfg = load_config() or {}
    opts = cfg.get("goals") or {}
    if opts.get("runtime", "hermes") not in {"hermes", "codex"}:
        raise ValueError("goals.runtime must be hermes or codex")
    model = cfg.get("model") or {}
    mode = model.get("openai_runtime") if isinstance(model, dict) else None
    db = _get_session_db()
    row = db.get_session(session_id) if db is not None else None
    if row and row.get("model_config"):
        import json
        settings = row["model_config"]
        settings = json.loads(settings) if isinstance(settings, str) else settings
        mode = (settings.get("gateway_runtime") or {}).get("api_mode", mode)
    native_owner = state is not None and state.runtime == "codex"
    old_active = state is not None and state.runtime == "hermes" and state.status in {"active", "paused"}
    if native_owner or (not old_active and opts.get("runtime") == "codex" and mode == "codex_app_server"):
        budget = opts.get("codex_token_budget", 200000)
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
            raise ValueError("goals.codex_token_budget must be a non-negative integer or null")
        return CodexGoalManager(session_id, token_budget=budget or None, default_max_turns=default_max_turns)
    return GoalManager(session_id, default_max_turns=default_max_turns)


def record_native_goal(session_id, goal_id, native, *, completed_turn=False):
    """Generation-fenced mirror: late events cannot undo user pause/clear or replace a new goal."""
    with goal_lock(session_id):
        state = load_goal(session_id)
        if state is None or state.runtime != "codex" or state.goal_id != goal_id:
            return None
        if state.status == "active" and native.get("objective") != native_objective(state):
            state.status, state.paused_reason = "paused", "Goal criteria changed; set a new goal to authorize the revised objective"
        state.native_goal = native
        if completed_turn:
            state.turns_used += 1
            state.last_turn_at = time.time()
        if state.status == "active":
            status = native["status"]
            if status == "complete":
                mgr = GoalManager(session_id)
                mgr._state = state
                gate = mgr._check_gates()
                if gate is not None:
                    state.status, state.paused_reason = "paused", "Completion gate did not pass"
                else:
                    state.status, state.last_verdict = "done", "done"
                    state.last_reason = "Codex native Goal completed"
            elif status != "active":
                state.status, state.paused_reason = "paused", f"Codex Goal {status}"
        save_goal(session_id, state)
        return state


def pause_native_goal(session_id, goal_id, reason):
    with goal_lock(session_id):
        state = load_goal(session_id)
        if state is not None and state.goal_id == goal_id and state.status == "active":
            state.status, state.paused_reason = "paused", reason
            save_goal(session_id, state)
