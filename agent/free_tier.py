"""Common turn-boundary continuation gate. Never changes model-facing messages.

Admission is per logical turn, not per API request: an admitted turn and its
commissioned children finish even when a completed tool crosses the allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from hermes_cli import free_tier_usage as usage


@dataclass
class TurnAdmission:
    identity: str | None
    guide: bool = False


_ACTIVE_TURN: ContextVar[TurnAdmission | None] = ContextVar("free_tier_turn", default=None)
_NESTED_DISPATCH: ContextVar[bool] = ContextVar("free_tier_nested_dispatch", default=False)


@contextmanager
def nested_dispatch():
    token = _NESTED_DISPATCH.set(True)
    try:
        yield
    finally:
        _NESTED_DISPATCH.reset(token)


def is_batch_envelope(name, args) -> bool:
    if name != "tool_call":
        return False
    from tools.tool_search import resolve_underlying_call, CONNECTOR_BATCH_SENTINEL
    underlying, _, error = resolve_underlying_call(args)
    return not error and underlying == CONNECTOR_BATCH_SENTINEL


def record_dispatch_completion(name, args, tool_call_id) -> None:
    if not tool_call_id or _NESTED_DISPATCH.get():
        record_executed_tool(name, args)


def record_executed_tool(name, args) -> None:
    if not is_batch_envelope(name, args):
        record_tool_completion()


def _route_identity(agent) -> str | None:
    from hermes_cli.anon_auth import route_is_welcome_host
    # Only the selected welcome endpoint spends this allowance, never a model-name match.
    if not route_is_welcome_host(getattr(agent, "base_url", None)):
        return None
    return usage.current_identity()


def refusal(agent, history=None) -> dict | None:
    from hermes_cli.onboarding_profile import is_onboarding_profile
    if is_onboarding_profile():
        return None
    identity = _route_identity(agent)
    status = usage.identity_status(identity)
    if not status["capped"]:
        return None
    return {
        "final_response": usage.LIMIT_NOTICE, "messages": list(history or []),
        "error": usage.LIMIT_REASON, "refusal_reason": usage.LIMIT_REASON,
        "completed": False, "failed": True, "partial": False, "interrupted": False,
        "retryable": True, "failure_retryable": False, "failure_reason": usage.LIMIT_REASON,
        "api_calls": 0, "free_tier": status, "continuation_required": True,
    }


def _notice(agent) -> None:
    from agent.credits_tracker import AgentNotice
    emit = getattr(agent, "_emit_notice", None)
    if callable(emit):
        emit(AgentNotice(usage.LIMIT_NOTICE, level="warn", key="free_tier.limit", id="free_tier.limit"))


def inherit_turn(parent, child) -> None:
    """Called by the actual child constructor, never inferred from client-supplied names."""
    child._free_tier_parent_turn = getattr(parent, "_free_tier_turn", None)


def rehome_after_sign_in(agent) -> bool:
    """Adopt a settled account before allowing an old welcome agent another inference."""
    from hermes_cli.anon_auth import current_nous_state, is_guest_state, route_is_welcome_host
    if not route_is_welcome_host(getattr(agent, "base_url", None)):
        return False
    state = current_nous_state()
    if is_guest_state(state):
        from hermes_cli.auth_nous import _read_shared_nous_state
        shared = _read_shared_nous_state()
        if not shared or is_guest_state(shared):
            return False
        # Read/adopt under canonical locks, without entering the provisioning/minter path.
        from hermes_cli.auth import _provider_state_transaction, _save_provider_state_to_source
        from hermes_cli.auth_nous import _nous_shared_store_lock
        with _provider_state_transaction("nous") as (store, local, source):
            with _nous_shared_store_lock():
                shared = _read_shared_nous_state()
                if not shared or is_guest_state(shared):
                    return False
                state = dict(shared) if is_guest_state(local) else local
                if state is not local:
                    _save_provider_state_to_source(store, "nous", state, source)
    if not state or is_guest_state(state):
        return False
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly().get("model") or {}
    model = (cfg.get("default") if isinstance(cfg, dict) else cfg) or getattr(agent, "model", "")
    base_url = state.get("inference_base_url")
    api_key = state.get("agent_key")
    if not api_key:
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        runtime = resolve_nous_runtime_credentials()
        base_url, api_key = runtime["base_url"], runtime["api_key"]
    if route_is_welcome_host(base_url):
        raise ValueError("Sign-in routing has not settled. Use /model to choose your account's model.")
    agent.switch_model(new_model=model, new_provider="nous", base_url=base_url,
                       api_key=api_key, api_mode="chat_completions")
    return True


@contextmanager
def admit_turn(agent, history=None):
    from hermes_cli.onboarding_profile import is_onboarding_profile
    restore = getattr(agent, "_restore_primary_runtime", None)
    if callable(restore):
        restore()
    rehome_after_sign_in(agent)
    inherited = getattr(agent, "_free_tier_parent_turn", None)
    if not isinstance(inherited, TurnAdmission):
        parent_ref = getattr(agent, "_delegate_parent_ref", None)
        parent = parent_ref() if callable(parent_ref) else None
        inherited = getattr(parent, "_free_tier_turn", None)
    if not isinstance(inherited, TurnAdmission) or not (inherited.identity or inherited.guide):
        blocked = refusal(agent, history)
        if blocked is not None:
            _notice(agent)
            yield blocked
            return
        inherited = None
    turn = inherited or TurnAdmission(_route_identity(agent), guide=is_onboarding_profile())
    agent._free_tier_is_child_turn = inherited is not None
    previous = getattr(agent, "_free_tier_turn", None)
    agent._free_tier_turn = turn
    token = _ACTIVE_TURN.set(turn)
    try:
        yield None
    finally:
        _ACTIVE_TURN.reset(token)
        agent._free_tier_turn = previous
        agent._free_tier_parent_turn = None  # admission is for this child turn, not future sends


def record_tool_completion(agent=None) -> None:
    """One call at the terminal result seam, plus nested RPC calls outside that seam."""
    turn = getattr(agent, "_free_tier_turn", None) if agent is not None else _ACTIVE_TURN.get()
    if not isinstance(turn, TurnAdmission) or turn.guide or not turn.identity:
        return
    usage.record_completed_tool(turn.identity)


def finish_turn(agent, result: dict) -> dict:
    turn = getattr(agent, "_free_tier_turn", None)
    if not isinstance(turn, TurnAdmission) or not turn.identity or turn.guide or getattr(agent, "_free_tier_is_child_turn", False):
        return result
    state = usage.identity_status(turn.identity)
    result["free_tier"] = state
    if state["capped"]:
        result["continuation_required"] = True
        result["free_tier_notice"] = usage.LIMIT_NOTICE
        result["response_transformed"] = True
        # Display-only footer: persisted/cached assistant content stays byte-stable.
        result["final_response"] = (str(result.get("final_response") or "").rstrip() + "\n\n" + usage.LIMIT_NOTICE).lstrip()
        _notice(agent)
    return result
