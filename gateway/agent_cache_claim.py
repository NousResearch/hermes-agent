"""Cache-use ownership held from lookup/publication through the actual sync worker exit."""

from contextlib import nullcontext
from functools import wraps


def cache_agent_in_use(agent):
    return getattr(agent, "__dict__", {}).get("_gateway_cache_owner") is not None


def claim_cached_agent(turn, agent):
    """Caller holds the cache lock, so an eviction cannot observe an unclaimed checkout."""
    owner = getattr(agent, "__dict__", {}).get("_gateway_cache_owner")
    if owner is not None and owner is not turn:
        raise RuntimeError("cached agent is still executing another turn")
    agent._gateway_cache_owner = turn
    turn._claimed_cache_agent = agent


def unclaim_cached_agent(turn, agent):
    """Caller holds the cache lock; identity guards a displaced worker's late finally."""
    owner = getattr(agent, "__dict__", {}).get("_gateway_cache_owner")
    if owner is not None and owner is not turn:
        raise RuntimeError("cannot replace an agent still executing another turn")
    if owner is turn:
        del agent._gateway_cache_owner
        turn._claimed_cache_agent = None


def release_cached_agent(turn):
    agent = getattr(turn, "_claimed_cache_agent", None)
    if agent is None:
        return
    runner = turn._runner
    with runner._agent_cache_lock:
        unclaim_cached_agent(turn, agent)
    release_retired_agents(runner, turn._ctx.session_key)
    evict_deferred_agent(runner, turn._ctx.session_key, agent)


def evict_deferred_agent(runner, session_key, agent):
    """Consume a deferred request under the same lock as worker exit and cache eviction."""
    with getattr(runner, "_agent_cache_lock", None) or nullcontext():
        state = runner._peek_session_state(session_key)
        if state is not None and any(a is agent for a in state.turn.retired_agents):
            return  # detached owners keep their boundary until the retired drain can release them
        boundary = vars(agent).pop("_cache_evict_after_turn", None)
    if boundary is not None:
        runner._evict_cached_agent(session_key, commit_memory=boundary, expected_agent=agent)


def owns_cached_agent(method):
    @wraps(method)
    def run(turn, *args, **kwargs):
        try:
            return method(turn, *args, **kwargs)
        finally:
            release_cached_agent(turn)
    return run


def release_retired_agents(runner, session_key):
    """Drain upstream's detached one-turn owners only after their sync checkouts end."""
    with getattr(runner, "_agent_cache_lock", None) or nullcontext():
        state = runner._peek_session_state(session_key)
        if state is None:
            return
        ready = [a for a in state.turn.retired_agents if not cache_agent_in_use(a)]
        state.turn.retired_agents = [a for a in state.turn.retired_agents if cache_agent_in_use(a)]
        ready = [(a, vars(a).pop("_cache_evict_after_turn", False)) for a in ready]
    for agent, boundary in ready:
        target = runner._commit_then_release_soft if boundary else runner._release_evicted_agent_soft
        args = (agent, session_key, True) if boundary else (agent,)
        runner._spawn_release_thread(target, args, f"agent-retire-{str(session_key)[:24]}",
                                     inline_fallback=True, session_key=session_key)
