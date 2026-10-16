"""Cache-use ownership held from lookup/publication through the actual sync worker exit."""

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
        evict = "_cache_evict_after_turn" in vars(agent)
        entry = runner._agent_cache.get(turn._ctx.session_key)
        still_cached = isinstance(entry, tuple) and entry and entry[0] is agent
    if evict and still_cached:
        runner._evict_cached_agent(turn._ctx.session_key, commit_memory=agent._cache_evict_after_turn)


def owns_cached_agent(method):
    @wraps(method)
    def run(turn, *args, **kwargs):
        try:
            return method(turn, *args, **kwargs)
        finally:
            release_cached_agent(turn)
    return run
