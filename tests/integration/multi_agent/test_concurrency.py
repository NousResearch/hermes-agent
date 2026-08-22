"""P0 — the concurrency isolation invariant.

The highest-risk untested seam: ContextVars (profile + secret scope) crossing
``run_in_executor`` threads. If binding leaks across concurrently-running agents,
a response comes back with the wrong agent's home/credential. This drives many
interleaved requests through the ONE adapter and asserts, for every response,
that it executed as its OWN agent with its OWN key — the invariant the whole
"multiple agents, one gateway" promise rests on.
"""
import asyncio

import pytest

pytestmark = pytest.mark.asyncio


async def test_interleaved_requests_never_cross_contaminate(integ):
    """K agents × M concurrent requests: every response echoes its own
    agent_id + home-.env key. Zero cross-talk."""
    agents = {f"agent{i}": f"sk-agent{i}-key" for i in range(4)}
    env = integ(agents, multiplex=True)

    # Fire many requests, interleaved across agents, concurrently through one adapter.
    plan = [(aid, f"{aid}-req{m}") for m in range(6) for aid in agents]
    results = await asyncio.gather(*(env.post(aid, nonce) for aid, nonce in plan))

    assert len(results) == len(plan)
    for (aid, nonce), res in zip(plan, results):
        # The response for THIS request must reflect THIS request's agent.
        assert res["agent_id"] == aid, f"{nonce}: ran as {res['agent_id']}, expected {aid}"
        assert res["resolved_key"] == f"sk-{aid}-key", (
            f"{nonce}: key {res['resolved_key']}, expected sk-{aid}-key")
        assert res["home"].endswith(f"/profiles/{aid}")
        assert res["nonce"] == nonce  # response correlates to its own request


async def test_concurrent_same_two_agents_stay_isolated(integ):
    """Tight interleave of exactly two agents, high concurrency — the classic
    A/B cross-contamination shape."""
    env = integ({"alpha": "sk-alpha", "beta": "sk-beta"}, multiplex=True)

    plan = []
    for i in range(20):
        plan.append(("alpha", f"a{i}"))
        plan.append(("beta", f"b{i}"))
    results = await asyncio.gather(*(env.post(a, n) for a, n in plan))

    for (aid, _), res in zip(plan, results):
        assert res["agent_id"] == aid
        assert res["resolved_key"] == f"sk-{aid}"
