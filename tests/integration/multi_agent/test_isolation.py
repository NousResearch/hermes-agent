"""P0/P1 integration: per-agent isolation through a real api_server request.

These drive a real HTTP request through ``APIServerAdapter`` and assert what the
run actually executed as — proving routing → profile+secret scope → per-agent
home/credential end-to-end, not via mocks.
"""
import pytest

pytestmark = pytest.mark.asyncio


async def test_two_agents_resolve_their_own_credential(integ):
    """C — credential isolation: each routed agent's run resolves its OWN
    home-.env key, never the other's and never the process-global leak."""
    env = integ({"coder": "sk-coder-key", "research": "sk-research-key"}, multiplex=True)

    coder = await env.post("coder", "n1")
    research = await env.post("research", "n2")

    assert coder["agent_id"] == "coder"
    assert coder["resolved_key"] == "sk-coder-key"
    assert research["agent_id"] == "research"
    assert research["resolved_key"] == "sk-research-key"
    # neither leaked the process-global value
    assert "sk-ROOT-env" not in (coder["resolved_key"], research["resolved_key"])


async def test_two_agents_resolve_their_own_home_and_soul(integ):
    """B — profile isolation: home getters + SOUL resolve to the routed agent's
    own directory, not the other's."""
    env = integ({"coder": "sk-c", "research": "sk-r"}, multiplex=True)

    coder = await env.post("coder", "n1")
    research = await env.post("research", "n2")

    assert coder["home"].endswith("/profiles/coder")
    assert coder["soul_first_line"] == "I am CODER. Scope: coder."
    assert research["home"].endswith("/profiles/research")
    assert research["soul_first_line"] == "I am RESEARCH. Scope: research."
    assert coder["home"] != research["home"]


async def test_header_routes_to_agent_and_default_falls_through(integ):
    """A — routing: header selects the agent; absent header → default_agent."""
    env = integ({"main": "sk-main", "coder": "sk-coder"}, default_agent="main")

    routed = await env.post("coder", "n1")
    defaulted = await env.post(None, "n2")  # no X-Hermes-Chat-Id

    assert routed["agent_id"] == "coder"
    assert defaulted["agent_id"] == "main"


async def test_unknown_header_falls_through_to_default(integ):
    """A — an unmatched routing header is not an error; falls to default_agent."""
    env = integ({"main": "sk-main", "coder": "sk-coder"}, default_agent="main")
    res = await env.post("ghost-agent", "n1")
    assert res["agent_id"] == "main"


async def test_agent_without_key_does_not_leak_process_global(integ):
    """C2 — SECURITY: under multiplex, an agent whose scoped home .env has no
    usable provider key must NOT fall back to the process-global value. The
    scoped read yields a falsy value (empty/None), never sk-GLOBAL-leak — the
    fail-closed-vs-leak property the secret scope exists to guarantee."""
    env = integ({"nokey": ""}, default_agent="nokey", multiplex=True)
    res = await env.post("nokey", "n1")

    assert res["agent_id"] == "nokey"
    # The one thing that must never happen: leaking the process-global key.
    assert res["resolved_key"] != "sk-ROOT-env"
    assert not res["resolved_key"]  # falsy: scoped-but-absent, not leaked
