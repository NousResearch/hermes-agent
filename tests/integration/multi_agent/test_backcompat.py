"""H — backward compatibility: a gateway with NO multi-agent config must behave
exactly like a legacy single-agent install. The "we didn't break existing
deployments" guard for PR #62944.

A bare install (``agents: {}``, ``routes: []``) resolves the synthetic ``main``
profile with no per-agent home — the run executes at the ROOT ``HERMES_HOME`` and
reads the ROOT ``.env``, exactly as before the feature existed.
"""
import pytest

pytestmark = pytest.mark.asyncio


async def test_bare_install_runs_at_root_home_reading_root_env(integ):
    """No agents/routes: request runs as 'main' at the root home, resolving the
    root .env key — legacy single-agent behavior, unchanged."""
    env = integ({}, default_agent="main", multiplex=False)  # bare: no agents
    res = await env.post(None, "n1")  # no routing header

    assert res["agent_id"] == "main"
    # Home is the root HERMES_HOME, NOT a profiles/<id> subdir.
    assert "/profiles/" not in res["home"]
    # Reads the root .env (legacy os.environ-style path), no fail-close.
    assert res["resolved_key"] == "sk-ROOT-env"


async def test_bare_install_ignores_stray_routing_header(integ):
    """A stray X-Hermes-Chat-Id on a non-multi-agent gateway must not break the
    request — no route matches, so it stays the default 'main'."""
    env = integ({}, default_agent="main", multiplex=False)
    res = await env.post("some-agent", "n1")
    assert res["agent_id"] == "main"
    assert "/profiles/" not in res["home"]
