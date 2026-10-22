"""MCP teardown at shutdown: the caller's budget is divided, and the wildcard pass is not poisoned.

Each ``shutdown_mcp_servers`` call waits up to its own timeout on the MCP loop, so N per-profile
passes at the 15s default consumed the whole 5s caller budget and the trailing WILDCARD pass — the
only one that stops the shared loop — never ran. And the teardown worker must start from a FRESH
context: a caller sitting inside a served profile's scope would otherwise hand its HERMES_HOME
override to the launch-profile pass, which documents that it has none.
"""

import pytest

from gateway import run as gateway_run


class _Cfg:
    multiplex_profiles = True


@pytest.fixture
def homes(tmp_path, monkeypatch):
    homes = []
    for name in ("a", "b", "c"):
        home = tmp_path / name
        home.mkdir()
        homes.append((name, home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    (tmp_path / "launch").mkdir()
    monkeypatch.setattr(gateway_run, "_multiplex_profile_homes", lambda _cfg: homes)
    return homes


@pytest.mark.asyncio
async def test_every_pass_shares_the_callers_budget(homes, monkeypatch):
    import tools.mcp_tool_lifecycle as lifecycle

    seen: list[tuple] = []

    def _fake(*, scope=None, names=None, timeout=15.0):
        seen.append((scope, timeout))

    monkeypatch.setattr(lifecycle, "shutdown_mcp_servers", _fake)

    assert await gateway_run._shutdown_mcp_servers_nonblocking(timeout=8.0, config=_Cfg())

    assert len(seen) == 4, f"the wildcard pass did not run: {seen}"
    assert seen[-1][0] is None, "the last pass must be the process-wide wildcard"
    budget_per_pass = 8.0 / 4
    assert all(t <= budget_per_pass + 0.01 for _scope, t in seen), (
        f"a pass could outlive the caller's whole budget: {seen}")


@pytest.mark.asyncio
async def test_teardown_thread_does_not_inherit_the_callers_home_override(tmp_path, monkeypatch):
    import tools.mcp_tool_lifecycle as lifecycle
    from hermes_constants import (
        get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override)

    launch = tmp_path / "launch"
    poison = tmp_path / "profiles" / "poison"
    for path in (launch, poison):
        path.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    seen: list = []
    monkeypatch.setattr(
        lifecycle, "shutdown_mcp_servers",
        lambda **_kw: seen.append(get_hermes_home_override()))

    token = set_hermes_home_override(str(poison))
    try:
        await gateway_run._shutdown_mcp_servers_nonblocking(timeout=5.0, config=None)
    finally:
        reset_hermes_home_override(token)

    assert seen == [None], f"the wildcard teardown ran inside {seen} instead of a fresh context"
