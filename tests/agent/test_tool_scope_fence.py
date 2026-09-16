"""Tests for the runtime toolset-scope fence in agent/tool_executor.py."""

from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import (
    _SCOPE_FENCE_ESCAPE_HATCHES,
    _allowed_tool_names_for_agent,
    _tool_scope_decision,
)


def _make_agent(enabled_toolsets=None, disabled_toolsets=None):
    agent = MagicMock()
    agent.enabled_toolsets = enabled_toolsets
    agent.disabled_toolsets = disabled_toolsets
    # Fresh sentinels each time so the cache doesn't leak between tests.
    del agent._allowed_tool_names
    del agent._allowed_tool_names_key
    return agent


@pytest.fixture(autouse=True)
def _patch_tool_search_scoped_names():
    """``_tool_search_scoped_names`` does a real tool-def build; keep it empty
    for these tests so it doesn't widen the allowed set unexpectedly."""
    with patch("agent.tool_executor._tool_search_scoped_names", return_value=frozenset()):
        yield


def test_tool_in_enabled_toolset_runs():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        assert _tool_scope_decision(agent, "terminal") is None


def test_tool_not_in_toolset_blocked_under_enforce():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        assert _tool_scope_decision(agent, "web_search") == "enforce"


def test_tool_not_in_toolset_shadow_only():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="shadow"):
        assert _tool_scope_decision(agent, "web_search") == "shadow"


def test_tool_not_in_toolset_off_mode_no_fence():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="off"):
        assert _tool_scope_decision(agent, "web_search") is None


@pytest.mark.parametrize("name", sorted(_SCOPE_FENCE_ESCAPE_HATCHES))
def test_escape_hatches_always_allowed(name):
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        assert _tool_scope_decision(agent, name) is None


def test_tool_call_bridge_always_allowed():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        from tools import tool_search as _ts
        assert _tool_scope_decision(agent, _ts.TOOL_CALL_NAME) is None


def test_unrestricted_session_no_enforcement():
    agent = _make_agent(enabled_toolsets=None)
    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        assert _allowed_tool_names_for_agent(agent) is None
        assert _tool_scope_decision(agent, "web_search") is None


def test_kanban_env_adds_kanban_tools_to_allowed_set(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    allowed = _allowed_tool_names_for_agent(agent)
    assert allowed is not None

    from toolsets import resolve_toolset

    assert set(resolve_toolset("kanban")) <= set(allowed)


def test_allowed_tool_names_cached_on_agent():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    first = _allowed_tool_names_for_agent(agent)
    # Repeat call with unchanged toolsets returns the cached value.
    second = _allowed_tool_names_for_agent(agent)
    assert first == second


def test_allowed_tool_names_cache_recomputes_on_toolset_change():
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    first = _allowed_tool_names_for_agent(agent)
    # Mutate the underlying toolsets; the keyed cache must recompute.
    agent.enabled_toolsets = ["web_tools"]
    second = _allowed_tool_names_for_agent(agent)
    assert first != second


def test_allowed_tool_names_cache_recomputes_on_kanban_env_change(monkeypatch):
    agent = _make_agent(enabled_toolsets=["terminal_tools"], disabled_toolsets=["kanban"])
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    first = _allowed_tool_names_for_agent(agent)

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-456")
    second = _allowed_tool_names_for_agent(agent)

    assert first != second
    from toolsets import resolve_toolset
    assert set(resolve_toolset("kanban")) <= set(second)


def test_kanban_lifecycle_tools_survive_kanban_in_disabled_toolsets(monkeypatch):
    """FIX 1: a kanban worker whose profile disables the kanban toolset for
    normal chat must still retain its lifecycle handoff tools."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-789")
    agent = _make_agent(enabled_toolsets=["terminal_tools"], disabled_toolsets=["kanban"])

    allowed = _allowed_tool_names_for_agent(agent)
    assert allowed is not None
    assert "kanban_complete" in allowed
    assert "kanban_block" in allowed
    assert "kanban_heartbeat" in allowed


def test_tool_scope_decision_fallback_mode_is_shadow_on_exception():
    """FIX 3: a transient config-read error must fail to 'shadow', not 'off'."""
    agent = _make_agent(enabled_toolsets=["terminal_tools"])
    with patch("agent.skill_utils.get_tool_enforcement_mode", side_effect=RuntimeError("boom")):
        assert _tool_scope_decision(agent, "web_search") == "shadow"


def test_tool_grant_unblocks_fenced_tool(tmp_path, monkeypatch):
    """Phase 4: a tool blocked by the fence under enforce becomes allowed once
    a grant is recorded for the current profile+task via the tool broker."""
    from hermes_cli import profile_activity_ledger as pal
    import tools.tool_grants as tg

    db = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(pal, "ledger_db_path", lambda: db)
    monkeypatch.setattr(pal, "_governance_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "octacon")

    agent = _make_agent(enabled_toolsets=["terminal_tools"])

    with patch("agent.skill_utils.get_tool_enforcement_mode", return_value="enforce"):
        # Blocked before any grant.
        assert _tool_scope_decision(agent, "web_search") == "enforce"

        decision = tg.grant_tool("octacon", "web_search", "t_1", "need it")
        assert decision["granted"] is True

        # Allowed after the grant.
        assert _tool_scope_decision(agent, "web_search") is None
