"""Security regressions for agent-scoped runtime tool allowlists."""

import logging
from types import SimpleNamespace

from agent.tool_executor import _runtime_tool_scope_block


def test_runtime_tool_allowlist_is_opt_in():
    agent = SimpleNamespace()

    assert _runtime_tool_scope_block(agent, "terminal") is None


def test_runtime_tool_allowlist_allows_only_named_tools():
    agent = SimpleNamespace(
        _runtime_tool_allowlist=frozenset({"skills_list", "skill_view"}),
        _runtime_tool_scope_name="curator dry-run",
    )

    assert _runtime_tool_scope_block(agent, "skill_view") is None
    denied = _runtime_tool_scope_block(agent, "terminal")

    assert denied is not None
    assert "terminal" in denied
    assert "curator dry-run" in denied
    assert "not allowed" in denied


def test_runtime_tool_allowlist_empty_set_denies_everything():
    agent = SimpleNamespace(
        _runtime_tool_allowlist=frozenset(),
        _runtime_tool_scope_name="locked agent",
    )

    assert _runtime_tool_scope_block(agent, "skill_view") is not None


def test_runtime_tool_allowlist_fails_closed_for_mutable_policy():
    agent = SimpleNamespace(
        _runtime_tool_allowlist=["skill_view"],
        _runtime_tool_scope_name="curator",
    )

    denied = _runtime_tool_scope_block(agent, "skill_view")

    assert denied is not None
    assert "curator runtime scope" in denied


def test_runtime_tool_scope_denial_emits_security_audit_log(caplog):
    agent = SimpleNamespace(
        _runtime_tool_allowlist=frozenset({"skill_view"}),
        _runtime_tool_scope_name="curator dry-run",
        session_id="curator-session",
    )

    with caplog.at_level(logging.WARNING, logger="agent.tool_executor"):
        denied = _runtime_tool_scope_block(agent, "terminal")

    assert denied is not None
    record = caplog.records[-1]
    assert record.levelno == logging.WARNING
    assert "tool='terminal'" in record.getMessage()
    assert "scope='curator dry-run'" in record.getMessage()
    assert "session_id='curator-session'" in record.getMessage()
