"""Dispatcher-spawned kanban workers keep their lifecycle tools even when the
assignee profile's ``agent.disabled_toolsets`` lists ``kanban``.

A profile may disable the kanban toolset to hide board tools from its chat
sessions. Without the worker carve-out, the disabled-subtraction step ran
after the worker's kanban append and stripped ``kanban_complete`` /
``kanban_block`` from every worker, which could then never close its task.

Chat sessions are unaffected: without ``HERMES_KANBAN_TASK`` the disable
still strips the kanban tools.
"""

import pytest

import tools.kanban_tools  # noqa: F401 - ensure registered
from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tools.registry import invalidate_check_fn_cache


@pytest.fixture(autouse=True)
def _clear_caches():
    """Both caches key on process-global state (env flags, registry generation);
    a stale entry from a test that ran without ``HERMES_KANBAN_TASK`` would
    otherwise mask the worker-append behavior under test."""
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _names(tools):
    return {t["function"]["name"] for t in tools}


PINNED = ["clarify", "file", "memory", "todo", "web"]


def test_worker_keeps_lifecycle_tools_despite_profile_disable(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test123")
    names = _names(get_tool_definitions(
        enabled_toolsets=list(PINNED), disabled_toolsets=["kanban", "x_search"], quiet_mode=True,
    ))
    # The full worker lifecycle surface (complete, block, show, heartbeat)
    # must survive a profile-level kanban disable.
    assert "kanban_complete" in names
    assert "kanban_block" in names
    assert "kanban_show" in names
    assert "kanban_heartbeat" in names


def test_non_worker_still_loses_kanban_tools_when_disabled(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    names = _names(get_tool_definitions(
        enabled_toolsets=list(PINNED) + ["kanban"], disabled_toolsets=["kanban"], quiet_mode=True,
    ))
    assert "kanban_complete" not in names
    assert "kanban_block" not in names
    assert "kanban_show" not in names


def test_worker_disable_of_other_toolsets_still_applies(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test123")
    names = _names(get_tool_definitions(
        enabled_toolsets=list(PINNED), disabled_toolsets=["kanban", "web"], quiet_mode=True,
    ))
    assert "kanban_complete" in names
    assert "web_search" not in names
