"""Regression tests for BLK-003: dispatcher-spawned Kanban workers whose
profile disables the ``kanban`` toolset in their own ``agent.disabled_toolsets``
must still receive the task-lifecycle tools (kanban_complete, kanban_block,
kanban_heartbeat, ...).

Root cause: ``_compute_tool_definitions`` re-injects ``"kanban"`` into the
*enabled* toolset list when a dispatcher-owned worker is detected, but the
disabled-toolset subtraction pass ran unconditionally against the caller's
original ``disabled_toolsets`` afterwards — silently undoing the injection
for any worker profile whose own config disables ``kanban`` (the common,
recommended setup for worker profiles that should not have the full
orchestrator toolset in normal chat).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import model_tools


@pytest.fixture(autouse=True)
def _clear_cache():
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()


def _worker_env(monkeypatch, task_id="t_test123"):
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)


class TestKanbanWorkerToolsetInjectionSurvivesOwnDisabledToolsets:
    def test_worker_with_kanban_in_disabled_toolsets_still_gets_kanban_complete(
        self, monkeypatch
    ):
        """A dispatcher-owned worker whose profile disables `kanban` in its
        own agent.disabled_toolsets (the analyst/scout pattern) must still
        see kanban_complete in its resolved tool definitions."""
        _worker_env(monkeypatch)
        with (
            patch("model_tools._is_delegated_child_context", return_value=False),
            patch("model_tools._is_dispatcher_owned_worker", return_value=True),
        ):
            tools = model_tools.get_tool_definitions(
                enabled_toolsets=["file", "memory"],
                disabled_toolsets=["kanban"],
                quiet_mode=True,
            )
        names = {t["function"]["name"] for t in tools}
        assert "kanban_complete" in names
        assert "kanban_block" in names
        assert "kanban_heartbeat" in names

    def test_non_worker_call_still_honors_disabled_toolsets(self, monkeypatch):
        """Outside a dispatcher-owned worker context, disabling `kanban`
        must still strip kanban_complete as before — the fix must not
        weaken the normal disabled_toolsets contract for non-worker calls."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        tools = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "memory"],
            disabled_toolsets=["kanban"],
            quiet_mode=True,
        )
        names = {t["function"]["name"] for t in tools}
        assert "kanban_complete" not in names

    def test_delegated_child_context_does_not_get_forced_kanban(self, monkeypatch):
        """A delegate_task child running inside a kanban worker's process
        must not inherit the worker's kanban lifecycle tools even if
        HERMES_KANBAN_TASK leaks into its env (#67567-style isolation)."""
        _worker_env(monkeypatch)
        with (
            patch("model_tools._is_delegated_child_context", return_value=True),
            patch("model_tools._is_dispatcher_owned_worker", return_value=True),
        ):
            tools = model_tools.get_tool_definitions(
                enabled_toolsets=["file", "memory"],
                disabled_toolsets=["kanban"],
                quiet_mode=True,
            )
        names = {t["function"]["name"] for t in tools}
        assert "kanban_complete" not in names

    def test_explicit_kanban_disabled_toolsets_list_not_mutated(self, monkeypatch):
        """The caller's disabled_toolsets list must not be mutated in place
        — only a local working copy should have 'kanban' removed."""
        _worker_env(monkeypatch)
        original = ["kanban", "web"]
        with (
            patch("model_tools._is_delegated_child_context", return_value=False),
            patch("model_tools._is_dispatcher_owned_worker", return_value=True),
        ):
            model_tools.get_tool_definitions(
                enabled_toolsets=["file", "memory"],
                disabled_toolsets=original,
                quiet_mode=True,
            )
        assert original == ["kanban", "web"]
