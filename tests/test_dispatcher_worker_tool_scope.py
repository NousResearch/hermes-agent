"""Dispatcher workers must not surface nested code execution by default."""

from __future__ import annotations

import model_tools


def _names(monkeypatch, enabled=None, disabled=None, *, task=True, dispatcher=True):
    if task:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(model_tools, "_is_delegated_child_context", lambda: False)
    monkeypatch.setattr(model_tools, "_is_dispatcher_owned_worker", lambda: dispatcher)
    return model_tools._select_tool_names(enabled, disabled, quiet_mode=True)


def test_dispatcher_worker_default_drops_execute_code(monkeypatch):
    assert "execute_code" not in _names(monkeypatch, ["hermes-cli"])


def test_dispatcher_worker_all_toolsets_default_drops_execute_code(monkeypatch):
    assert "execute_code" not in _names(monkeypatch, None)


def test_dispatcher_worker_explicit_code_execution_opt_in(monkeypatch):
    assert "execute_code" in _names(monkeypatch, ["hermes-cli", "code_execution"])


def test_disabled_code_execution_wins_over_explicit_opt_in(monkeypatch):
    assert "execute_code" not in _names(
        monkeypatch,
        ["hermes-cli", "code_execution"],
        ["code_execution"],
    )


def test_interactive_operator_keeps_execute_code(monkeypatch):
    assert "execute_code" in _names(monkeypatch, ["hermes-cli"], task=False)


def test_non_dispatcher_context_keeps_execute_code(monkeypatch):
    assert "execute_code" in _names(monkeypatch, ["hermes-cli"], dispatcher=False)
