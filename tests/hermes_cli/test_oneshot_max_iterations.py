"""oneshot must honor HERMES_MAX_ITERATIONS (bridge / headless callers)."""

from __future__ import annotations

import sys

from hermes_cli.oneshot import _resolve_oneshot_max_iterations


def test_env_max_iterations_wins_over_config(monkeypatch):
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "10")
    assert _resolve_oneshot_max_iterations({"agent": {"max_turns": 120}}) == 10


def test_config_agent_max_turns_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    assert _resolve_oneshot_max_iterations({"agent": {"max_turns": 42}}) == 42


def test_root_max_turns_fallback(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    assert _resolve_oneshot_max_iterations({"max_turns": 7}) == 7


def test_unset_defaults_to_unlimited(monkeypatch):
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
    assert _resolve_oneshot_max_iterations({}) == sys.maxsize
