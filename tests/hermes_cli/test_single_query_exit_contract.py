"""A dispatcher-spawned ``chat -q`` worker reports its outcome in its exit code.

The Kanban dispatcher spawns workers as ``hermes ... chat -q <prompt>`` (the
non-quiet one-shot path), which used to fall through to an implicit rc=0 for
every outcome. The reaper reads rc=0 with the task still ``running`` as a
protocol violation, so a provider quota wall re-dispatched the card straight
back into the same wall (#101800, #48000, #91177; salvage of #110917).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.fixture(autouse=True)
def _no_inherited_kanban_env(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


def _run_non_quiet(monkeypatch, turn_result):
    """Drive ``_run_single_query_mode`` down the non-quiet tail; return the exit code (None = fell through)."""
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=SimpleNamespace(print=lambda *a, **k: None),
        _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response",
        _print_exit_summary=lambda **k: None,
        _last_turn_result=turn_result,
    )
    try:
        cli._run_single_query_mode(stub, "work kanban task t_abc123", None, False, True)
    except SystemExit as exc:
        return exc.code
    return None


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_dispatcher_spawned_worker_signals_a_quota_wall_not_a_protocol_violation(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason})
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_a_human_one_shot_run_is_unaffected(monkeypatch):
    """No HERMES_KANBAN_TASK: a person's ``-q`` run keeps exiting 0 even when the turn failed."""
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": "rate_limit"})
    assert code is None
