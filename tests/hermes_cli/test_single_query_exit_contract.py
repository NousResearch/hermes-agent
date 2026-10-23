"""One-shot ``chat -q`` runs report their outcome in the exit code, like ``-Q`` always did.

The non-quiet one-shot path used to fall through to an implicit rc=0 for every
outcome, so scripts could not tell a failed run from a good one and the Kanban
dispatcher (which spawns ``chat -q`` workers) booked a provider quota wall as a
protocol violation (#111770, #101800; salvage of #110917 / #97623).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE, KANBAN_TERMINAL_PROVIDER_EXIT_CODE


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
        cli._run_single_query_mode(stub, "do the thing", None, False, True)
    except SystemExit as exc:
        return exc.code
    return None


@pytest.mark.parametrize(
    "reason", ["rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout"]
)
def test_dispatcher_spawned_worker_signals_a_provider_outage_not_a_protocol_violation(monkeypatch, reason):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    code = _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason})
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.parametrize(
    "reason", ["auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked"]
)
def test_dispatcher_spawned_worker_signals_a_terminal_provider_error(monkeypatch, reason):
    """A revoked credential / missing model / WAF User-Agent block cannot be retried into working:
    the worker says so with EX_CONFIG so the dispatcher parks the card after one spawn. A person's
    run keeps 1."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason}) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": reason}) == 1


def test_dispatcher_spawned_worker_keeps_a_plain_failure_at_one(monkeypatch):
    """Control: a task-level failure (or an unknown reason) is neither transient nor terminal —
    the worker exits 1 and the dispatcher counts it against ``kanban.failure_limit`` as before."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    assert _run_non_quiet(monkeypatch, {"failed": True, "failure_reason": "some_unknown_reason"}) == 1
    assert _run_non_quiet(monkeypatch, {"failed": True}) == 1


@pytest.mark.parametrize(
    ("turn_result", "expected"),
    [
        ({"final_response": "done", "completed": True}, 0),
        ({"failed": True, "failure_reason": "rate_limit"}, 1),  # a person's run: a wall is just a failure
        ({"final_response": "half", "completed": False, "partial": True}, 1),
        ({"completed": False, "interrupted": True}, 130),
        (None, 1),  # credentials / agent init failed before any turn ran
    ],
)
def test_a_plain_one_shot_run_reports_its_outcome(monkeypatch, turn_result, expected):
    assert _run_non_quiet(monkeypatch, turn_result) == expected


@pytest.mark.parametrize(("rate_limited", "expected"), [(True, KANBAN_RATE_LIMIT_EXIT_CODE), (False, 1)])
def test_quiet_kanban_worker_exits_tempfail_when_credentials_are_rate_limited(monkeypatch, rate_limited, expected):
    """A 429 at startup never produces a turn result; the worker must still exit 75 (#117482).
    A real credential failure keeps exit 1."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    stub = SimpleNamespace(
        _claim_active_session=lambda *a, **k: True,
        _ensure_runtime_credentials=lambda: False,
        _credentials_rate_limited=rate_limited,
        session_id="s1",
        model="gpt-x",
    )
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(stub, "do the thing", None, True, True)
    assert exc.value.code == expected
