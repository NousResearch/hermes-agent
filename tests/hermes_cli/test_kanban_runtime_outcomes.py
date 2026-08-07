"""Canonical runtime-outcome contract tests for Kanban execution paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.error_classifier import FailoverReason
from hermes_cli import kanban_db as kb
from hermes_cli.runtime_outcomes import (
    RuntimeOutcome,
    outcome_for_provider_reason,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


@pytest.mark.parametrize(
    ("outcome", "counts_against_failure_budget"),
    [
        (RuntimeOutcome.provider_overload(), False),
        (RuntimeOutcome.launcher_transport_failure(), False),
        (RuntimeOutcome.ex_tempfail(), False),
        (RuntimeOutcome.code_failure(), True),
    ],
)
def test_runtime_outcome_contract_has_one_budget_policy(
    outcome, counts_against_failure_budget
):
    assert outcome.counts_against_failure_budget is counts_against_failure_budget
    assert outcome.is_transient is not counts_against_failure_budget


def test_provider_overload_maps_to_transient_outcome():
    outcome = outcome_for_provider_reason(FailoverReason.overloaded)
    assert outcome.kind == "provider_overload"
    assert outcome.is_transient


@pytest.mark.parametrize(
    "serialized",
    [RuntimeOutcome.provider_overload().to_dict(), {"kind": "overloaded"}],
)
def test_serialized_provider_outcome_preserves_transient_policy(serialized):
    outcome = RuntimeOutcome.from_value(serialized)
    assert outcome.kind == "provider_overload"
    assert outcome.is_transient


def test_serialized_provider_rate_limit_preserves_transient_policy():
    outcome = RuntimeOutcome.from_value({"kind": "rate_limit"})
    assert outcome.kind == "rate_limit"
    assert outcome.is_transient


def test_provider_overload_does_not_increment_failure_or_block_recurrence(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="overloaded", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None

        blocked = kb._record_task_failure(
            conn,
            task_id,
            "provider overloaded",
            outcome=RuntimeOutcome.provider_overload(),
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )

        task = kb.get_task(conn, task_id)
        assert blocked is False
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.block_recurrences == 0


def test_launcher_transport_does_not_increment_failure_or_block_recurrence(
    kanban_home, monkeypatch
):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="transport", assignee="worker")
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ConnectionError("launcher transport disconnected")
            ),
            failure_limit=1,
        )

        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.block_recurrences == 0
        assert task_id not in result.auto_blocked
        assert result.runtime_outcomes[0]["kind"] == "launcher_transport_failure"


@pytest.mark.parametrize("status", ["ready", "review"])
def test_launcher_code_failure_is_persisted_as_string_and_blocks_at_limit(
    kanban_home, monkeypatch, status
):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=f"code-{status}", assignee="worker")
        if status == "review":
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
            conn.commit()
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("launcher code failure")
            ),
            failure_limit=1,
        )

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        assert task_id in result.auto_blocked
        assert result.runtime_outcomes[-1]["kind"] == "spawn_failure"
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run["outcome"] == "gave_up"


def test_clean_exit_protocol_violation_is_counting_outcome(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="protocol", assignee="worker")
        task = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:worker")
        assert task is not None
        pid = 99124
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, task_id))
        conn.commit()
        kb._record_worker_exit(pid, 0)

        kb.detect_crashed_workers(conn)

        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert '"kind": "code_failure"' in event["payload"]
        outcome = kb.detect_crashed_workers._last_runtime_outcomes[-1]
        assert outcome["kind"] == "code_failure"


def test_ex_tempfail_is_transient_worker_exit():
    pid = os.getpid()
    kb._record_worker_exit(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8)
    outcome = kb.worker_exit_outcome(pid)
    assert outcome.kind == "ex_tempfail"
    assert outcome.is_transient


def test_ordinary_worker_exit_still_counts_as_failure(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="crash", assignee="worker")
        task = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:worker")
        assert task is not None
        pid = 99123
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, task_id))
        conn.commit()
        kb._record_worker_exit(pid, 1 << 8)

        kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, task_id)
        assert task.consecutive_failures == 1
        assert task.status == "ready"
