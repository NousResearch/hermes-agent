"""Worker exit-code and dispatcher classification for Kanban protocol failures."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_exit_codes import (
    KANBAN_PROTOCOL_EXIT_CODE,
    KANBAN_RATE_LIMIT_EXIT_CODE,
    single_query_exit_code,
)
from agent.kanban_stop import HandoffStatus, assess_kanban_handoff
from tools.kanban_tools import _handle_block, _handle_complete


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb.Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _terminal_messages(name: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-real",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": name,
            "tool_call_id": "call-real",
            "content": content,
        },
    ]


def _claim_with_pid(conn, *, title: str, pid: int, max_runtime: int | None = None):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="worker",
        max_runtime_seconds=max_runtime,
    )
    claimed = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
    assert claimed is not None
    conn.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?",
        (pid, task_id),
    )
    conn.commit()
    return task_id, claimed.current_run_id


def test_single_query_exit_code_distinguishes_worker_outcomes() -> None:
    assert single_query_exit_code({"failed": False}, kanban_worker=True) == 0
    assert single_query_exit_code({"failed": True}, kanban_worker=True) == 1
    assert (
        single_query_exit_code(
            {"failed": True, "failure_reason": "rate_limit"},
            kanban_worker=True,
        )
        == KANBAN_RATE_LIMIT_EXIT_CODE
    )
    assert (
        single_query_exit_code(
            {"failed": True, "failure_reason": "kanban_protocol"},
            kanban_worker=True,
        )
        == KANBAN_PROTOCOL_EXIT_CODE
    )
    assert (
        single_query_exit_code(
            {"failed": True, "failure_reason": "kanban_protocol"},
            kanban_worker=False,
        )
        == 1
    )


def test_real_complete_handler_emits_a_valid_post_commit_receipt(
    kanban_home: Path,
    monkeypatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="finish me", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        assert claimed is not None
        assert claimed.current_run_id is not None
        run_id = claimed.current_run_id

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    content = _handle_complete({"summary": "verified"})
    assessment = assess_kanban_handoff(
        _terminal_messages("kanban_complete", content)
    )
    assert assessment.status is HandoffStatus.VALID
    assert assessment.successful_count == 1

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.id == run_id
        assert run.outcome == "completed"


def test_real_block_handler_is_a_valid_external_blocker_handoff(
    kanban_home: Path,
    monkeypatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="need a decision", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        assert claimed is not None
        assert claimed.current_run_id is not None
        run_id = claimed.current_run_id

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    content = _handle_block(
        {"reason": "operator decision required", "kind": "needs_input"}
    )
    assessment = assess_kanban_handoff(
        _terminal_messages("kanban_block", content)
    )
    assert assessment.status is HandoffStatus.VALID

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "blocked"


def test_protocol_exit_is_not_classified_as_process_crash(
    kanban_home: Path,
    monkeypatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="handoff required", assignee="worker")
        host_prefix = kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, task_id, claimer=f"{host_prefix}:test")
        assert claimed is not None
        fake_pid = 991940
        kb._set_worker_pid(conn, task_id, fake_pid)
        kb._record_worker_exit(fake_pid, KANBAN_PROTOCOL_EXIT_CODE << 8)
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        reclaimed = kb.detect_crashed_workers(conn)

        assert task_id in reclaimed  # compatibility: reaped worker ids
        assert task_id in getattr(kb.detect_crashed_workers, "_last_protocol_violations")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        events = kb.list_events(conn, task_id)
        protocol = [event for event in events if event.kind == "protocol_violation"]
        assert len(protocol) == 1
        payload = protocol[0].payload
        assert payload is not None
        assert payload["exit_code"] == KANBAN_PROTOCOL_EXIT_CODE
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "protocol_violation"
        assert not any(event.kind == "crashed" for event in events)


@pytest.mark.parametrize(
    ("label", "raw_status", "event_kind", "run_outcome", "exit_kind"),
    [
        ("clean-output-0", 0, "protocol_violation", "protocol_violation", "clean_exit"),
        ("exception-exit-1", 1 << 8, "crashed", "crashed", "nonzero_exit"),
        ("sigterm", int(signal.SIGTERM), "crashed", "crashed", "signaled"),
    ],
)
def test_reaper_distinguishes_output_zero_exception_and_signal(
    kanban_home: Path,
    monkeypatch,
    label: str,
    raw_status: int,
    event_kind: str,
    run_outcome: str,
    exit_kind: str,
) -> None:
    pid = {
        "clean-output-0": 991950,
        "exception-exit-1": 991951,
        "sigterm": 991952,
    }[label]
    with kb.connect_closing() as conn:
        task_id, _ = _claim_with_pid(conn, title=label, pid=pid)
        kb._record_worker_exit(pid, raw_status)
        monkeypatch.setattr(kb, "_pid_alive", lambda candidate: False)
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        reclaimed = kb.detect_crashed_workers(conn)
        assert task_id in reclaimed
        events = kb.list_events(conn, task_id)
        terminal = [event for event in events if event.kind == event_kind]
        assert len(terminal) == 1
        payload = terminal[0].payload
        assert payload is not None
        assert payload["exit_kind"] == exit_kind
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == run_outcome

        protocol_ids = getattr(
            kb.detect_crashed_workers,
            "_last_protocol_violations",
            [],
        )
        assert (task_id in protocol_ids) is (raw_status == 0)


def test_timeout_remains_distinct_from_protocol_violation(
    kanban_home: Path,
    monkeypatch,
) -> None:
    pid = 991960
    sent: list[tuple[int, int]] = []
    with kb.connect_closing() as conn:
        task_id, run_id = _claim_with_pid(
            conn,
            title="deadline",
            pid=pid,
            max_runtime=1,
        )
        assert run_id is not None
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (int(time.time()) - 5, run_id),
        )
        conn.commit()
        monkeypatch.setattr(kb, "_pid_alive", lambda candidate: False)

        timed_out = kb.enforce_max_runtime(
            conn,
            signal_fn=lambda candidate, sig: sent.append((candidate, int(sig))),
        )
        assert timed_out == [task_id]
        assert sent == [(pid, int(signal.SIGTERM))]
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        events = kb.list_events(conn, task_id)
        assert any(event.kind == "timed_out" for event in events)
        assert not any(event.kind == "protocol_violation" for event in events)
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.outcome == "timed_out"


def test_protocol_exit_code_matches_posix_ex_protocol() -> None:
    assert KANBAN_PROTOCOL_EXIT_CODE == getattr(os, "EX_PROTOCOL", 76)
