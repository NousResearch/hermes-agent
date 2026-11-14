"""Regression coverage for per-run Kanban worker logs and clean-exit evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_clean_exit_keeps_each_run_log_and_records_early_stderr(kanban_home, monkeypatch):
    """A retry must not overwrite the clean-exit run's diagnostic output."""
    monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_args: False)

    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="run log retention", assignee="worker")
        first = kb.claim_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        with kbd._open_worker_log(first, None) as log:
            log.write(b"first run stderr: missing session row\n")
        kbd._set_worker_pid(conn, task_id, 640001)
        kbd._record_worker_exit(640001, 0)
        assert kbd.detect_crashed_workers(conn) == [task_id]

        protocol_event = next(
            event for event in kb.list_events(conn, task_id)
            if event.kind == "protocol_violation"
        )
        assert protocol_event.payload["stderr_first_60s"] == (
            "first run stderr: missing session row\n"
        )

        second = kb.claim_task(conn, task_id)
        assert second is not None and second.current_run_id is not None
        with kbd._open_worker_log(second, None) as log:
            log.write(b"retry output\n")

        paths = sorted((kanban_home / "kanban" / "logs").glob(f"{task_id}.*.log"))
        assert paths == [
            kb.worker_run_log_path(task_id, first.current_run_id),
            kb.worker_run_log_path(task_id, second.current_run_id),
        ]
        assert paths[0].read_text(encoding="utf-8") == "first run stderr: missing session row\n"
        assert paths[1].read_text(encoding="utf-8") == "retry output\n"
    finally:
        conn.close()
