"""Registration of dispatcher worker sessions in single-query mode."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cli as cli_module
import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_single_query_worker_registers_root_session_before_work(
    kanban_home, monkeypatch
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="builder")
        kb.claim_task(conn, task_id, claimer="host:worker")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))

    cli_module._bind_kanban_worker_session_q(
        SimpleNamespace(session_id="root-worker-session")
    )

    with kb.connect() as conn:
        run = kb.get_run(conn, task.current_run_id)
        events = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "worker_session_bound"
        ]

    assert run is not None
    assert run.worker_session_id == "root-worker-session"
    assert len(events) == 1
