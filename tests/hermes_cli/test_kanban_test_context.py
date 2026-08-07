"""Regression tests for hermetic Kanban E2E/probe harnesses."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


def _create_sentinel_db(path: Path) -> bytes:
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE production_sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO production_sentinel VALUES ('untouched')")
    return path.read_bytes()


def test_isolated_context_ignores_worker_pins_and_preserves_pinned_db(
    tmp_path, monkeypatch
):
    production_db = tmp_path / "production" / "kanban.db"
    production_before = _create_sentinel_db(production_db)
    worker_env = {
        "HERMES_HOME": str(tmp_path / "worker-profile"),
        "HERMES_KANBAN_DB": str(production_db),
        "HERMES_KANBAN_BOARD": "pascas-labs",
        "HERMES_KANBAN_HOME": str(production_db.parent),
        "HERMES_KANBAN_WORKSPACES_ROOT": str(production_db.parent / "workspaces"),
        "HERMES_KANBAN_LOGS_ROOT": str(production_db.parent / "logs"),
        "HERMES_KANBAN_TASK": "t_production_worker",
        "HERMES_KANBAN_FUTURE_ROUTING_PIN": str(production_db.parent),
    }
    for name, value in worker_env.items():
        monkeypatch.setenv(name, value)

    from hermes_cli.kanban_test_context import isolated_kanban_test_context

    with isolated_kanban_test_context() as isolated:
        assert not any(name.startswith("HERMES_KANBAN_") for name in os.environ)
        assert Path(os.environ["HERMES_HOME"]).resolve() == isolated.home

        from hermes_cli import kanban_db as kb

        assert kb.kanban_db_path().resolve() == isolated.db_path
        isolated.db_path.relative_to(isolated.home)
        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="regression probe",
                assignee="probe-profile",
            )
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.title == "regression probe"

    assert production_db.read_bytes() == production_before
    for name, value in worker_env.items():
        assert os.environ[name] == value


def test_isolated_context_fails_closed_when_db_resolves_outside_temp_home(
    tmp_path, monkeypatch
):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_test_context import isolated_kanban_test_context

    outside_db = tmp_path / "outside" / "kanban.db"
    monkeypatch.setattr(kb, "kanban_db_path", lambda: outside_db)

    with pytest.raises(RuntimeError, match="outside its temporary HERMES_HOME"):
        with isolated_kanban_test_context():
            pass
