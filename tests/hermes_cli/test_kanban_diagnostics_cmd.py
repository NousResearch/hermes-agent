"""``hermes kanban diagnostics`` fleet mode excludes terminal-status cards (t_1debdb38).

A done card can never clear event-backed diagnostics (the advisory
``suspected_hallucinated_references`` event is emitted after the ``completed``
event and a done card is never edited again), so the CLI fleet view must match
what the dashboard attention surfaces show: open/active cards only. One-task
mode (``--task``) stays unfiltered so per-card history stays inspectable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban import _cmd_diagnostics


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_done_and_active(kanban_home) -> tuple[str, str]:
    """Done card with the real post-completion suspected event + active control card."""
    with kbc.connect_closing() as conn:
        a_id = kb.create_task(conn, title="done card", assignee="web")
        b_id = kb.create_task(conn, title="active card", assignee="web")
        assert kb.complete_task(
            conn, a_id, summary="done for t_deadc0de00", result="ok", fire_lifecycle_hook=False)
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, ?, ?)",
                (b_id, "suspected_hallucinated_references",
                 json.dumps({"phantom_refs": ["t_deadc0de01"], "source": "completion_summary"}),
                 int(time.time())),
            )
    return a_id, b_id


def test_fleet_mode_excludes_terminal_cards(kanban_home, capsys):
    a_id, b_id = _seed_done_and_active(kanban_home)

    rc = _cmd_diagnostics(argparse.Namespace(task=None, severity=None, json=True))
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    ids = {r["task_id"] for r in rows}
    assert b_id in ids
    assert a_id not in ids
    active_row = next(r for r in rows if r["task_id"] == b_id)
    assert any(d["kind"] == "prose_phantom_refs" for d in active_row["diagnostics"])


def test_one_task_mode_keeps_terminal_history(kanban_home, capsys):
    a_id, _b_id = _seed_done_and_active(kanban_home)

    rc = _cmd_diagnostics(argparse.Namespace(task=a_id, severity=None, json=True))
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(
        r["task_id"] == a_id and any(d["kind"] == "prose_phantom_refs" for d in r["diagnostics"])
        for r in rows)
