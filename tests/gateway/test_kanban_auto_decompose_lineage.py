"""Auto-decompose tick skips re-triaged decompose children (#140323).

A dashboard bulk move parking decompose children back in triage must not feed
them to the fan-out again (geometric cascade); a hand-written triage card in
the same tick is still decomposed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_auto_decompose_tick_skips_retriaged_decompose_children(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    with kbc.connect() as conn:
        root = kb.create_task(conn, title="rough idea", triage=True)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orchestrator",
            children=[{"title": "child A", "assignee": "researcher"}],
            author="auto-decomposer",
        )
        assert child_ids
        fresh = kb.create_task(conn, title="fresh rough idea", triage=True)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (child_ids[0],))

    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default"])
    seen = []

    def fake_decompose(task_id, author=None):
        seen.append(task_id)
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=lambda: [child_ids[0], fresh], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    decomposed = asyncio.run(_to_thread_process_service(_dispatcher().auto_decompose_tick, 5))

    assert decomposed == 1
    assert seen == [fresh]
