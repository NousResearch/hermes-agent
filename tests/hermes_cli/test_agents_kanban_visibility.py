"""Kanban running tasks in /agents output (CLI + gateway).

/agents previously reported only the session's own children (process
registry, async delegations). Dispatcher-spawned kanban workers are separate
processes on separate profiles, invisible to both — the shared board is the
only authoritative source. These tests drive the REAL snapshot function
against a REAL sqlite board via the public kanban_db API (no mocked DB), plus
the gateway projection and the CLI section rendering.
"""

import asyncio

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    """Point the shared board at a temp root and open a default-board DB."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    conn = kbc.connect()  # initializes + schema + WAL on the temp board
    yield conn
    conn.close()


def _seed(conn, title, assignee):
    """Create + claim a task the way the dispatcher does (ready -> running)."""
    task_id = kb.create_task(conn, title=title, assignee=assignee, created_by="test")
    claimed = kb.claim_task(conn, task_id, claimer="test-worker")
    assert claimed is not None
    return task_id


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_key_for_source = lambda source: "agent:main:test:dm:1"
    return runner


class _Event:
    source = None


def test_list_running_tasks_reports_dispatcher_workers(kanban_home):
    from hermes_cli.kanban_status import list_running_tasks

    t1 = _seed(kanban_home, "web: brand kit", "frontend-dev")
    t2 = _seed(kanban_home, "scraper: reconcile", "backend-dev")
    rows = list_running_tasks()
    ids = {r["task_id"] for r in rows}
    assert {t1, t2} <= ids
    by_id = {r["task_id"]: r for r in rows}
    assert by_id[t1]["assignee"] == "frontend-dev"
    assert by_id[t1]["board"] == "default"
    assert by_id[t1]["elapsed_seconds"] is not None and by_id[t1]["elapsed_seconds"] >= 0
    # Done tasks must not leak into the running view.
    kb.complete_task(kanban_home, t2, result="done")
    ids = {r["task_id"] for r in list_running_tasks()}
    assert t1 in ids and t2 not in ids


def test_list_running_tasks_empty_board_returns_empty(kanban_home):
    from hermes_cli.kanban_status import list_running_tasks

    assert list_running_tasks() == []


def test_list_running_tasks_never_creates_or_initializes(tmp_path, monkeypatch):
    """A bare HERMES_KANBAN_HOME must not gain a kanban.db from a status read."""
    from hermes_cli.kanban_status import list_running_tasks

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    assert list_running_tasks() == []
    assert not (tmp_path / "kanban.db").exists()


def test_list_running_tasks_skips_unreadable_board(kanban_home):
    """One corrupt board must not hide the others (fail-open per board)."""
    from hermes_cli import kanban_status

    _seed(kanban_home, "web: brand kit", "frontend-dev")
    (kb.boards_root() / "junkboard").mkdir(parents=True)
    junk = kb.boards_root() / "junkboard" / "kanban.db"
    junk.write_bytes(b"this is not sqlite")
    rows = kanban_status.list_running_tasks()
    assert any(r["task_id"] for r in rows)


def test_gateway_agents_includes_kanban_section(kanban_home):
    _seed(kanban_home, "web: brand kit", "frontend-dev")
    runner = _runner()
    out = asyncio.run(runner._handle_agents_command(_Event()))
    assert "Kanban tasks running" in out
    assert "web: brand kit" in out
    assert "frontend-dev" in out


def test_gateway_agents_empty_board_shows_none_line(kanban_home):
    runner = _runner()
    out = asyncio.run(runner._handle_agents_command(_Event()))
    # Empty board → canonical "nothing" line, no kanban section.
    assert "Kanban tasks running" not in out
    assert "No active agents or running tasks." in out


def test_cli_agents_prints_kanban_section(kanban_home, capsys):
    _seed(kanban_home, "scraper: reconcile", "backend-dev")
    from cli import HermesCLI  # the real class owning the handler

    instance = object.__new__(HermesCLI)
    instance._agent_running = False
    instance._handle_agents_command()
    captured = capsys.readouterr().out
    assert "Kanban tasks running: 1" in captured
    assert "scraper: reconcile" in captured
    assert "backend-dev" in captured
