from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _blocked(conn, kind: str = "needs_input") -> str:
    tid = kb.create_task(conn, title="blocked", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker")
    assert kb.block_task(conn, tid, reason="original reason", kind=kind)
    return tid


@pytest.mark.parametrize(
    ("old_kind", "new_kind"),
    [
        ("needs_input", "dependency"),
        ("dependency", "transient"),
        ("transient", "capability"),
        ("capability", "needs_input"),
    ],
)
def test_reclassify_changes_only_kind_and_appends_audit(
    board: Path,
    old_kind: str,
    new_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        kb,
        "_fire_kanban_lifecycle_hook",
        lambda *args, **kwargs: lifecycle_calls.append(args),
    )
    with kb.connect_closing() as conn:
        tid = _blocked(conn)
        lifecycle_calls.clear()
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind=?, block_recurrences=7 WHERE id=?",
                (old_kind, tid),
            )
        before = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
        runs_before = [dict(row) for row in conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,)
        )]
        assert kb.reclassify_blocked_task(
            conn, tid, kind=new_kind, actor="operator", note="classifier fix"
        )
        after = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
        runs_after = [dict(row) for row in conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (tid,)
        )]
        assert after.pop("block_kind") == new_kind
        assert before.pop("block_kind") == old_kind
        assert after == before
        assert runs_after == runs_before
        event = [e for e in kb.list_events(conn, tid) if e.kind == "block_reclassified"][-1]
        assert event.payload == {
            "old_kind": old_kind,
            "new_kind": new_kind,
            "actor": "operator",
            "note": "classifier fix",
        }
        assert "BLOCK CLASSIFIER" in kb.list_comments(conn, tid)[-1].body
        assert lifecycle_calls == []


def test_reclassify_is_blocked_only(board: Path) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="ready", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert not kb.reclassify_blocked_task(
            conn, tid, kind="capability", actor="operator"
        )
        assert kb.get_task(conn, tid).status == "ready"


def test_existing_block_operation_still_refuses_already_blocked(board: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked(conn)
        assert not kb.block_task(
            conn, tid, reason="must still refuse", kind="capability"
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_reclassify_survives_copied_wal_board(board: Path, tmp_path: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked(conn)
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    copied = tmp_path / "copied.db"
    shutil.copy2(kb.kanban_db_path(), copied)
    conn = kb.connect(copied)
    try:
        assert kb.reclassify_blocked_task(
            conn, tid, kind="transient", actor="operator"
        )
        assert kb.get_task(conn, tid).block_kind == "transient"
    finally:
        conn.close()


def _parse(argv: list[str]) -> argparse.Namespace:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    cli.build_parser(sub)
    return root.parse_args(["kanban", *argv])


def test_cli_reclassify_enforces_worker_scope_and_serializes(
    board: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with kb.connect_closing() as conn:
        own = _blocked(conn)
        other = _blocked(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", own)
    assert cli.kanban_command(
        _parse(["reclassify-block", other, "--kind", "capability"])
    ) == 1
    assert "may only mutate" in capsys.readouterr().err
    assert cli.kanban_command(
        _parse(["reclassify-block", own, "--kind", "capability", "--note", "fix"])
    ) == 0
    assert f"Reclassified {own}: capability" in capsys.readouterr().out
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert cli.kanban_command(
        _parse(["reclassify-block", other, "--kind", "transient"])
    ) == 0
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, own).block_kind == "capability"
        assert kb.get_task(conn, other).block_kind == "transient"
