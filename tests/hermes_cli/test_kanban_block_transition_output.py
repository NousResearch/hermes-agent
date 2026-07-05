from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hermes")
    sub = root.add_subparsers(dest="command")
    kc.build_parser(sub)
    return root


def _run_kanban(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    args = _parser().parse_args(["kanban", *argv])
    rc = kc.kanban_command(args)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _running_task(conn, title: str = "task") -> str:
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def test_block_success_prints_changed_true_and_verified_status(
    kanban_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)

    rc, out, err = _run_kanban(
        ["block", tid, "need", "operator", "--kind", "needs_input"],
        capsys,
    )

    assert rc == 0
    assert err == ""
    assert f"task_id={tid}" in out
    assert "changed=true" in out
    assert "status=blocked" in out
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_block_no_transition_returns_nonzero_changed_false_and_no_comment_by_default(
    kanban_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="already done", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))

    rc, out, err = _run_kanban(["block", tid, "too", "late"], capsys)

    assert rc == 1
    assert out == ""
    assert f"cannot block {tid}" in err
    assert "changed=false" in err
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"
        assert kb.list_comments(conn, tid) == []


def test_block_comment_only_fallback_is_explicit_and_still_reports_changed_false(
    kanban_home: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="already done", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))

    rc, out, err = _run_kanban(
        ["block", tid, "audit", "note", "--comment-only-ok"],
        capsys,
    )

    assert rc == 0
    assert err == ""
    assert f"task_id={tid}" in out
    assert "changed=false" in out
    assert "comment_only=true" in out
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"
        comments = kb.list_comments(conn, tid)
    assert len(comments) == 1
    assert comments[0].body == "BLOCKED: audit note"
