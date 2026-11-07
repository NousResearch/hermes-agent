"""Scratch workspaces holding read-only files are removed, not half-removed.

Git marks pack, index and loose object files read-only, so on Windows
``os.unlink`` fails on them with ``WinError 5``. Completion, archive, the
deferred parent sweep (#33774) and ``kanban gc`` all used
``shutil.rmtree(..., ignore_errors=True)``, which skipped those files without a
word: every finished task that cloned a repo left its packs, and the directory,
on disk. On POSIX the same failure is a read-only directory, since unlinking
needs a writable parent; the planted tree covers both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _plant_read_only_pack(ws: Path) -> None:
    """A clone-shaped tree: a read-only pack file in a read-only directory."""
    pack_dir = ws / "repo" / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    pack = pack_dir / "pack-0123456789abcdef0123456789abcdef01234567.pack"
    pack.write_bytes(b"PACK")
    pack.chmod(0o444)
    pack_dir.chmod(0o555)  # POSIX unlink needs a writable parent


def _scratch_task(conn, title: str = "task", *, running: bool = True) -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title)
    if running:
        assert kb.claim_task(conn, tid) is not None
    ws = Path(kbw.resolve_workspace(kb.get_task(conn, tid)))
    kbw.set_workspace_path(conn, tid, ws)
    _plant_read_only_pack(ws)
    return tid, ws


def test_complete_task_removes_read_only_scratch(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn)
        assert kb.complete_task(conn, tid, summary="done")
    assert not ws.exists()


def test_archive_task_removes_read_only_scratch(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, ws = _scratch_task(conn, running=False)
        assert kb.archive_task(conn, tid)
    assert not ws.exists()


def test_deferred_parent_sweep_removes_read_only_scratch(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parent, ws = _scratch_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert kb.complete_task(conn, parent, summary="parent done")
        assert ws.is_dir()  # the live child still needs the handoff files
        assert kb.archive_task(conn, child)
    assert not ws.exists()


def test_gc_removes_read_only_scratch(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="archived")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', workspace_kind='scratch' WHERE id=?",
                (tid,),
            )
    ws = kb.workspaces_root() / tid
    _plant_read_only_pack(ws)
    args = argparse.Namespace(event_retention_days=30, log_retention_days=30)
    assert kanban_ops._cmd_gc(args) == 0
    assert not ws.exists()
