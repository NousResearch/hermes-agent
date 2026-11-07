"""Lock files for a symlinked kanban.db anchor at its target (issue #135107).

``_cross_process_init_lock`` / ``_dispatch_tick_lock`` derived their lock paths
from the symlink itself. SQLite follows the link and keeps ``-wal``/``-shm``
next to the TARGET, so the locks belong there too; on a board whose link sits
in a read-only directory (``ProtectSystem=strict``) the old placement made the
init-lock ``open()`` fail every connect (it is outside the try) and degraded
the dispatch lock to a no-op — the single-writer guard silently gone.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def symlinked_board(tmp_path):
    """A kanban.db symlink in its own directory pointing at a writable target."""
    link_dir = tmp_path / "link-dir"
    target_dir = tmp_path / "target-dir"
    link_dir.mkdir()
    target_dir.mkdir()
    target = target_dir / "kanban.db"
    target.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    link = link_dir / "kanban.db"
    os.symlink(target, link)
    return link, target


@pytest.mark.require_symlinks
def test_init_lock_lands_next_to_target(symlinked_board):
    link, target = symlinked_board
    with kbc._cross_process_init_lock(link):
        assert (target.parent / "kanban.db.init.lock").exists()
    assert not (link.parent / "kanban.db.init.lock").exists()


@pytest.mark.require_symlinks
def test_dispatch_lock_lands_next_to_target_and_holds(symlinked_board):
    link, target = symlinked_board
    with kbc._dispatch_tick_lock(link) as acquired:
        assert acquired is True
    assert (target.parent / "kanban.db.dispatch.lock").exists()
    assert not (link.parent / "kanban.db.dispatch.lock").exists()


@pytest.mark.require_symlinks
@pytest.mark.skipif(os.name == "nt", reason="POSIX read-only directory semantics")
def test_init_lock_survives_read_only_link_dir(symlinked_board, tmp_path):
    """The #135107 repro: the link's directory is read-only, the target's is not.

    The lock ``open()`` is outside the try block by design (an init we cannot
    serialize is a hard error), so it must target the writable directory."""
    link, target = symlinked_board
    link_dir = link.parent
    link_dir.chmod(0o500)
    try:
        with kbc._cross_process_init_lock(link):
            assert (target.parent / "kanban.db.init.lock").exists()
    finally:
        link_dir.chmod(0o700)  # let tmp_path cleanup rm the directory


@pytest.mark.require_symlinks
@pytest.mark.skipif(os.name == "nt", reason="POSIX read-only directory semantics")
def test_dispatch_lock_survives_read_only_link_dir(symlinked_board):
    """On a read-only link dir the dispatch lock must hold at the target, not
    degrade to the no-op (acquired=True with no file) the except branch gives."""
    link, target = symlinked_board
    link_dir = link.parent
    link_dir.chmod(0o500)
    try:
        with kbc._dispatch_tick_lock(link) as acquired:
            assert acquired is True
        assert (target.parent / "kanban.db.dispatch.lock").exists()
    finally:
        link_dir.chmod(0o700)


def test_plain_board_path_is_not_normalized(tmp_path):
    """A non-symlink board keeps its exact (even relative) path — the anchor
    only follows actual symlinks, so cache keys and log lines are unchanged."""
    plain = tmp_path / "kanban.db"
    plain.write_bytes(b"")
    assert kbc._lock_anchor(plain) is plain
    assert kbc._lock_anchor(Path("relative/kanban.db")) == Path("relative/kanban.db")
