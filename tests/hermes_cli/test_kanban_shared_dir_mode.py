"""Card dirs must inherit a shared (setgid) board tree's mode at creation.

Invariants:

1. A per-card dir created under a tree that declares itself shared (the setgid
   bit, i.e. 2775) is born group-writable -- that is the property the tree
   declares. A bare ``mkdir`` cannot promise it: mkdir applies the creating
   process's umask, so a worker running umask 022 under a 2775 root creates 2755
   -- the setgid bit IS inherited from the parent, group WRITE is not -- and the
   other user on that tree cannot write into the card dir.
2. Every component ``mkdir_aligned`` creates is aligned, not just the leaf, so a
   missing intermediate root is created right the first time.
3. A tree that does NOT declare itself shared keeps the historical umask
   behaviour, so the change is invisible to single-user installs.
4. A dir that already exists is never re-moded: creation is the only moment this
   helper owns.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_db_workspace import resolve_workspace


def _setgid_survives_here() -> bool:
    """BSD/macOS silently drop S_ISGID from directories.

    Where that happens a shared tree cannot be expressed at all, so the mode
    property is moot and every test here is skipped rather than red.
    """
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe"
        probe.mkdir()
        os.chmod(probe, 0o2775)
        return bool(os.stat(probe).st_mode & stat.S_ISGID)


pytestmark = pytest.mark.skipif(
    not _setgid_survives_here(),
    reason=(
        "filesystem drops the setgid bit on directories, so a shared tree cannot "
        "be expressed here"
    ),
)


@pytest.fixture
def umask_022():
    """The umask an unattended worker runs with. Saved and restored: umask is
    process-global and leaking 022 into the rest of the suite would be rude."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def shared_board_home(tmp_path, monkeypatch):
    """A board home whose per-card roots are laid out as a SHARED tree (2775).

    Mirrors what an operator (or a provisioning step) does for two OS users on
    one board tree; the code under test is what must keep it true afterwards.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ATTACHMENTS_ROOT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    kb.init_db()
    for root in (kb.attachments_root(), kb.workspaces_root()):
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o2775)
    return home


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_bare_mkdir_loses_group_write_under_a_shared_root(tmp_path, umask_022):
    """Negative control: the defect the helper exists for.

    If this ever stops holding the helper is unnecessary -- and if it holds but
    the tests below pass too, the helper is doing what it claims.
    """
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2775)
    card = root / "t_deadbeef"
    card.mkdir(parents=True, exist_ok=True)  # what the old call sites did
    assert _mode(card) == 0o2755, (
        "a bare mkdir under a 2775 root is expected to lose group write "
        "(sgid inherited, umask 022 applied); if this fails the platform no "
        "longer behaves as the fix assumes"
    )


def test_card_dir_inherits_the_shared_roots_mode(tmp_path, umask_022):
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2775)
    card = kb.mkdir_aligned(root / "t_deadbeef")
    assert _mode(card) == 0o2775


def test_every_component_is_aligned_not_just_the_leaf(tmp_path, umask_022):
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2775)
    card = kb.mkdir_aligned(root / "missing-root" / "t_deadbeef")
    assert _mode(root / "missing-root") == 0o2775
    assert _mode(card) == 0o2775


def test_a_private_tree_keeps_umask_behaviour(tmp_path, umask_022):
    root = tmp_path / "private"
    root.mkdir(mode=0o755)
    card = kb.mkdir_aligned(root / "t_deadbeef")
    assert _mode(card) == 0o755


def test_an_existing_dir_is_never_remoded(tmp_path, umask_022):
    root = tmp_path / "shared"
    root.mkdir()
    os.chmod(root, 0o2775)
    existing = root / "t_deadbeef"
    existing.mkdir()
    os.chmod(existing, 0o700)
    assert _mode(kb.mkdir_aligned(existing)) == 0o700


def test_the_upload_path_creates_a_group_writable_card_dir(shared_board_home, umask_022):
    """The real write path (dashboard / tool / CLI uploads), not the helper alone."""
    conn = connect()
    task_id = kb.create_task(conn, title="shared tree card")
    kb.store_attachment_bytes(conn, task_id, "note.txt", b"hi")
    card = kb.task_attachments_dir(task_id)
    assert _mode(card) == 0o2775
    assert (card / "note.txt").read_bytes() == b"hi"


def test_the_scratch_workspace_resolver_creates_a_group_writable_card_dir(
    shared_board_home, umask_022
):
    conn = connect()
    task_id = kb.create_task(conn, title="shared tree workspace")
    task = kb.get_task(conn, task_id)
    assert task is not None
    workspace = resolve_workspace(task)
    assert _mode(workspace) == 0o2775
