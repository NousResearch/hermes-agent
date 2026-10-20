"""Quick snapshots must not traverse excluded working copies or attachments."""
import json
import os
import sqlite3
from pathlib import Path

import pytest


def test_quick_snapshot_prunes_excluded_subtrees_and_keeps_board_wal(tmp_path, monkeypatch):
    from hermes_cli.backup import create_quick_snapshot

    home = tmp_path / 'hermes'
    board = home / 'kanban' / 'boards' / '_archived' / 'example'
    board.mkdir(parents=True)
    (board / 'board.json').write_text('{"name":"example"}')
    (board / '.metadata').write_text('keep hidden files')
    for name in ('workspaces', 'attachments'):
        subtree = board / name / 'nested'
        subtree.mkdir(parents=True)
        (subtree / 'large-regenerable-file').write_text('excluded')
    scanned_excluded = []
    real_scandir = os.scandir

    def track_scandir(path):
        if not isinstance(path, int):
            parts = Path(path).parts
            if 'workspaces' in parts or 'attachments' in parts:
                scanned_excluded.append(str(path))
        return real_scandir(path)

    conn = sqlite3.connect(board / 'kanban.db')
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE notes (body TEXT)')
        conn.execute("INSERT INTO notes VALUES ('committed in WAL')")
        conn.commit()
        monkeypatch.setattr(os, 'scandir', track_scandir)
        snap_id = create_quick_snapshot(hermes_home=home, keep=5)
        snap = home / 'state-snapshots' / snap_id
        copied = snap / board.relative_to(home)
        assert (copied / 'board.json').read_text() == '{"name":"example"}'
        assert (copied / '.metadata').read_text() == 'keep hidden files'
        with sqlite3.connect(copied / 'kanban.db') as saved:
            assert saved.execute('SELECT body FROM notes').fetchall() == [('committed in WAL',)]
        manifest = json.loads((snap / 'manifest.json').read_text())
        assert not any('/workspaces/' in name or '/attachments/' in name for name in manifest['files'])
        assert scanned_excluded == [], 'Excluded trees must be pruned before filesystem traversal'
    finally:
        conn.close()


@pytest.mark.platforms("posix")
def test_quick_snapshot_keeps_file_links_without_following_directory_links(tmp_path):
    from hermes_cli.backup import create_quick_snapshot

    board = tmp_path / "kanban" / "boards" / "example"
    board.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "note.txt").write_text("keep the linked file")
    (board / "note.txt").symlink_to(external / "note.txt")
    (board / "linked-tree").symlink_to(external, target_is_directory=True)
    for name in ("workspaces", "attachments"):
        (board / name).write_text("excluded even when it is a file")

    snapshot_id = create_quick_snapshot(hermes_home=tmp_path)
    snapshot = tmp_path / "state-snapshots" / snapshot_id
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert set(manifest["files"]) == {"kanban/boards/example/note.txt"}
    assert (snapshot / "kanban/boards/example/note.txt").read_text() == "keep the linked file"
