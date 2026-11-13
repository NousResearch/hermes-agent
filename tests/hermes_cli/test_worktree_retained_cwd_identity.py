"""A retained ``--tui -w`` tree stays live when its session's stored cwd spells the same native
path differently (Windows: case-insensitive, either separator)."""
import ntpath
import os
import sqlite3

import pytest

from hermes_cli.worktree_ops import _retained_tree_is_live


def _store(tmp_path, cwd):
    db = tmp_path / 'state.db'
    conn = sqlite3.connect(db)
    conn.executescript('CREATE TABLE sessions (id TEXT, cwd TEXT, message_count INTEGER);'
                       'CREATE TABLE session_admissions (target_session_id TEXT, status TEXT);')
    conn.execute('INSERT INTO sessions VALUES (?, ?, 3)', ('s1', cwd))
    conn.commit()
    conn.close()
    return str(db)


@pytest.mark.parametrize('spell, live', [
    (lambda p: p.upper(), True),                               # same path, other casing
    (lambda p: p.upper().replace('/', '\\') + '\\sub', True),  # nested, backslash separators
    (lambda p: p + '-other', False),                           # sibling prefix is not inside
    (lambda p: os.path.dirname(p), False),                     # parent dir is not inside
])
def test_retained_tree_uses_windows_path_identity(tmp_path, monkeypatch, spell, live):
    tree = tmp_path / 'Tree'
    tree.mkdir()
    db = _store(tmp_path, spell(str(tree.resolve())))
    monkeypatch.setattr(os.path, 'normcase', ntpath.normcase)  # the native Windows identity
    assert _retained_tree_is_live(db, str(tree)) is live
