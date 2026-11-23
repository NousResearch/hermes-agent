"""A retained ``--tui -w`` tree stays live when its session's stored cwd spells the same native
path differently (Windows: case-insensitive, either separator)."""
import ntpath
import os

import pytest

from hermes_cli.worktree_ops import _retained_tree_is_live
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _store(tmp_path, cwd):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
        seed_session(store, 's1', 'cli', cwd=cwd)
        seed_messages(store, 's1', *({'role': 'user', 'content': f'turn {n}'} for n in range(3)))
    return str(tmp_path / 'state.db')


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
