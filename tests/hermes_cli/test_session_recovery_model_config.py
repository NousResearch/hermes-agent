"""Invariant: session recovery neutralises unparseable ``sessions.model_config``.

``integrity_check`` never looks at column contents, so a source whose JSON was
truncated by the damage recovers "clean" and the recovered store then raises
``OperationalError: malformed JSON`` the first time a caller resumes a parent
session (``reopen_session`` rewrites reset-child markers with ``json_set``).
Both recovery lanes pass through ``_finalize_derived_metadata``, so the reset
belongs there rather than in either lane's copy loop.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_cli.session_recovery import recover_session_database
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session

_TRUNCATED_JSON = '{"model": "sonnet", "cw'


def _damaged_source(path: Path) -> None:
    """A structurally healthy DB whose child row carries truncated model_config JSON."""
    with open_persistence(profile_scope(home=path.parent)) as store:
        seed_session(store, "parent", "telegram", session_key="telegram:u:c")
        seed_messages(store, "parent", {"role": "user", "content": "hello"})
        store.sessions.end("parent", "session_reset")
        seed_session(store, "child", "telegram", session_key="telegram:u:c", parent_session_id="parent")
    raw = sqlite3.connect(str(path))
    try:
        raw.execute("UPDATE sessions SET model_config = ? WHERE id = 'child'", (_TRUNCATED_JSON,))
        raw.commit()
    finally:
        raw.close()


def test_recovery_resets_unparseable_model_config(tmp_path):
    source = tmp_path / "state.db"
    output = tmp_path / "recovered" / "state.db"
    output.parent.mkdir()
    _damaged_source(source)

    report = recover_session_database(source, output)

    assert report["verification"]["healthy"], report["verification"]["errors"]

    with open_persistence(profile_scope(home=output.parent)) as recovered:
        # The blob was unrecoverable; what matters is that resuming the parent still works
        # (on base this raises OperationalError: malformed JSON).
        recovered.sessions.reopen("parent")
