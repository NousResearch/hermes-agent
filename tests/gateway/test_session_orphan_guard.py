"""A namespace change must not silently strand a profile's sessions.

Real incident (Box 1, 2026-09-16): ``gateway.multiplex_profiles`` flipped to default-on in an
upstream update. Session keys are namespaced ``agent:<profile>:``  — ``agent:main`` for the
default profile, ``agent:<name>`` for a named one — so every existing key computed a DIFFERENT
key after the flip. 246 sessions carrying 9,764 messages stopped resolving. Nothing warned; each
chat simply opened a new empty session, and the operator's "please continue" had nothing to
continue from.

``rekey_profile_state`` already exists and fixes this exactly, but it only runs on an EXPLICIT
profile rename. The gap is the un-announced namespace change: an update, a config default flip,
a migration. This module is the detector for that gap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _make_db(path: Path, rows: list[tuple[str, str]]) -> None:
    """Minimal sessions table: (id, session_key)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, session_key TEXT)")
    conn.executemany("INSERT INTO sessions (id, session_key) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def test_orphaned_namespace_is_detected(tmp_path: Path) -> None:
    """Sessions under a namespace no live profile claims must be reported, not ignored."""
    from gateway.session_orphan_guard import find_orphaned_namespaces

    db = tmp_path / "state.db"
    _make_db(db, [
        ("s1", "agent:main:buzz:dm:chan:thread"),
        ("s2", "agent:main:buzz:dm:chan2:thread2"),
        ("s3", "agent:wik:buzz:dm:chan:thread"),
    ])

    # Only the 'wik' profile exists now — 'main'/default no longer claims anything.
    orphans = find_orphaned_namespaces(db, live_profiles={"wik"})

    assert orphans == {"default": 2}, orphans


def test_no_orphans_when_every_namespace_is_claimed(tmp_path: Path) -> None:
    """The healthy case must stay silent — no false alarm on every boot."""
    from gateway.session_orphan_guard import find_orphaned_namespaces

    db = tmp_path / "state.db"
    _make_db(db, [
        ("s1", "agent:wik:buzz:dm:chan:thread"),
        ("s2", "agent:main:telegram:dm:chan:thread"),
    ])

    orphans = find_orphaned_namespaces(db, live_profiles={"wik", "default"})

    assert orphans == {}


def test_missing_or_unreadable_db_never_blocks_boot(tmp_path: Path) -> None:
    """A guard that can crash the gateway is worse than the bug it detects."""
    from gateway.session_orphan_guard import find_orphaned_namespaces

    assert find_orphaned_namespaces(tmp_path / "absent.db", live_profiles={"wik"}) == {}

    junk = tmp_path / "junk.db"
    junk.write_text("not a database", encoding="utf-8")
    assert find_orphaned_namespaces(junk, live_profiles={"wik"}) == {}
