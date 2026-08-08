"""Tests for the extracted spawnability probes.

``has_spawnable_ready`` / ``has_spawnable_review`` moved verbatim from
``hermes_cli/kanban_db.py`` (region R4, cluster c12) into
``hermes_cli/spawnable_probe.py`` as part of the kanban god-file kill
(#78632). The god file re-exports both names, so this file also pins the
identity seam: ``kanban_db.X is spawnable_probe.X``.

The probes query only ``tasks.assignee / status / claim_lock``, so the tests
use a minimal in-memory table carrying exactly those columns.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db, profiles, spawnable_probe


def _make_conn(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            assignee   TEXT,
            status     TEXT NOT NULL,
            claim_lock TEXT
        )
        """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, claim_lock) "
            "VALUES (?, ?, ?, ?, ?)",
            row,
        )
    return conn


# ---------------------------------------------------------------------------
# Identity seam (god-file re-export)
# ---------------------------------------------------------------------------

def test_godfile_reexports_probe_functions_by_identity():
    """The shim in kanban_db.py must re-export the real function objects.

    Consumers call ``kanban_db.has_spawnable_ready`` through the module
    attribute; a copy or wrapper would silently diverge from what
    ``spawnable_probe`` provides. Object identity guards double-definition.
    """
    assert kanban_db.has_spawnable_ready is spawnable_probe.has_spawnable_ready
    assert kanban_db.has_spawnable_review is spawnable_probe.has_spawnable_review


# ---------------------------------------------------------------------------
# has_spawnable_ready
# ---------------------------------------------------------------------------

def test_no_rows_returns_false(monkeypatch):
    conn = _make_conn([])
    assert spawnable_probe.has_spawnable_ready(conn) is False


def test_ready_row_with_existing_profile_returns_true(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "ready", None)])
    assert spawnable_probe.has_spawnable_ready(conn) is True


def test_ready_row_with_nonexistent_profile_returns_false(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    conn = _make_conn([("t1", "task", "alice", "ready", None)])
    assert spawnable_probe.has_spawnable_ready(conn) is False


def test_only_claimed_rows_are_filtered_out(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "ready", "claim-1")])
    # claim_lock IS NULL filter drops the only row -> no spawnable work.
    assert spawnable_probe.has_spawnable_ready(conn) is False


def test_mixed_claimed_and_unclaimed_picks_unclaimed(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn(
        [
            ("t1", "task", "alice", "ready", "claim-1"),
            ("t2", "task", "bob", "ready", None),
        ]
    )
    assert spawnable_probe.has_spawnable_ready(conn) is True


def test_distinct_assignees_dedup_before_profile_check(monkeypatch):
    # Two ready tasks for the same nonexistent assignee: DISTINCT collapses
    # to one row; if any OTHER assignee is spawnable the probe still fires.
    calls: list[str] = []

    def fake_exists(name: str) -> bool:
        calls.append(name)
        return name == "bob"

    monkeypatch.setattr(profiles, "profile_exists", fake_exists)
    conn = _make_conn(
        [
            ("t1", "task", "alice", "ready", None),
            ("t2", "task", "alice", "ready", None),
            ("t3", "task", "bob", "ready", None),
        ]
    )
    assert spawnable_probe.has_spawnable_ready(conn) is True
    assert "alice" in calls and "bob" in calls


def test_profile_exists_import_failure_falls_back_to_true(monkeypatch):
    """Partial install: hermes_cli.profiles unimportable -> assume spawnable.

    The probe wraps the lazy ``from hermes_cli.profiles import profile_exists``
    in try/except and returns True on failure (legacy behavior). Model a
    genuinely missing module by halting the import via a ``None`` entry in
    ``sys.modules``.
    """
    import sys

    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", None)
    conn = _make_conn([("t1", "task", "alice", "ready", None)])
    assert spawnable_probe.has_spawnable_ready(conn) is True


def test_non_ready_status_is_ignored(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "review", None)])
    assert spawnable_probe.has_spawnable_ready(conn) is False


# ---------------------------------------------------------------------------
# has_spawnable_review
# ---------------------------------------------------------------------------

def test_review_row_with_existing_profile_returns_true(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "review", None)])
    assert spawnable_probe.has_spawnable_review(conn) is True


def test_review_no_rows_returns_false(monkeypatch):
    conn = _make_conn([])
    assert spawnable_probe.has_spawnable_review(conn) is False


def test_review_claimed_row_is_filtered_out(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "review", "claim-1")])
    assert spawnable_probe.has_spawnable_review(conn) is False


def test_review_ready_row_is_ignored(monkeypatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    conn = _make_conn([("t1", "task", "alice", "ready", None)])
    assert spawnable_probe.has_spawnable_review(conn) is False
