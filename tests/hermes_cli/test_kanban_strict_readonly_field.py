"""Tests for the STRICT-READONLY Kanban worker capability field.

Regression matrix coverage:

S1  strict capability propagates objective -> task -> dispatcher env
    (data persistence component — companion dispatch test covers env)
S2  ordinary autonomous writable task NOT made strict purely by origin
    (task-default component — companion initiator test covers origin)
S3  ordinary Kanban task unchanged

The companion tests cover the remaining S-regressions:

  tests/hermes_cli/test_kanban_worker_strict_dispatch.py     S1-env S4 S15 S16 S18 S20 S21
  tests/hermes_cli/test_kanban_initiator_strict_propagation.py  S1-init S2 origin
  tests/tools/test_file_tools_strict_workspace_gate.py       S5-S14 S17 S19

These tests use ONLY stdlib + hermes internals; no live network calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hermes_state
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Schema (S1, S3)
# ---------------------------------------------------------------------------


def test_strict_readonly_column_present_in_fresh_schema(kanban_home):
    """S1+S3: fresh schema carries ``strict_readonly`` (default 0)."""
    with kb.connect_closing() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "strict_readonly" in cols


def test_strict_readonly_migration_idempotent(kanban_home):
    """Migration runs more than once without error or duplicate columns."""
    kb._migrate_add_optional_columns(_open_conn())  # noqa: SLF001 — internal API by design
    kb._migrate_add_optional_columns(_open_conn())
    with kb.connect_closing() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    # Exactly one strict_readonly column.
    strict_cols = [name for name in cols if name == "strict_readonly"]
    assert len(strict_cols) == 1


def test_strict_readonly_default_zero_on_existing_rows(kanban_home):
    """S3: ordinary Kanban tasks created before opt-in default to 0."""
    with kb.connect_closing() as conn:
        # Ordinary writable task (no strict_readonly kwarg).
        task_id = kb.create_task(
            conn,
            title="ordinary",
            created_by="user",
            initial_status="running",
        )
        row = conn.execute(
            "SELECT strict_readonly FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    assert row is not None and row["strict_readonly"] == 0


# ---------------------------------------------------------------------------
# Round-trip (S1, S3)
# ---------------------------------------------------------------------------


def test_strict_readonly_persists_and_round_trips(kanban_home):
    """S1: create_task(strict_readonly=True) persists 1 and reads back True."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="strict",
            created_by="user",
            strict_readonly=True,
            initial_status="running",
        )
        row = conn.execute(
            "SELECT strict_readonly FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    assert row is not None and row["strict_readonly"] == 1

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task.strict_readonly is True


def test_strict_readonly_explicit_false_persists(kanban_home):
    """Explicit False persists as 0 (default behavior preserved)."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="explicit-false",
            created_by="user",
            strict_readonly=False,
            initial_status="running",
        )
        task = kb.get_task(conn, task_id)
    assert task.strict_readonly is False


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_create_task_without_strict_readonly_kwarg_still_works(kanban_home):
    """S3: existing callers that don't pass strict_readonly continue to work."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="legacy-caller",
            created_by="user",
            goal_mode=True,  # exercise an adjacent kwarg too
            initial_status="running",
        )
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.strict_readonly is False
    assert task.goal_mode is True


def test_task_dataclass_field_default_false():
    """The dataclass default is False so legacy Task(...) constructions are safe."""
    # Default-only construction (no persistence round-trip needed).
    from dataclasses import fields as dc_fields
    f = next(f for f in dc_fields(kb.Task) if f.name == "strict_readonly")
    assert f.default is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_conn() -> sqlite3.Connection:
    """Open a connection to the current kanban DB (for migration runs)."""
    from hermes_cli import kanban_db as _kb
    conn = _kb.connect()
    return conn
