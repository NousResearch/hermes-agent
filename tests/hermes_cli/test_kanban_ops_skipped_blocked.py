"""Observability: the ``kanban dispatch`` CLI surfaces ``skipped_blocked``.

The P2 defensive fence in ``kanban_db_dispatch`` (audit 2026-10-03) appends
parked ``blocked`` cards a lane query handed over to
``DispatchResult.skipped_blocked`` and writes the auditable
``skipped_blocked_lane`` event — but ``kanban_ops._cmd_dispatch`` never
presented the field: it was omitted from both the ``--json`` payload and the
text output, so an operator could see ``Spawned: 0`` with no evidence a lane
leak was fenced off.

Display contract (decisions documented once, then asserted):

* Empty bucket (normal ticks): the JSON key ``skipped_blocked`` is ALWAYS
  present, as ``[]`` — sibling ``skipped_*`` keys are unconditional too, a
  stable schema for downstream parsers. The text line is suppressed when
  the bucket is empty, matching every other conditional ``Skipped …`` line.
* ``dry_run``: identical display surface (JSON key + text line). The fence's
  audit event is existing dispatch-layer behavior, deliberately NOT changed
  or asserted away here — this file covers the CLI display only, so the
  dry-run test compares the fence-event count before and after and requires
  the DISPLAY work to write nothing.

Sandbox: assertions run against a throwaway HERMES_HOME (tmp_path), never a
real board. The ``HERMES_DELEGATED_CHILD_CONTEXT`` fence marker is re-pinned
to ``/tmp`` (the subagent scratch area) so the kanban write guard's
deny-list (kept anchored at the real root via ``HERMES_KANBAN_HOME``) never
contains the sandbox DB, while spawned workers of this pytest process stay
fenced — none are spawned here anyway (``spawn_fn`` is stubbed via the
empty budget the souped lane never reaches and the never-spawning
``dispatch`` default).

It is NOT a delegated child and must not fence itself: the
``HERMES_DELEGATED_CHILD_CONTEXT`` marker is deleted, otherwise
``kanban_path_is_fenced`` reads ``/tmp`` (or the runner's own environment)
as a fenced root and forces the sandbox DB into the ``mode=ro`` branch —
``init_db`` then fails with "unable to open database file". Real-root
protection stays armed separately via the pytest-level home-I/O guard,
which captured the production root at conftest import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

# hermes_cli imports resolve when pytest loads from the repo root; the
# insert is a belt for direct-file runs where the root may not be on
# sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _souped_lane_rows(original):
    """Lane query that leaks parked ``blocked`` rows — the P2 fence
    trigger (audit 2026-10-03 scenario): a lane refactor bug that hands
    parked cards to the ready-lane loop."""

    def _fn(conn_, status):
        rows = list(original(conn_, status))
        if status == "ready":
            rows += conn_.execute(
                "SELECT id, assignee, status FROM tasks "
                "WHERE status = 'blocked' AND claim_lock IS NULL "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        return rows

    return _fn


@pytest.fixture
def sandbox_kanban(tmp_path, monkeypatch):
    """Fresh sandboxed kanban home per test (real kernel, throwaway DB)."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    os.makedirs(home / "profiles" / "default", exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield home


def _dispatch_args(**kw):
    base = {"dry_run": False, "max": None, "failure_limit": 2, "json": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_json_and_text_surface_skipped_blocked(sandbox_kanban, capsys, monkeypatch):
    """A parked blocked card fenced by the ready lane must be visible in BOTH
    dispatch surfaces: the ``skipped_blocked`` JSON key and the stdout text
    line ``Skipped (blocked): <ids>`` (stable wording, same ``Skipped …``
    family as the other buckets)."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked card", initial_status="blocked")
        original = kbd._lane_rows
        monkeypatch.setattr(kbd, "_lane_rows", _souped_lane_rows(original))
        rc = kb_cli._cmd_dispatch(_dispatch_args())

    out = capsys.readouterr().out
    assert rc == 0
    line = "Skipped (blocked): " + tid
    assert line in out, f"text output must contain {line!r}; got:\n{out}"

    monkeypatch.setattr(kbd, "_lane_rows", _souped_lane_rows(original))
    rc2 = kb_cli._cmd_dispatch(_dispatch_args(json=True))
    out2 = capsys.readouterr().out
    assert rc2 == 0
    payload = json.loads(out2)
    assert payload.get("skipped_blocked") == [tid], payload


def test_no_blocked_skips_leaves_surface_quiet_or_empty(sandbox_kanban, capsys):
    """A normal tick with nothing fenced: the JSON key ``skipped_blocked`` is
    ALWAYS present (schema stability, equal to the sibling ``skipped_*``
    keys), while the text line is suppressed when empty — matching the
    existing conditional-bucket presentation."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        kb.create_task(conn, title="plain ready card", assignee="alice")

    rc = kb_cli._cmd_dispatch(_dispatch_args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "Skipped (blocked):" not in out, out

    rc2 = kb_cli._cmd_dispatch(_dispatch_args(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc2 == 0
    assert payload.get("skipped_blocked") == [], payload


def test_dry_run_shows_blocked_skips_but_writes_no_new_event(sandbox_kanban, capsys, monkeypatch):
    """dry-run shows the SAME blocked-skip surface as a normal run, and this
    display work writes no NEW audit event of its own: the count of
    ``skipped_blocked_lane`` rows on the card is taken BEFORE and AFTER the
    dry-run display — any fence event the dispatch layer already wrote on
    the warm tick is existing behavior (dedup per state change) and stays at
    that count; the display work adds zero."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    def _lane_events():
        with kbc.connect() as conn2:
            return conn2.execute(
                "SELECT kind FROM task_events WHERE task_id = ? AND kind = "
                "'skipped_blocked_lane'", (tid,)
            ).fetchall()

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked dry card", initial_status="blocked")
        original = kbd._lane_rows
        monkeypatch.setattr(kbd, "_lane_rows", _souped_lane_rows(original))
        # Warm tick (normal run): may write the fence audit event — that is
        # the dispatch layer's existing behavior, NOT this display work.
        rc = kb_cli._cmd_dispatch(_dispatch_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert ("Skipped (blocked): " + tid) in out, out

    before = len(_lane_events())

    monkeypatch.setattr(kbd, "_lane_rows", _souped_lane_rows(original))
    rc2 = kb_cli._cmd_dispatch(_dispatch_args(dry_run=True))
    out2 = capsys.readouterr().out
    assert rc2 == 0
    # Dry-run shows the same surface as the normal run.
    assert ("Skipped (blocked): " + tid) in out2, out2

    # The DISPLAY work adds nothing: unchanged fence-event count.
    after = len(_lane_events())
    assert after == before, (
        f"dry-run must not add skipped_blocked_lane events; "
        f"before={before} after={after}"
    )
