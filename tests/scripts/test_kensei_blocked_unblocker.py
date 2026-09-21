"""Invariants for the kensei-blocked-unblocker pre-check.

The cron pre-check at ``scripts/kensei-blocked-unblocker.py`` runs as a
no_agent=True script that decides whether the LLM agent that follows it
gets woken up — so a bug here either wakes the agent on garbage or,
worse, makes it act on tasks that are already human-owned.

These tests pin three contracts against the REPOSITORY source (the
deployed copy under ``HERMES_HOME/scripts/`` is a build artifact of this
file; testing the repo source keeps CI reproducible from a checkout):

  1. **Unit tests for ``_classify_row``** — the row-classification seam
     that decides per-row whether to skip a human-owned task, skip a
     too-young task, stop at MAX_TASKS, or include the row. The module
     is import-safe: importing it must not scan boards or ``sys.exit``.

  2. **Integration tests** — drive the repo script as a subprocess
     against a fake HERMES_HOME and assert the current structured
     output contract:
       stdout  ``{"wakeAgent": false}`` when nothing is actionable, else
               a JSON object with ``routine_unblock`` / ``escalate`` /
               ``human_owned_skipped`` (+ scan metadata);
       stderr  one JSON diagnostic record;
       exit    0 in both cases. Stuck-task escalations are deduped via
               ``state/blocked-unblocker-dedup.json``.

  3. **status='blocked' filter** — the SQL pre-filter excludes the
     ``decision-needed`` status (a terminal human-gated state set by
     ``block_task``), so the script never even sees decision-needed
     tasks in its row loop.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

# The repository source of the cron pre-check. The scheduler runs the
# deployed copy at HERMES_HOME/scripts/kensei-blocked-unblocker.py, which
# is synced from this file at deploy time; tests exercise the repo source
# so a source-tree CI run does not depend on mutable per-host state.
REPO_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "kensei-blocked-unblocker.py"
)


def _load_module(monkeypatch, fake_home: Path):
    """Import the repo script with HERMES_HOME pointed at fake_home.

    The module reads HERMES_HOME at import for its constants but must not
    do any scanning work at import time (that lives in ``main()``).
    """
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    spec = importlib.util.spec_from_file_location(
        "kensei_blocked_unblocker_under_test", str(REPO_SCRIPT)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    """An empty HERMES_HOME pointing at tmp_path. No real DBs are touched."""
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


@pytest.fixture
def module(fake_home, monkeypatch):
    """Load the repo script with HERMES_HOME pointed at fake_home."""
    return _load_module(monkeypatch, fake_home)


def _row(**overrides):
    """A minimal blocked-task row matching the script's SELECT shape."""
    base = {
        "id": "t_test",
        "title": "Test task",
        "assignee": "tester",
        "priority": 0,
        "status_reason": "",
        "created_at": 0,
        "updated_at": 0,
        "consecutive_failures": 0,
        "max_retries": None,
        "escalation_target": None,
        "block_kind": None,
        "body_snippet": "",
    }
    base.update(overrides)
    return base


# ── Import safety ────────────────────────────────────────────────────────────


def test_module_import_is_side_effect_free(module, fake_home):
    """Importing the script must not scan boards, write state, or exit.

    Regression: the collector used to run at module top level and call
    ``sys.exit(0)``, which made it untestable as a unit.
    """
    assert callable(module.main)
    assert callable(module._classify_row)
    # No state dir / dedup file was created by the import.
    assert not (fake_home / "state").exists()


# ── Unit tests for _classify_row ────────────────────────────────────────────


def test_classify_skip_human_owned_when_escalation_target_set(module):
    """A blocked row with escalation_target set must be classified as
    skip_human_owned — the cron must not surface it for auto-unblock or
    escalation.
    """
    row = _row(escalation_target="sahil", block_kind="needs_input")
    action, payload = module._classify_row(row, now=1_000_000, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_human_owned"
    assert payload == {
        "escalation_target": "sahil",
        "block_kind": "needs_input",
    }


def test_classify_skip_human_owned_with_capability_block_kind(module):
    """The skip applies regardless of block_kind — a stakeholder
    escalation overrides any auto-recovery intent, even for capability
    blocks where auto-repair might otherwise be appropriate.
    """
    row = _row(escalation_target="sahil", block_kind="capability")
    action, _ = module._classify_row(row, now=1_000_000, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_human_owned"


def test_classify_skip_human_owned_with_empty_block_kind(module):
    """Older rows may have escalation_target set but block_kind NULL
    (legacy data). Skip must still fire.
    """
    row = _row(escalation_target="sahil", block_kind=None)
    action, payload = module._classify_row(row, now=1_000_000, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_human_owned"
    assert payload["block_kind"] == ""


def test_classify_include_when_escalation_target_null_and_old_enough(module):
    """A blocked row with escalation_target NULL and updated_at older
    than ESCALATION_AGE must be classified as include.
    """
    now = 1_000_000
    old = now - module.ESCALATION_AGE - 1
    row = _row(escalation_target=None, updated_at=old)
    action, _ = module._classify_row(row, now=now, max_tasks=50, blocked_tasks_len=0)
    assert action == "include"


def test_classify_skip_too_young_when_recently_updated(module):
    """A row updated more recently than ESCALATION_AGE is filtered out
    by the age check before any human-owned consideration.
    """
    now = 1_000_000
    recent = now - 10  # 10 seconds old
    row = _row(escalation_target=None, updated_at=recent)
    action, _ = module._classify_row(row, now=now, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_too_young"


def test_classify_stop_when_max_tasks_reached(module):
    """The MAX_TASKS cap fires before any other check, so a human-owned
    row would not even be classified once the cap is hit (caller breaks
    out of the per-board loop).
    """
    row = _row(escalation_target=None, updated_at=0)
    action, _ = module._classify_row(row, now=1_000_000, max_tasks=10, blocked_tasks_len=10)
    assert action == "stop"


def test_classify_skip_when_timestamps_missing(module):
    """Defensive: if both updated_at and created_at are NULL, the row is
    dropped (no basis to compute age). Mirrors the pre-existing
    behaviour in the script.
    """
    row = _row(escalation_target=None, updated_at=None, created_at=None)
    action, _ = module._classify_row(row, now=1_000_000, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_too_young"


def test_classify_does_not_explicitly_check_status(module):
    """The status='blocked' filter lives in the SQL, not in
    _classify_row. _classify_row only sees rows that already passed the
    filter; this test pins that _classify_row is intentionally
    status-agnostic and the SQL is what enforces the decision-needed
    exclusion.
    """
    row = _row(
        id="t_dec",
        escalation_target="sahil",  # the SQL filter would not show this
        block_kind="needs_input",
    )
    action, _ = module._classify_row(row, now=1_000_000, max_tasks=50, blocked_tasks_len=0)
    assert action == "skip_human_owned"


# ── Integration tests: subprocess against a fake HERMES_HOME ────────────────


def _populate_board_db(path: Path, tasks: list[dict]) -> None:
    """Create a minimal tasks schema with the columns the script's
    SELECT reads, then insert the given tasks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                title TEXT,
                assignee TEXT,
                priority INTEGER,
                status TEXT,
                status_reason TEXT,
                body TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                consecutive_failures INTEGER,
                max_retries INTEGER,
                escalation_target TEXT,
                block_kind TEXT
            )
            """
        )
        for t in tasks:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, title, assignee, priority, status, status_reason,
                    body, created_at, updated_at, consecutive_failures,
                    max_retries, escalation_target, block_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    t["id"],
                    t["title"],
                    t.get("assignee", "tester"),
                    t.get("priority", 0),
                    t["status"],
                    t.get("status_reason", ""),
                    t.get("body", ""),
                    t.get("created_at", 0),
                    t.get("updated_at", 0),
                    t.get("consecutive_failures", 0),
                    t.get("max_retries"),
                    t.get("escalation_target"),
                    t.get("block_kind"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _run_script(hermes_home: Path):
    """Run the repo script as a subprocess against hermes_home."""
    env = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "KENSEI_UNBLOCKER_AGE": "3600",
        "KENSEI_UNBLOCKER_MAX": "50",
        "KENSEI_UNBLOCKER_STUCK_FAILURES": "3",
        "KENSEI_UNBLOCKER_DEDUP_HOURS": "12",
        "PYTHONPATH": "",  # avoid leaking the real project root
    }
    return subprocess.run(
        [sys.executable, str(REPO_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _diag_record(stderr: str) -> dict:
    """Parse the single JSON diagnostic record the script logs to stderr."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]
    diags = [p for p in parsed if "blocked_found" in p]
    assert len(diags) == 1, f"expected exactly one diag record, got: {parsed!r}"
    return diags[0]


def test_integration_empty_board_does_not_wake_agent(fake_home):
    """No boards at all → {"wakeAgent": false}, exit 0, diag on stderr."""
    proc = _run_script(fake_home)
    assert proc.returncode == 0, (
        f"script failed: stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )
    assert json.loads(proc.stdout) == {"wakeAgent": False}
    diag = _diag_record(proc.stderr)
    assert diag["boards_scanned"] == 0
    assert diag["blocked_found"] == 0
    assert diag["errors"] == []


def test_integration_routes_routine_and_human_owned(fake_home):
    """A board with a human-owned and an auto-recoverable blocked task:
    the auto one goes to routine_unblock, the human-owned one goes to
    human_owned_skipped (never routine/escalate), and non-blocked rows
    (decision-needed, done) are invisible via the SQL status filter.
    """
    db_path = fake_home / "kanban" / "boards" / "test_board" / "kanban.db"
    now = 1_700_000_000
    old_updated = now - 7200  # well past ESCALATION_AGE (wall clock >> now)

    _populate_board_db(
        db_path,
        [
            {
                "id": "t_human",
                "title": "Stakeholder decision needed",
                "status": "blocked",
                "escalation_target": "sahil",
                "block_kind": "needs_input",
                "updated_at": old_updated,
            },
            {
                "id": "t_auto",
                "title": "Routine auto-recoverable",
                "status": "blocked",
                "escalation_target": None,
                "block_kind": "transient",
                "updated_at": old_updated,
            },
            {
                "id": "t_decision",
                "title": "Terminal decision-needed",
                "status": "decision-needed",
                "escalation_target": "sahil",
                "block_kind": "needs_input",
                "updated_at": old_updated,
            },
            {
                "id": "t_done",
                "title": "Already done",
                "status": "done",
                "escalation_target": None,
                "updated_at": old_updated,
            },
        ],
    )

    proc = _run_script(fake_home)
    assert proc.returncode == 0, (
        f"script failed: stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )

    out = json.loads(proc.stdout)
    assert out["wakeAgent"] is True
    assert out["total_blocked"] == 2  # t_human + t_auto; SQL filtered the rest

    routine_ids = {t["id"] for t in out["routine_unblock"]}
    escalate_ids = {t["id"] for t in out["escalate"]}
    human_ids = {t["id"] for t in out["human_owned_skipped"]}
    assert routine_ids == {"t_auto"}
    assert escalate_ids == set()
    assert human_ids == {"t_human"}

    # The human-owned entry carries its routing fields.
    human = out["human_owned_skipped"][0]
    assert human["human_owned"] is True
    assert human["escalation_target"] == "sahil"
    assert human["block_kind"] == "needs_input"
    assert human["board"] == "test_board"

    # The routine entry matches the current per-task schema (no
    # escalation_target/block_kind keys on auto tasks).
    auto = out["routine_unblock"][0]
    assert auto["human_owned"] is False
    assert "escalation_target" not in auto
    assert "block_kind" not in auto

    # decision-needed / done rows never surfaced anywhere.
    everywhere = routine_ids | escalate_ids | human_ids
    assert "t_decision" not in everywhere
    assert "t_done" not in everywhere

    diag = _diag_record(proc.stderr)
    assert diag["boards_scanned"] == 1
    assert diag["blocked_found"] == 2
    assert diag["truncated"] is False


def test_integration_stuck_task_escalates_with_dedup(fake_home):
    """A task at STUCK_FAILURES consecutive failures goes to escalate
    and is recorded in the dedup state file; an immediate second run
    does not re-escalate it (wakeAgent false within DEDUP_HOURS).
    """
    db_path = fake_home / "kanban" / "boards" / "test_board" / "kanban.db"
    now = 1_700_000_000
    _populate_board_db(
        db_path,
        [
            {
                "id": "t_stuck",
                "title": "Stuck repeated failure",
                "status": "blocked",
                "escalation_target": None,
                "block_kind": "transient",
                "updated_at": now - 7200,
                "consecutive_failures": 3,
            },
        ],
    )

    first = _run_script(fake_home)
    assert first.returncode == 0
    out = json.loads(first.stdout)
    assert out["wakeAgent"] is True
    assert [t["id"] for t in out["escalate"]] == ["t_stuck"]
    assert out["routine_unblock"] == []

    state_file = fake_home / "state" / "blocked-unblocker-dedup.json"
    assert state_file.exists()
    assert "t_stuck" in json.loads(state_file.read_text())

    second = _run_script(fake_home)
    assert second.returncode == 0
    assert json.loads(second.stdout) == {"wakeAgent": False}


def test_integration_human_owned_only_does_not_wake_agent(fake_home):
    """A board whose only blocked task is human-owned must not wake the
    agent: human-owned tasks are reported for visibility only when a
    wake happens anyway, never a reason to wake by themselves.
    """
    db_path = fake_home / "kanban" / "boards" / "test_board" / "kanban.db"
    now = 1_700_000_000
    _populate_board_db(
        db_path,
        [
            {
                "id": "t_human",
                "title": "Stakeholder decision needed",
                "status": "blocked",
                "escalation_target": "sahil",
                "block_kind": "needs_input",
                "updated_at": now - 7200,
            },
        ],
    )

    proc = _run_script(fake_home)
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {"wakeAgent": False}
    diag = _diag_record(proc.stderr)
    assert diag["blocked_found"] == 1  # seen and counted, just not actionable
