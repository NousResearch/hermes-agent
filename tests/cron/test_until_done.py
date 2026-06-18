"""
Tests for cron/until_done.py — Band 2 work-based loop primitive.

These tests exercise the parser, every checker (kanban_idle, kanban_empty,
kanban_done, files_match, always_done), the custom-status filter semantics,
and the 1440-attempt cap behavior. Kanban checkers use a temporary sqlite
file (no real ~/.hermes/kanban.db dependency) so the suite is hermetic and
fast.

Run:
    pytest tests/cron/test_until_done.py -v
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

# Repo-root on sys.path so `from cron.until_done import ...` works without
# a wheel install. Matches the convention in tests/cron/test_cron_prompt_injection_skill.py.
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cron import until_done as ud  # noqa: E402


# ── Parser ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, kind, extras",
    [
        ("always_done", "always_done", {}),
        ("kanban_idle", "kanban_idle", {}),
        ("kanban_idle:workspace=clinic-protocols", "kanban_idle",
         {"workspace": "clinic-protocols"}),
        ("kanban_idle:workspace=epic-1-embed,status=todo", "kanban_idle",
         {"workspace": "epic-1-embed", "status": "todo"}),
        ("kanban_idle:status=triage|ready", "kanban_idle",
         {"status": "triage|ready"}),
        ("kanban_empty", "kanban_empty", {}),
        ("kanban_empty:workspace=scratch", "kanban_empty",
         {"workspace": "scratch"}),
        ("kanban_done:id=t_abc", "kanban_done", {"id": "t_abc"}),
        ("kanban_done:workspace=scratch", "kanban_done",
         {"workspace": "scratch"}),
        ("files_match:glob=*.md,mtime_gt=1000.0", "files_match",
         {"glob": "*.md", "mtime_gt": "1000.0"}),
    ],
)
def test_parse_criteria_valid(text, kind, extras):
    parsed = ud.parse_criteria(text)
    assert parsed["kind"] == kind
    for k, v in extras.items():
        assert parsed.get(k) == v


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "kanban_idle:",
        "kanban_done:foo=bar",
        "files_match:mtime_gt=1",
        "nonsense_criterion",
        "kanban_idle:noequals",
    ],
)
def test_parse_criteria_rejects_invalid(text):
    with pytest.raises(ValueError):
        ud.parse_criteria(text)


# ── Module-level invariants ────────────────────────────────────────────


def test_module_constants():
    assert ud.DEFAULT_POLL_SECONDS == 60
    assert ud.MAX_ATTEMPTS == 1440
    # 1440 * 60s = 24h cap on continuous re-queueing.
    assert ud.MAX_ATTEMPTS * ud.DEFAULT_POLL_SECONDS == 86_400
    assert ud._STATUS_OPEN == {"triage", "todo", "ready", "running", "blocked"}
    assert ud._STATUS_TERMINAL == {"done", "rejected", "cancelled", "archived"}
    # Every advertised checker is registered.
    expected = {"kanban_idle", "kanban_empty", "kanban_done",
                "files_match", "always_done"}
    assert set(ud._CHECKERS.keys()) == expected


# ── Checkers, hermetic via monkeypatch on the kanban path ──────────────


@pytest.fixture
def kanban(monkeypatch, tmp_path):
    """Plant a fresh sqlite kanban.db, point ud._kanban_db_path at it."""
    db = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT,
            status TEXT,
            workspace_kind TEXT
        );
        """
    )
    rows = [
        # workspace, status
        ("clinic-protocols", "todo"),
        ("clinic-protocols", "blocked"),
        ("scratch", "done"),
        ("scratch", "todo"),
        ("epic-1", "running"),
    ]
    conn.executemany(
        "INSERT INTO tasks (id, title, status, workspace_kind) VALUES (?, ?, ?, ?)",
        [(f"t_{i}", f"task {i}", s, w) for i, (w, s) in enumerate(rows)],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(ud, "_kanban_db_path", lambda: str(db))
    return db


def test_kanban_idle_no_scope_has_open_tasks(kanban):
    # 4 open tasks total (todo+blocked+running)
    is_done, msg = ud.check_by_text("kanban_idle")
    assert is_done is False
    assert "4 open task(s)" in msg


def test_kanban_idle_scoped(kanban):
    # clinic-protocols: 2 open (todo+blocked)
    is_done, _ = ud.check_by_text("kanban_idle:workspace=clinic-protocols")
    assert is_done is False

    # epic-1: 1 open (running)
    is_done, _ = ud.check_by_text("kanban_idle:workspace=epic-1")
    assert is_done is False


def test_kanban_idle_custom_status_filter(kanban):
    # NB: the `status=` filter is the "what counts as open" set, not a
    # status we want to wait for. Filtering to `running` is the only
    # status in our fixture that has 0+ matches in some workspaces, so
    # the query is non-trivial.
    # Filter to only "running" → epic-1 has 1, others have 0 → not idle
    is_done, _ = ud.check_by_text("kanban_idle:status=running")
    assert is_done is False

    # Filter to a status that has zero matches in any workspace → idle.
    # ("cancelled" has 0 rows in the fixture, so open_set={'cancelled'}
    # yields 0 open tasks → idle.)
    is_done, _ = ud.check_by_text("kanban_idle:status=cancelled")
    assert is_done is True

    # Pipe-separated, scoped to scratch: scratch has 1 todo, 1 done
    # → open_set={todo, blocked} matches 1 row → not idle
    is_done, _ = ud.check_by_text(
        "kanban_idle:workspace=scratch,status=todo|blocked"
    )
    assert is_done is False

    # Same workspace, different filter: open_set={running} → 0 rows
    # in scratch → idle
    is_done, _ = ud.check_by_text(
        "kanban_idle:workspace=scratch,status=running"
    )
    assert is_done is True


def test_kanban_empty_vacuous_on_unknown_workspace(kanban):
    is_done, msg = ud.check_by_text("kanban_empty:workspace=does-not-exist")
    assert is_done is True
    assert "0 task(s) total" in msg


def test_kanban_empty_false_when_tasks_exist(kanban):
    is_done, _ = ud.check_by_text("kanban_empty:workspace=clinic-protocols")
    assert is_done is False


def test_kanban_done_missing_task_id_is_vacuous(kanban):
    # Missing task is treated as done (vacuous truth). This is intentional:
    # if the task id has been deleted, we don't want the job to block forever.
    is_done, msg = ud.check_by_text("kanban_done:id=t_definitely_not_real")
    assert is_done is True
    assert "not found" in msg


def test_kanban_done_workspace_requires_all_terminal(kanban):
    # scratch has 1 todo, 1 done → not all terminal → not done
    is_done, msg = ud.check_by_text("kanban_done:workspace=scratch")
    assert is_done is False
    assert "1 open in scratch" in msg


def test_always_done_is_trivially_true():
    is_done, msg = ud.check_by_text("always_done")
    assert is_done is True
    assert msg == "always_done"


def test_files_match_no_files_is_true(tmp_path, monkeypatch):
    # No files matching → vacuous done (nothing to wait on).
    is_done, msg = ud.check_by_text(
        f"files_match:glob={tmp_path}/_no_such_glob_*.xyz,mtime_gt=0"
    )
    assert is_done is True
    assert "no files match" in msg


def test_files_match_threshold_in_future_marks_all_stale(tmp_path):
    # Plant one file in tmp_path and use a mtime threshold far in the future.
    f = tmp_path / "x.md"
    f.write_text("hello")
    is_done, msg = ud.check_by_text(
        f"files_match:glob={tmp_path}/*.md,mtime_gt=99999999999.0"
    )
    assert is_done is False
    assert "1 matched" in msg and "1 stale" in msg


def test_check_by_text_catches_parse_errors():
    # check_by_text should never raise on bad input — it returns (False, msg).
    is_done, msg = ud.check_by_text("not a real criteria")
    assert is_done is False
    assert "parse error" in msg


# ── Decision loop semantics (re-queue vs done vs stuck) ───────────────


def test_checker_returns_done_when_criteria_satisfied(kanban):
    # Filter clinic-protocols to status=running only (0 such tasks) → idle.
    is_done, _ = ud.check_by_text(
        "kanban_idle:workspace=clinic-protocols,status=running"
    )
    assert is_done is True


def test_max_attempts_constant_supports_24h_polling():
    # 60s * 1440 = 24 hours of continuous polling. If you change this,
    # update the cap rationale comment in cron/until_done.py.
    assert ud.MAX_ATTEMPTS * ud.DEFAULT_POLL_SECONDS == 86_400
