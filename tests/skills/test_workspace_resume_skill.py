"""Behavioral contract tests for the workspace-resume skill's helper script.

Uses only stdlib + pytest + unittest.mock — no live network, no real
~/.hermes. Builds a disposable HERMES_HOME so the script's sqlite reads and
profile resolution match what a real session would produce.
"""
from __future__ import annotations

import runpy
import sqlite3
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "optional-skills" / "autonomous-ai-agents" / "workspace-resume" / "scripts" / "workspace_recall.py"


@pytest.fixture()
def mod():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """A HERMES_HOME laid out like the real one: state.db under it."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    return home


def _seed_sessions(db_path: Path, rows):
    """Insert session rows into a live-on-disk store the way hermes_state does."""
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, model TEXT, model_config TEXT,
            system_prompt TEXT, system_prompt_hash TEXT, parent_session_id TEXT,
            started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
            message_count INTEGER DEFAULT 0, cwd TEXT, git_branch TEXT,
            git_repo_root TEXT, last_activity_at REAL, profile_name TEXT,
            archived INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    for r in rows:
        con.execute(
            """
            INSERT INTO sessions
                (id, source, started_at, cwd, git_branch, git_repo_root,
                 last_activity_at, profile_name, archived)
            VALUES (?,?,?,?,?,?,?,?,?)
            """, r,
        )
    con.commit()
    con.close()


def _touch_repo(root: Path):
    (root / ".git").mkdir()


# --- profile resolution -------------------------------------------------


def test_default_profile_resolves_default_home(mod, fake_home):
    assert mod["_profile_name"]() == "default"
    assert mod["_state_db_path"]() == fake_home / "state.db"


def test_named_profile_prefers_profile_scoped_store(mod, fake_home, monkeypatch):
    prof = fake_home / "profiles" / "council-builder"
    prof.mkdir(parents=True)
    monkeypatch.setenv("HERMES_PROFILE", "council-builder")
    assert mod["_profile_name"]() == "council-builder"
    assert mod["_state_db_path"]() == prof / "state.db"


def test_named_profile_falls_back_to_default_when_uncreated(mod, fake_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "ghost-profile")
    assert mod["_state_db_path"]() == fake_home / "state.db"


# --- recall pass --------------------------------------------------------


def test_recall_returns_none_with_no_state_db(mod, fake_home):
    assert mod["recall_last_workspace"](fake_home / "state.db") is None


def test_recall_returns_none_with_empty_store(mod, fake_home):
    _seed_sessions(fake_home / "state.db", [])
    assert mod["recall_last_workspace"](fake_home / "state.db") is None


def test_recall_picks_repo_session_with_repo_root(mod, fake_home, tmp_path):
    repo = tmp_path / "my-repo"
    repo.mkdir()
    _touch_repo(repo)
    _seed_sessions(fake_home / "state.db", [
        ("s1", "cli", time.time() - 100, str(tmp_path), None, None, None, None, 0),
        ("s2", "cli", time.time() - 60, str(repo), "main", str(repo), None, None, 0),
    ])
    hit = mod["recall_last_workspace"](fake_home / "state.db")
    assert hit is not None
    assert hit["mode"] == "recall"
    assert hit["target"] == str(repo)
    assert hit["git_branch"] == "main"


def test_recall_skips_home_dir_sessions_without_git_context(mod, fake_home, tmp_path):
    """A session that ran in ~ must NOT recall the agent back to ~."""
    _seed_sessions(fake_home / "state.db", [
        ("s1", "cli", time.time() - 60, str(Path.home()), None, None, None, None, 0),
    ])
    assert mod["recall_last_workspace"](fake_home / "state.db") is None


def test_recall_skips_deleted_workspace(mod, fake_home, tmp_path):
    gone = tmp_path / "deleted-repo"
    gone.mkdir()
    _touch_repo(gone)
    _seed_sessions(fake_home / "state.db", [
        ("s1", "cli", time.time() - 60, str(gone), "main", str(gone), None, None, 0),
    ])
    # Really delete: remove .git then rmdir
    for p in sorted(gone.rglob("*"), reverse=True):
        p.rmdir()
    gone.rmdir()
    assert mod["recall_last_workspace"](fake_home / "state.db") is None


# --- rank pass ----------------------------------------------------------


def test_rank_orders_by_recency_then_dirty(fake_home, tmp_path, monkeypatch):
    """A repo committed yesterday beats a 6-day-old repo; a dirty 6-day-old
    repo beats a clean 6-day-old repo (tie-breakers resolve equal recency)."""
    from unittest.mock import patch

    a = tmp_path / "a-recent"; a.mkdir(); _touch_repo(a)
    b = tmp_path / "b-old-clean"; b.mkdir(); _touch_repo(b)
    c = tmp_path / "c-old-dirty"; c.mkdir(); _touch_repo(c)
    age_by_path = {str(a): 1.0, str(b): 6.0, str(c): 6.0}

    def fake_git(args, cwd):
        age_days = age_by_path[str(cwd)]
        if args[:2] == ["log", "-1"]:
            return 0, str(int(time.time() - age_days * 86400))
        if args[:2] == ["status", "--porcelain"]:
            return 0, (" M file.txt" if str(cwd) == str(c) else "")
        return 0, ""

    monkeypatch.setenv("CWD_ROOTS", str(tmp_path))
    # Load the script as a real importable module so mod._git is patchable at
    # the call site (runpy.run_path returns a plain dict — mutating it has no
    # effect on the script's actual globals).
    import importlib.util
    spec = importlib.util.spec_from_file_location("workspace_recall", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with patch.object(mod, "_git", side_effect=fake_git):
        ranked = mod.rank_candidates(fake_home / "state.db")

    paths = [r["path"] for r in ranked]
    assert paths == [str(a), str(c), str(b)]
    assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]
    assert ranked[1]["dirty"] is True and ranked[2]["dirty"] is False


def test_recency_score_steps(monkeypatch):
    # Load the function without touching git/sqlite
    g = runpy.run_path(str(SCRIPT))
    recency = g["_recency_score"]
    assert recency(None) == 0
    assert recency(0.2) == 4.0   # within 1d
    assert recency(2.0) == 3.0   # within 3d
    assert recency(5.0) == 2.0   # within 7d
    assert recency(15.0) == 1.0  # within 30d
    assert recency(120.0) == 0.0


def test_candidates_include_session_count_boost(mod, fake_home, tmp_path):
    repo = tmp_path / "worked-repo"
    repo.mkdir()
    _touch_repo(repo)
    _seed_sessions(fake_home / "state.db", [
        ("s1", "cli", time.time() - 10, str(repo), None, None, None, None, 0),
        ("s2", "cli", time.time() - 5, str(repo), None, str(repo), None, None, 0),
    ])
    counts = mod["_session_counts"](fake_home / "state.db")
    assert counts == {str(repo): 2}