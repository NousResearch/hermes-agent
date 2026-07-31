"""F10: workspace-filtered `hermes sessions list` paginates correctly.

The workspace key is derived (git_repo_root / cwd), not a DB column, so
the filter runs in Python. The old code applied the SQL offset+limit
BEFORE the filter, so a workspace whose matches fell outside the fetched
window returned nothing (or the wrong page) even though matching sessions
existed. The fix fetches a superset WITHOUT the SQL offset when a filter
is active, filters, then slices to [offset, offset+limit).
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed(home: Path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    conn = db._conn
    for i, (sid, repo, started) in enumerate(
        [
            ("w1_a", "/work/repo1", 1_000.0),
            ("w2_a", "/work/repo2", 1_500.0),
            ("w1_b", "/work/repo1", 2_000.0),
            ("w2_b", "/work/repo2", 2_500.0),
            ("w1_c", "/work/repo1", 3_000.0),
            ("w1_d", "/work/repo1", 4_000.0),
        ]
    ):
        db.create_session(sid, "cli")
        db.set_session_title(sid, f"Session {sid}")
        db.append_message(sid, "user", f"opener {sid}", timestamp=started)
        conn.execute(
            "UPDATE sessions SET started_at = ?, git_repo_root = ? WHERE id = ?",
            (started, repo, sid),
        )
    conn.commit()
    db.close()


def _run(home: Path, *argv) -> str:
    env = {**os.environ, "HERMES_HOME": str(home), "TZ": "UTC"}
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "sessions", "list", *argv],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=120,
    )
    return result.stdout


def test_workspace_matches_beyond_recent_n_still_appear(tmp_path):
    """All 4 repo1 matches appear with -l 4, even though the 4 most-recent
    sessions overall are a mix of repos. Old code applied the SQL limit
    first (fetch 4 -> filter -> 3, silently dropping w1_a); the fix fetches
    a superset, filters, then slices."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    _seed(home)
    out = _run(home, "--workspace", "repo1", "-l", "4")
    assert "w1_d" in out and "w1_c" in out
    assert "w1_b" in out and "w1_a" in out
    assert "w2_a" not in out and "w2_b" not in out


def test_workspace_filter_respects_limit(tmp_path):
    """The limit still bounds the result after the filter: -l 2 with 4
    repo1 matches shows the 2 most recent repo1 sessions only."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    _seed(home)
    out = _run(home, "--workspace", "repo1", "-l", "2")
    assert "w1_d" in out and "w1_c" in out
    assert "w1_b" not in out and "w1_a" not in out
    assert "w2_a" not in out and "w2_b" not in out


def test_unfiltered_limit_still_bounds_rows(tmp_path):
    """Without a workspace filter the SQL limit applies directly."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    _seed(home)
    out = _run(home, "-l", "2")
    assert "w1_d" in out and "w1_c" in out
    assert "w1_a" not in out and "w2_a" not in out


def test_workspace_filter_page_two_slices_correct_window(tmp_path):
    """Page 2 with a workspace filter: superset fetch WITHOUT the SQL
    offset, filter, then slice to [offset, offset+limit) — the exact F10
    regression. Old code applied offset+limit in SQL first (rows 2..4 of
    the global list), then filtered to repo1, returning the wrong window."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    _seed(home)
    out = _run(home, "--workspace", "repo1", "-l", "2", "2")
    assert "w1_b" in out and "w1_a" in out
    assert "w1_d" not in out and "w1_c" not in out
    assert "w2_a" not in out and "w2_b" not in out


def test_page_two_shows_next_window_unfiltered(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    _seed(home)
    out = _run(home, "-l", "2", "2")
    assert "w1_b" in out and "w2_b" in out
    assert "w1_d" not in out and "w1_c" not in out
