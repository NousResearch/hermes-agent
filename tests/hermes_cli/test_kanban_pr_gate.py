"""Tests for the PR merge-readiness gate in hermes_cli.kanban_db.

These tests mock ``_check_pr_status`` so no real ``gh`` or GitHub API calls are
made. They exercise both the DB-layer gate in ``complete_task`` and the tool
handlers for setting PR URLs and creating tasks with ``pr_url``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from hermes_cli import kanban_db as kb


PR_URL = "https://github.com/owner/repo/pull/42"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_ready_task(conn, title="task", assignee="test", pr_url=None):
    tid = kb.create_task(conn, title=title, assignee=assignee, pr_url=pr_url)
    return tid


def _open_task(conn, tid):
    """Fetch a task and fail loudly if missing."""
    task = kb.get_task(conn, tid)
    assert task is not None
    return task


def _complete(conn, tid, summary="done"):
    """Complete a task with a fresh in-memory claim/run like a worker would have."""
    kb.claim_task(conn, tid)
    return kb.complete_task(conn, tid, summary=summary)


# ---------------------------------------------------------------------------
# DB layer: _check_pr_status mocked
# ---------------------------------------------------------------------------


def test_open_zero_commits_blocked(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("OPEN", True, 0)):
            with pytest.raises(kb.PRNotReadyError, match="zero commits"):
                _complete(conn, tid)
        task = _open_task(conn, tid)
        assert task.status == "running"


def test_merged_allowed(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("MERGED", False, 3)):
            assert _complete(conn, tid) is True
        task = _open_task(conn, tid)
        assert task.status == "done"


def test_open_mergeable_with_commits_allowed(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("OPEN", True, 2)):
            assert _complete(conn, tid) is True
        task = _open_task(conn, tid)
        assert task.status == "done"


def test_closed_not_merged_blocked(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("CLOSED", False, 2)):
            with pytest.raises(kb.PRNotReadyError, match="closed without merging"):
                _complete(conn, tid)
        assert _open_task(conn, tid).status == "running"


def test_open_not_mergeable_blocked(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("OPEN", False, 2)):
            with pytest.raises(kb.PRNotReadyError, match="merge conflicts"):
                _complete(conn, tid)
        assert _open_task(conn, tid).status == "running"


def test_null_pr_url_skips_gate(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=None)
        with mock.patch.object(kb, "_check_pr_status") as mock_check:
            assert _complete(conn, tid) is True
            mock_check.assert_not_called()
        assert _open_task(conn, tid).status == "done"


def test_gh_failure_fails_open_and_emits_event(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        fake_proc = mock.MagicMock()
        fake_proc.returncode = 1
        fake_proc.stderr = "network error"
        fake_proc.stdout = ""
        with mock.patch("hermes_cli.kanban_db.subprocess.run", return_value=fake_proc):
            assert _complete(conn, tid) is True
        task = _open_task(conn, tid)
        assert task.status == "done"
        events = kb.list_events(conn, tid)
        assert any(e.kind == "pr_gate_skipped" for e in events)


def test_block_adds_rationale_comment(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("OPEN", True, 0)):
            with pytest.raises(kb.PRNotReadyError):
                _complete(conn, tid)
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        body = comments[0].body
        assert PR_URL in body
        assert "State: OPEN" in body
        assert "Mergeable: True" in body
        assert "Commits: 0" in body
        assert "Fix the PR and retry" in body


def test_block_emits_completion_blocked_event(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("OPEN", False, 2)):
            with pytest.raises(kb.PRNotReadyError):
                _complete(conn, tid)
        events = kb.list_events(conn, tid)
        blocked = [e for e in events if e.kind == "completion_blocked_pr_not_ready"]
        assert len(blocked) == 1
        payload = blocked[0].payload
        assert payload["pr_url"] == PR_URL
        assert payload["pr_state"] == "OPEN"
        assert payload["mergeable"] is False
        assert payload["commit_count"] == 2


def test_unknown_state_blocked(kanban_home):
    with kb.connect() as conn:
        tid = _make_ready_task(conn, pr_url=PR_URL)
        with mock.patch.object(kb, "_check_pr_status", return_value=("DRAFT", True, 1)):
            with pytest.raises(kb.PRNotReadyError, match="DRAFT.*not merge-ready"):
                _complete(conn, tid)


# ---------------------------------------------------------------------------
# set_pr_url helper
# ---------------------------------------------------------------------------


def test_set_pr_url_persists(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="a")
        kb.set_pr_url(conn, tid, PR_URL)
        task = _open_task(conn, tid)
        assert task.pr_url == PR_URL


def test_set_pr_url_empty_clears(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="a", pr_url=PR_URL)
        kb.set_pr_url(conn, tid, "")
        task = _open_task(conn, tid)
        assert task.pr_url is None


def test_set_pr_url_none_clears(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="a", pr_url=PR_URL)
        kb.set_pr_url(conn, tid, None)
        task = _open_task(conn, tid)
        assert task.pr_url is None


# ---------------------------------------------------------------------------
# Tool handlers (need HERMES_KANBAN_TASK absent so set_pr_url is available)
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator_env(monkeypatch, tmp_path):
    """Isolated Hermes home with the kanban toolset enabled."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    import tools.kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache
    from hermes_cli.config import cfg_get

    # Enable the kanban toolset for this profile.
    (home / "config.yaml").write_text("toolsets:\n  - kanban\n")
    invalidate_check_fn_cache()

    # Force load_config to pick up the new config.
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    return kt


def test_tool_set_pr_url_sets_and_clears(orchestrator_env, tmp_path, monkeypatch):
    kt = orchestrator_env
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as _kb
    conn = _kb.connect()
    try:
        tid = _kb.create_task(conn, title="x", assignee="a")
    finally:
        conn.close()

    out = kt._handle_set_pr_url({"task_id": tid, "pr_url": PR_URL})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["pr_url"] == PR_URL

    conn = _kb.connect()
    try:
        assert _kb.get_task(conn, tid).pr_url == PR_URL
    finally:
        conn.close()

    out = kt._handle_set_pr_url({"task_id": tid, "pr_url": None})
    assert json.loads(out)["ok"] is True
    conn = _kb.connect()
    try:
        assert _kb.get_task(conn, tid).pr_url is None
    finally:
        conn.close()


def test_tool_create_with_pr_url(orchestrator_env, tmp_path, monkeypatch):
    kt = orchestrator_env
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as _kb
    conn = _kb.connect()
    try:
        out = kt._handle_create({
            "title": "child",
            "assignee": "factory",
            "pr_url": PR_URL,
        })
        d = json.loads(out)
        assert d["ok"] is True
        tid = d["task_id"]
        task = _open_task(conn, tid)
        assert task.pr_url == PR_URL
        events = _kb.list_events(conn, tid)
        assert any(e.kind == "created" and e.payload.get("pr_url") == PR_URL for e in events)
    finally:
        conn.close()
