"""Tests for hermes_cli.kanban_pr — PR discovery, status, merge auto-complete."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr as kp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_extract_github_pr_url_strips_trailing_punctuation():
    text = "See https://github.com/acme/widget/pull/42)."
    assert kp.extract_github_pr_url(text) == "https://github.com/acme/widget/pull/42"


def test_find_pr_url_from_comment(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="with-pr", assignee="alice")
        kb.add_comment(
            conn,
            task_id,
            "worker",
            "Opened https://github.com/acme/widget/pull/7 for review.",
        )
        urls = kp.find_pr_urls_for_tasks(conn, [task_id])
    assert urls[task_id] == "https://github.com/acme/widget/pull/7"


def test_find_pr_url_from_run_summary(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="run-pr", assignee="alice")
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at, summary) "
            "VALUES (?, 'done', 'completed', ?, ?, ?)",
            (
                task_id,
                int(time.time()) - 60,
                int(time.time()),
                "PR: https://github.com/acme/widget/pull/99",
            ),
        )
        conn.commit()
        urls = kp.find_pr_urls_for_tasks(conn, [task_id])
    assert urls[task_id] == "https://github.com/acme/widget/pull/99"


def test_fetch_pull_request_info_parses_merged(kanban_home, monkeypatch):
    kp._PR_STATUS_CACHE.clear()

    class FakeResponse:
        status_code = 200

        def is_success(self):
            return True

        def json(self):
            return {
                "state": "closed",
                "merged_at": "2026-01-01T00:00:00Z",
                "draft": False,
                "base": {"ref": "main"},
                "html_url": "https://github.com/acme/widget/pull/7",
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(kp.httpx, "Client", FakeClient)
    monkeypatch.setattr(kp, "_resolve_github_token", lambda: "test-token")

    info = kp.fetch_pull_request_info("https://github.com/acme/widget/pull/7")
    assert info is not None
    assert info.merged is True
    assert info.label == "Merged"
    assert info.target_branch == "main"


def test_sync_merged_pull_requests_completes_blocked_task(kanban_home, monkeypatch):
    kp._PR_STATUS_CACHE.clear()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="await-merge", assignee="alice")
        kb.block_task(conn, task_id, reason="review-required: waiting on PR")
        kb.add_comment(
            conn,
            task_id,
            "worker",
            "https://github.com/acme/widget/pull/12",
        )

    merged = kp.PullRequestInfo(
        url="https://github.com/acme/widget/pull/12",
        state="closed",
        merged=True,
        draft=False,
        target_branch="develop",
        label="Merged",
    )

    with patch.object(kp, "fetch_pull_request_info", return_value=merged):
        with kb.connect() as conn:
            completed = kp.sync_merged_pull_requests(conn)
            task = kb.get_task(conn, task_id)
            summary = kb.latest_summary(conn, task_id)

    assert completed == [task_id]
    assert task.status == "done"
    assert summary == "PR Merged into develop"


def test_attach_pr_status_to_task_dicts(kanban_home, monkeypatch):
    kp._PR_STATUS_CACHE.clear()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="card-pr", assignee="alice")
        kb.add_comment(
            conn,
            task_id,
            "worker",
            "https://github.com/acme/widget/pull/3",
        )

    open_pr = kp.PullRequestInfo(
        url="https://github.com/acme/widget/pull/3",
        state="open",
        merged=False,
        draft=False,
        target_branch="main",
        label="Open",
    )

    with patch.object(kp, "fetch_pull_request_info", return_value=open_pr):
        with kb.connect() as conn:
            task_dicts = [{"id": task_id}]
            kp.attach_pr_status_to_task_dicts(conn, task_dicts)

    assert task_dicts[0]["pr"]["label"] == "Open"
    assert task_dicts[0]["pr"]["url"] == "https://github.com/acme/widget/pull/3"
