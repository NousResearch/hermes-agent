"""Tests for kanban comment author identity enforcement.

Verifies that the CLI and dashboard API surfaces stamp a machine identity
on comment authors rather than accepting arbitrary caller-supplied strings.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_comments as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def task_id(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="test task", assignee="tester")
    return tid


class TestCLICommentAuthorStamping:

    def test_cli_comment_stamps_cli_prefix_on_profile_author(
        self, task_id, monkeypatch
    ):
        monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        ns = argparse.Namespace(
            task_id=task_id,
            text=["hello", "world"],
            author=None,
            max_len=None,
        )
        with patch.object(kc, "_profile_author", return_value="myprofile"):
            rc = kc._cmd_comment(ns)
        assert rc == 0
        with kb.connect() as conn:
            comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
        assert comments[0].author == "cli:myprofile"

    def test_cli_comment_stamps_cli_prefix_on_explicit_author(
        self, task_id, monkeypatch
    ):
        ns = argparse.Namespace(
            task_id=task_id,
            text=["testing"],
            author="bot-ci",
            max_len=None,
        )
        with patch.object(kc, "_profile_author", return_value="myprofile"):
            rc = kc._cmd_comment(ns)
        assert rc == 0
        with kb.connect() as conn:
            comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
        assert comments[0].author == "cli:bot-ci"

    def test_cli_comment_author_cannot_impersonate_dashboard(
        self, task_id, monkeypatch
    ):
        ns = argparse.Namespace(
            task_id=task_id,
            text=["forged"],
            author="dashboard",
            max_len=None,
        )
        with patch.object(kc, "_profile_author", return_value="attacker"):
            rc = kc._cmd_comment(ns)
        assert rc == 0
        with kb.connect() as conn:
            comments = kb.list_comments(conn, task_id)
        assert len(comments) == 1
        assert comments[0].author == "cli:dashboard"
        assert comments[0].author != "dashboard"


class TestCLIAttachAuthorStamping:

    def test_cli_attach_stamps_cli_prefix(self, task_id, tmp_path, monkeypatch):
        f = tmp_path / "test.txt"
        f.write_text("content")
        ns = argparse.Namespace(
            task_id=task_id,
            path=str(f),
            author=None,
            name=None,
            content_type=None,
        )
        with patch.object(kc, "_profile_author", return_value="myprofile"):
            rc = kc._cmd_attach(ns)
        assert rc == 0
        with kb.connect() as conn:
            atts = kb.list_attachments(conn, task_id)
        assert len(atts) == 1
        assert atts[0].uploaded_by == "cli:myprofile"


class TestDashboardAPIAuthorStamping:

    @pytest.fixture(autouse=True)
    def _require_fastapi(self):
        pytest.importorskip("fastapi")

    def test_dashboard_api_ignores_client_supplied_author(
        self, task_id, monkeypatch
    ):
        from plugins.kanban.dashboard import plugin_api

        class FakePayload:
            body = "test comment"
            author = "attacker-forged-name"

        conn = kb.connect()
        monkeypatch.setattr(
            plugin_api, "_resolve_board", lambda b: None
        )
        monkeypatch.setattr(
            plugin_api, "_conn", lambda board=None: conn
        )
        result = plugin_api.add_comment(task_id, FakePayload(), board=None)

        assert result == {"ok": True}
        with kb.connect() as conn2:
            comments = kb.list_comments(conn2, task_id)
        authored = [c for c in comments if c.body.strip() == "test comment"]
        assert len(authored) == 1
        assert authored[0].author == "dashboard"


class TestKanbanModuleReExports:

    def test_kanban_reexports_comment_and_attachment_handlers(self):
        assert kanban_cli._cmd_comment is kc._cmd_comment
        assert kanban_cli._cmd_attach is kc._cmd_attach
        assert kanban_cli._cmd_attachments is kc._cmd_attachments
        assert kanban_cli._cmd_attach_rm is kc._cmd_attach_rm

