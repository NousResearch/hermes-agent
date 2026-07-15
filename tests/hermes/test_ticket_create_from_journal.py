"""Tests for issue #9: journal dev-todo approve-to-backlog button.

AC coverage:
  AC1  - Dev-category todos from journal_brief.latest.json appear in morning brief.
  AC2  - Each todo has an [Approve] button; only authorised users can trigger it.
  AC3  - [Approve] posts to POST /api/tickets/create with title, body, project,
         extra_labels containing origin:journal.
  AC4  - On success the bot confirms in-channel ("✅ Ticket created: #<id>").
  AC5  - If a todo is never approved, no ticket is created and no side-effects occur.
  AC6  - Idempotent: a second approval for the same todo id does NOT create a duplicate.
  AC7  - Each attempt (success or duplicate-skip) produces an audit log line with
         at minimum: timestamp, todo_id, actor, outcome.
  AC8  - origin:journal label existence is a pre-deploy checklist item, not created
         at runtime (verified by checking the label is forwarded as-is in the request).
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_brief(todos: list[dict]) -> dict:
    """Build a minimal journal_brief.latest.json structure."""
    return {"todos": todos}


def _write_brief(path: Path, todos: list[dict]) -> None:
    path.write_text(json.dumps(_make_brief(todos)), encoding="utf-8")


def _make_todo(
    todo_id: str = "todo-1",
    title: str = "Fix auth bug",
    body: str = "Details about the bug",
    category: str = "dev",
) -> dict:
    return {"id": todo_id, "category": category, "title": title, "body": body}


def _reload_modules():
    for mod in list(sys.modules):
        if "services.hermes" in mod:
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# AC1 — load_dev_todos filters by category="dev"
# ---------------------------------------------------------------------------

class TestLoadDevTodos:
    def setup_method(self):
        _reload_modules()

    def test_returns_only_dev_category(self, tmp_path):
        """AC1 — only dev-category todos are surfaced."""
        brief_file = tmp_path / "journal_brief.latest.json"
        _write_brief(brief_file, [
            _make_todo("t1", "Task 1", category="dev"),
            _make_todo("t2", "Task 2", category="fitness"),
            _make_todo("t3", "Task 3", category="dev"),
        ])
        from services.hermes.journal_approve import load_dev_todos
        todos = load_dev_todos(str(brief_file))
        ids = [t["id"] for t in todos]
        assert "t1" in ids
        assert "t3" in ids
        assert "t2" not in ids

    def test_empty_brief_returns_empty_list(self, tmp_path):
        """AC1 — no todos → empty list, no error."""
        brief_file = tmp_path / "journal_brief.latest.json"
        _write_brief(brief_file, [])
        from services.hermes.journal_approve import load_dev_todos
        assert load_dev_todos(str(brief_file)) == []

    def test_missing_file_returns_empty_list(self, tmp_path):
        """AC1 — missing file returns [] without raising."""
        from services.hermes.journal_approve import load_dev_todos
        result = load_dev_todos(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_malformed_json_returns_empty_list(self, tmp_path):
        """AC1 — corrupt file returns [] without raising."""
        brief_file = tmp_path / "journal_brief.latest.json"
        brief_file.write_text("not valid json", encoding="utf-8")
        from services.hermes.journal_approve import load_dev_todos
        assert load_dev_todos(str(brief_file)) == []

    def test_no_dev_category_todos_returns_empty(self, tmp_path):
        """AC1 — brief with only non-dev todos returns []."""
        brief_file = tmp_path / "journal_brief.latest.json"
        _write_brief(brief_file, [
            _make_todo("t1", category="fitness"),
            _make_todo("t2", category="mindset"),
        ])
        from services.hermes.journal_approve import load_dev_todos
        assert load_dev_todos(str(brief_file)) == []

    def test_todo_has_required_fields(self, tmp_path):
        """AC1 — returned todos carry id, title, body."""
        brief_file = tmp_path / "journal_brief.latest.json"
        _write_brief(brief_file, [_make_todo("t1", "My Title", "My Body")])
        from services.hermes.journal_approve import load_dev_todos
        todo = load_dev_todos(str(brief_file))[0]
        assert todo["id"] == "t1"
        assert todo["title"] == "My Title"
        assert todo["body"] == "My Body"


# ---------------------------------------------------------------------------
# AC3 — handle_journal_approve POSTs to /api/tickets/create correctly
# ---------------------------------------------------------------------------

class TestHandleJournalApprove:
    def setup_method(self):
        _reload_modules()

    def _run_approve(
        self, monkeypatch, tmp_path,
        *,
        status: int = 201,
        response_body: dict | None = None,
        todo_id: str = "todo-1",
        title: str = "Fix bug",
        body: str = "Details",
        project: str = "owner/repo",
        user_id: str = "u123",
    ):
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        resp_data = json.dumps(response_body or {"number": 42, "url": "https://github.com/owner/repo/issues/42"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = status
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from services.hermes.journal_approve import handle_journal_approve
            result = handle_journal_approve(
                todo_id=todo_id,
                title=title,
                body=body,
                project=project,
                user_id=user_id,
            )
        return result, mock_urlopen

    def test_success_posts_to_correct_endpoint(self, monkeypatch, tmp_path):
        """AC3 — POSTs to POST /api/tickets/create."""
        result, mock_urlopen = self._run_approve(monkeypatch, tmp_path)
        assert result["success"] is True
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert "/api/tickets/create" in req.full_url

    def test_post_includes_title_and_body(self, monkeypatch, tmp_path):
        """AC3 — form data includes title, body fields."""
        result, mock_urlopen = self._run_approve(
            monkeypatch, tmp_path, title="Fix login", body="The bug is..."
        )
        req = mock_urlopen.call_args[0][0]
        payload = req.data.decode("utf-8")
        assert "Fix+login" in payload or "Fix%20login" in payload or "Fix login" in payload.replace("+", " ").replace("%20", " ")
        assert "bug" in payload.lower()

    def test_post_includes_project(self, monkeypatch, tmp_path):
        """AC3 — project field is included in the form data."""
        result, mock_urlopen = self._run_approve(
            monkeypatch, tmp_path, project="zealchaiwut/hermes-agent"
        )
        req = mock_urlopen.call_args[0][0]
        payload = req.data.decode("utf-8")
        assert "hermes" in payload.lower() or "zealchaiwut" in payload.lower()

    def test_post_includes_origin_journal_label(self, monkeypatch, tmp_path):
        """AC3 / AC8 — extra_labels includes 'origin:journal' (label forwarded as-is)."""
        result, mock_urlopen = self._run_approve(monkeypatch, tmp_path)
        req = mock_urlopen.call_args[0][0]
        payload = req.data.decode("utf-8")
        assert "origin" in payload and "journal" in payload

    def test_returns_ticket_number_on_success(self, monkeypatch, tmp_path):
        """AC4 — success result contains ticket number so the bot can confirm."""
        result, _ = self._run_approve(
            monkeypatch, tmp_path,
            response_body={"number": 77, "url": "https://github.com/x/y/issues/77"},
        )
        assert result["success"] is True
        assert result.get("ticket_number") == 77 or result.get("number") == 77

    def test_missing_commander_url_returns_error(self, monkeypatch, tmp_path):
        """AC3 — missing COMMANDER_API_URL → clear error, no HTTP call."""
        monkeypatch.delenv("COMMANDER_API_URL", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with patch("urllib.request.urlopen") as mock_urlopen:
            from services.hermes.journal_approve import handle_journal_approve
            result = handle_journal_approve(
                todo_id="t1", title="T", body="B", project="o/r", user_id="u1"
            )
        assert result["success"] is False
        assert result.get("error")
        mock_urlopen.assert_not_called()

    def test_http_error_returns_error(self, monkeypatch, tmp_path):
        """AC3 — non-2xx response → clear error, success=False."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        exc = urllib.error.HTTPError("url", 502, "Bad Gateway", {}, None)
        with patch("urllib.request.urlopen", side_effect=exc):
            from services.hermes.journal_approve import handle_journal_approve
            result = handle_journal_approve(
                todo_id="t1", title="T", body="B", project="o/r", user_id="u1"
            )
        assert result["success"] is False
        assert result.get("error")

    def test_url_error_returns_error(self, monkeypatch, tmp_path):
        """AC3 — connection refused → friendly error."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        exc = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=exc):
            from services.hermes.journal_approve import handle_journal_approve
            result = handle_journal_approve(
                todo_id="t1", title="T", body="B", project="o/r", user_id="u1"
            )
        assert result["success"] is False
        assert result.get("error")
        assert "URLError" not in (result.get("error") or "")


# ---------------------------------------------------------------------------
# AC5 — no side-effects when todo is never approved
# ---------------------------------------------------------------------------

class TestNoSideEffectsWhenNotApproved:
    """AC5 — verifies that load_dev_todos + idle state produce no tickets."""

    def setup_method(self):
        _reload_modules()

    def test_loading_todos_makes_no_http_call(self, tmp_path):
        """AC5 — merely loading dev todos must not POST anything."""
        brief_file = tmp_path / "journal_brief.latest.json"
        _write_brief(brief_file, [_make_todo("t1")])
        with patch("urllib.request.urlopen") as mock_urlopen:
            from services.hermes.journal_approve import load_dev_todos
            load_dev_todos(str(brief_file))
        mock_urlopen.assert_not_called()

    def test_unapproved_todo_leaves_no_approval_record(self, tmp_path, monkeypatch):
        """AC5 — an unapproved todo has no entry in the idempotency store."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import is_todo_approved
        assert is_todo_approved("todo-xyz") is False


# ---------------------------------------------------------------------------
# AC6 — idempotency: second approval for the same todo id is a no-op
# ---------------------------------------------------------------------------

class TestIdempotency:
    def setup_method(self):
        _reload_modules()

    def test_second_approval_returns_duplicate(self, monkeypatch, tmp_path):
        """AC6 — second call returns duplicate=True, makes no HTTP POST."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 1, "url": "https://github.com/x/y/issues/1"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from services.hermes.journal_approve import handle_journal_approve
            # First approval — should succeed
            r1 = handle_journal_approve(
                todo_id="dup-id", title="T", body="B", project="o/r", user_id="u1"
            )
            assert r1["success"] is True

        _reload_modules()

        with patch("urllib.request.urlopen") as mock_urlopen2:
            monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
            from services.hermes.journal_approve import handle_journal_approve
            # Second approval — must be a no-op
            r2 = handle_journal_approve(
                todo_id="dup-id", title="T", body="B", project="o/r", user_id="u1"
            )

        assert r2.get("duplicate") is True
        mock_urlopen2.assert_not_called()

    def test_is_todo_approved_returns_true_after_first_approval(self, monkeypatch, tmp_path):
        """AC6 — idempotency store is written on first success."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 5}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from services.hermes.journal_approve import handle_journal_approve, is_todo_approved
            handle_journal_approve(
                todo_id="store-test", title="T", body="B", project="o/r", user_id="u1"
            )

        _reload_modules()

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import is_todo_approved
        assert is_todo_approved("store-test") is True

    def test_different_todo_ids_are_independent(self, monkeypatch, tmp_path):
        """AC6 — approving todo A doesn't mark todo B as approved."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 10}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from services.hermes.journal_approve import handle_journal_approve, is_todo_approved
            handle_journal_approve(
                todo_id="todo-A", title="T", body="B", project="o/r", user_id="u1"
            )

        _reload_modules()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import is_todo_approved
        assert is_todo_approved("todo-B") is False


# ---------------------------------------------------------------------------
# AC7 — audit log: every attempt writes timestamp, todo_id, actor, outcome
# ---------------------------------------------------------------------------

class TestAuditLog:
    def setup_method(self):
        _reload_modules()

    def test_success_writes_audit_entry(self, monkeypatch, tmp_path):
        """AC7 — successful approval produces an audit log line."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 99}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="t-audit-1", title="T", body="B", project="o/r", user_id="actor-u1"
            )

        log_path = tmp_path / "logs" / "journal-approve-audit.log"
        assert log_path.exists(), "Audit log not created"
        entry = json.loads(log_path.read_text().strip().splitlines()[-1])
        assert "ts" in entry
        assert entry["todo_id"] == "t-audit-1"
        assert entry["actor"] == "actor-u1"
        assert entry["outcome"].startswith("created") or entry["outcome"] in ("success", "201")

    def test_duplicate_skip_writes_audit_entry(self, monkeypatch, tmp_path):
        """AC7 — duplicate-skip attempt also writes an audit log line."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 100}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="t-dup-audit", title="T", body="B", project="o/r", user_id="u1"
            )

        _reload_modules()
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("urllib.request.urlopen") as mock_urlopen2:
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="t-dup-audit", title="T", body="B", project="o/r", user_id="u2"
            )

        log_path = tmp_path / "logs" / "journal-approve-audit.log"
        lines = [json.loads(l) for l in log_path.read_text().strip().splitlines() if l]
        dup_entry = [e for e in lines if e.get("actor") == "u2"]
        assert dup_entry, "No audit entry for duplicate attempt"
        assert dup_entry[0]["todo_id"] == "t-dup-audit"
        assert "skip" in dup_entry[0]["outcome"].lower() or "dup" in dup_entry[0]["outcome"].lower()

    def test_audit_log_never_raises(self, monkeypatch, tmp_path):
        """AC7 — audit log write failure must not block the command."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 1}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch("builtins.open", side_effect=OSError("disk full")):
                from services.hermes.journal_approve import handle_journal_approve
                # Must not raise
                handle_journal_approve(
                    todo_id="t-err", title="T", body="B", project="o/r", user_id="u1"
                )

    def test_audit_entry_required_fields_present(self, monkeypatch, tmp_path):
        """AC7 — log entry has ts, todo_id, actor, outcome at minimum."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 3}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="t-fields", title="T", body="B", project="o/r", user_id="u-actor"
            )

        log_path = tmp_path / "logs" / "journal-approve-audit.log"
        entry = json.loads(log_path.read_text().strip().splitlines()[-1])
        for field in ("ts", "todo_id", "actor", "outcome"):
            assert field in entry, f"Required field '{field}' missing from audit log"


# ---------------------------------------------------------------------------
# AC8 — origin:journal label is forwarded as-is (not created at runtime)
# ---------------------------------------------------------------------------

class TestOriginJournalLabel:
    """The label is simply forwarded in the POST; the test verifies no label
    creation call is made — only the ticket creation POST."""

    def setup_method(self):
        _reload_modules()

    def test_only_one_http_call_made(self, monkeypatch, tmp_path):
        """AC8 — exactly one HTTP call (ticket creation); no label-create call."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 7}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="label-test", title="T", body="B", project="o/r", user_id="u1"
            )

        assert mock_urlopen.call_count == 1, (
            "Expected exactly one HTTP call (ticket create); got "
            f"{mock_urlopen.call_count} — possible runtime label creation"
        )

    def test_extra_labels_contains_origin_journal(self, monkeypatch, tmp_path):
        """AC3 / AC8 — POST form data contains 'origin:journal' in extra_labels."""
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:8000")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        resp_data = json.dumps({"number": 8}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 201
        mock_resp.read.return_value = resp_data

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            from services.hermes.journal_approve import handle_journal_approve
            handle_journal_approve(
                todo_id="lbl-test", title="T", body="B", project="o/r", user_id="u1"
            )

        req = mock_urlopen.call_args[0][0]
        payload = req.data.decode("utf-8")
        assert "origin" in payload and "journal" in payload, (
            f"origin:journal not found in POST payload: {payload!r}"
        )


# ---------------------------------------------------------------------------
# Config tests — COMMANDER_API_URL from environment
# ---------------------------------------------------------------------------

class TestCommanderConfig:
    def setup_method(self):
        _reload_modules()

    def test_get_commander_api_url_from_env(self, monkeypatch):
        monkeypatch.setenv("COMMANDER_API_URL", "http://localhost:9999")
        from services.hermes.config import get_commander_api_url
        assert get_commander_api_url() == "http://localhost:9999"

    def test_get_commander_api_url_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("COMMANDER_API_URL", raising=False)
        from services.hermes.config import get_commander_api_url
        assert get_commander_api_url() is None


# ---------------------------------------------------------------------------
# Config tests — get_approve_projects (JOURNAL_APPROVE_PROJECTS / _PROJECT)
# ---------------------------------------------------------------------------

class TestGetApproveProjects:
    def setup_method(self):
        _reload_modules()

    def _clear(self, monkeypatch):
        monkeypatch.delenv("JOURNAL_APPROVE_PROJECTS", raising=False)
        monkeypatch.delenv("JOURNAL_APPROVE_PROJECT", raising=False)

    def test_comma_list_parsed_into_multiple_projects(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "owner/repo1,owner/repo2,owner/repo3")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/repo1", "owner/repo2", "owner/repo3"]

    def test_comma_list_entries_are_stripped(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", " owner/repo1 , owner/repo2 ,  owner/repo3  ")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/repo1", "owner/repo2", "owner/repo3"]

    def test_comma_list_drops_empty_entries(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "owner/repo1,,owner/repo2,  ,owner/repo3,")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/repo1", "owner/repo2", "owner/repo3"]

    def test_single_entry_in_plural_var_returns_single_item_list(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "owner/repo1")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/repo1"]

    def test_plural_var_all_commas_and_whitespace_returns_empty_list(self, monkeypatch):
        """JOURNAL_APPROVE_PROJECTS set but contains no real entries -> []
        (does NOT fall back to JOURNAL_APPROVE_PROJECT, since the plural var
        is present/non-empty and takes precedence)."""
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", " , , ")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == []

    def test_fallback_to_singular_when_plural_unset(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECT", "owner/legacy-repo")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/legacy-repo"]

    def test_fallback_singular_is_stripped(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECT", "  owner/legacy-repo  ")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/legacy-repo"]

    def test_plural_takes_precedence_over_singular_when_both_set(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "owner/new1,owner/new2")
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECT", "owner/old")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/new1", "owner/new2"]

    def test_both_unset_returns_empty_list(self, monkeypatch):
        self._clear(monkeypatch)
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == []

    def test_plural_unset_and_singular_empty_string_returns_empty_list(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECT", "")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == []

    def test_plural_empty_string_falls_back_to_singular(self, monkeypatch):
        """An explicitly empty JOURNAL_APPROVE_PROJECTS ('') is falsy, so the
        function falls through to the JOURNAL_APPROVE_PROJECT fallback."""
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "")
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECT", "owner/legacy-repo")
        from services.hermes.config import get_approve_projects
        assert get_approve_projects() == ["owner/legacy-repo"]

    def test_return_type_is_list(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("JOURNAL_APPROVE_PROJECTS", "owner/repo1,owner/repo2")
        from services.hermes.config import get_approve_projects
        result = get_approve_projects()
        assert isinstance(result, list)
