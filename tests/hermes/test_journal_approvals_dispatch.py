"""Tests for the daily journal-approvals dispatcher's pure functions.

Covers:
  - map_dev_todos_for_send: journal-contract todo {id, content/text, note}
    -> DiscordAdapter.send_journal_dev_todos contract {id, title, body}.
  - has_posted_approvals_today / mark_approvals_posted: the date-guard file
    (HERMES_HOME/journal-approvals-posted.date) used to avoid re-posting the
    same day's [Approve] embeds after a Discord gateway restart.
"""
from __future__ import annotations

import os
import stat
import sys

import pytest


def _reload_modules():
    for mod in list(sys.modules):
        if "services.hermes" in mod:
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# map_dev_todos_for_send
# ---------------------------------------------------------------------------

class TestMapDevTodosForSend:
    def setup_method(self):
        _reload_modules()

    def test_happy_path_content_and_note(self):
        """content -> title, note -> body, id passed through."""
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t1", "content": "Fix auth bug", "note": "See PR #4"}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped == [{"id": "t1", "title": "Fix auth bug", "body": "See PR #4"}]

    def test_text_fallback_when_content_missing(self):
        """No 'content' key -> title falls back to 'text'."""
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t2", "text": "Refactor logger", "note": ""}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["title"] == "Refactor logger"

    def test_text_fallback_when_content_empty_string(self):
        """Empty-string 'content' is falsy -> falls back to 'text'."""
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t3", "content": "", "text": "Fallback text", "note": "n"}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["title"] == "Fallback text"

    def test_content_preferred_over_text_when_both_present(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t4", "content": "Content wins", "text": "Text loses", "note": ""}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["title"] == "Content wins"

    def test_missing_note_defaults_to_empty_body(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t5", "content": "No note here"}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["body"] == ""

    def test_missing_id_defaults_to_empty_string(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"content": "No id", "note": "n"}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["id"] == ""

    def test_missing_content_and_text_defaults_to_empty_title(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": "t6", "note": "just a note"}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["title"] == ""
        assert mapped[0]["body"] == "just a note"

    def test_empty_list_returns_empty_list(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        assert map_dev_todos_for_send([]) == []

    def test_preserves_order(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [
            {"id": "a", "content": "A", "note": ""},
            {"id": "b", "content": "B", "note": ""},
            {"id": "c", "content": "C", "note": ""},
        ]
        mapped = map_dev_todos_for_send(todos)
        assert [t["id"] for t in mapped] == ["a", "b", "c"]

    @pytest.mark.parametrize("malformed", [
        "just a string",
        None,
        123,
        ["not", "a", "dict"],
        3.14,
    ])
    def test_malformed_entries_are_skipped(self, malformed):
        """Non-dict items (no .get method) are skipped rather than raising."""
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [
            {"id": "good-1", "content": "Good todo", "note": ""},
            malformed,
            {"id": "good-2", "content": "Also good", "note": ""},
        ]
        mapped = map_dev_todos_for_send(todos)
        ids = [t["id"] for t in mapped]
        assert ids == ["good-1", "good-2"]

    def test_all_malformed_returns_empty_list(self):
        from services.hermes.journal_approve import map_dev_todos_for_send
        mapped = map_dev_todos_for_send(["a", None, 1, []])
        assert mapped == []

    def test_ids_and_fields_are_coerced_to_str(self):
        """Non-string id/content values are coerced via str() rather than raising."""
        from services.hermes.journal_approve import map_dev_todos_for_send
        todos = [{"id": 42, "content": 7, "note": None}]
        mapped = map_dev_todos_for_send(todos)
        assert mapped[0]["id"] == "42"
        assert mapped[0]["title"] == "7"
        assert mapped[0]["body"] == ""


# ---------------------------------------------------------------------------
# has_posted_approvals_today / mark_approvals_posted
# ---------------------------------------------------------------------------

class TestApprovalsDateGuard:
    def setup_method(self):
        _reload_modules()

    def test_has_posted_returns_false_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import has_posted_approvals_today
        assert has_posted_approvals_today("2026-07-15") is False

    def test_mark_then_has_posted_true_for_same_day(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import (
            has_posted_approvals_today,
            mark_approvals_posted,
        )
        mark_approvals_posted("2026-07-15")
        assert has_posted_approvals_today("2026-07-15") is True

    def test_has_posted_false_for_a_different_day(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import (
            has_posted_approvals_today,
            mark_approvals_posted,
        )
        mark_approvals_posted("2026-07-14")
        assert has_posted_approvals_today("2026-07-15") is False

    def test_mark_overwrites_previous_date(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from services.hermes.journal_approve import (
            has_posted_approvals_today,
            mark_approvals_posted,
        )
        mark_approvals_posted("2026-07-14")
        mark_approvals_posted("2026-07-15")
        assert has_posted_approvals_today("2026-07-14") is False
        assert has_posted_approvals_today("2026-07-15") is True

    def test_mark_creates_hermes_home_when_missing(self, monkeypatch, tmp_path):
        """HERMES_HOME doesn't exist yet on disk; mark_approvals_posted must
        create it (mkdir parents=True) rather than raising."""
        home = tmp_path / "does" / "not" / "exist" / "yet"
        monkeypatch.setenv("HERMES_HOME", str(home))
        from services.hermes.journal_approve import (
            has_posted_approvals_today,
            mark_approvals_posted,
        )
        mark_approvals_posted("2026-07-15")
        assert (home / "journal-approvals-posted.date").exists()
        assert has_posted_approvals_today("2026-07-15") is True

    def test_has_posted_never_raises_when_hermes_home_missing(self, monkeypatch, tmp_path):
        home = tmp_path / "also" / "does" / "not" / "exist"
        monkeypatch.setenv("HERMES_HOME", str(home))
        from services.hermes.journal_approve import has_posted_approvals_today
        assert has_posted_approvals_today("2026-07-15") is False

    def test_has_posted_never_raises_when_file_is_a_directory(self, monkeypatch, tmp_path):
        """Edge case: something else created a directory at the guard-file
        path. Reading it raises IsADirectoryError (an OSError subclass);
        the guard must swallow it and report 'not posted'."""
        home = tmp_path
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "journal-approvals-posted.date").mkdir(parents=True)
        from services.hermes.journal_approve import has_posted_approvals_today
        assert has_posted_approvals_today("2026-07-15") is False

    def test_mark_never_raises_on_unwritable_dir(self, monkeypatch, tmp_path):
        """HERMES_HOME parent exists but is read-only: mkdir/write_text will
        fail. mark_approvals_posted must swallow the error, not raise."""
        if os.name != "posix" or os.geteuid() == 0:
            pytest.skip("permission-based test requires a non-root POSIX user")

        locked_parent = tmp_path / "locked"
        locked_parent.mkdir()
        home = locked_parent / "hermes_home"
        monkeypatch.setenv("HERMES_HOME", str(home))

        original_mode = locked_parent.stat().st_mode
        locked_parent.chmod(stat.S_IREAD | stat.S_IEXEC)
        try:
            from services.hermes.journal_approve import mark_approvals_posted
            # Must not raise.
            mark_approvals_posted("2026-07-15")
        finally:
            locked_parent.chmod(original_mode)

    def test_default_hermes_home_used_when_env_unset(self, monkeypatch):
        """Falls back to ~/.hermes when HERMES_HOME is unset — verified via
        the resolved path rather than touching the real home directory."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        from pathlib import Path

        from services.hermes.journal_approve import _resolve_approvals_posted_file
        expected = Path.home() / ".hermes" / "journal-approvals-posted.date"
        assert _resolve_approvals_posted_file() == expected
