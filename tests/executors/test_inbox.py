#!/usr/bin/env python3
"""
Tests for executors/inbox.py — InboxManager.

Scope:
  - Add / list / update / convert / reject / archive / expire round-trip
  - Filters (status, source) on list_items
  - get_writeback_callback round-trip
  - writeback_destination descriptive string per source
  - count_by_status / count_pending_by_source
  - Corrupt JSON graceful handling
  - Does NOT write to the real repo /Users/gu/.hermes/hermes-agent
  - Does NOT create worktrees
  - Does NOT spawn subprocesses
  - Does NOT call any model
  - Does NOT actually write to ~/.hermes/inbox-results/ (only returns
    a descriptive destination string)

Strictly no live network calls, no model invocations, no worktree creation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from executors.inbox import INBOX_FILENAME, InboxManager, create_inbox_manager
from executors.types import (
    InboxItem,
    InboxResultCallback,
    InboxSource,
    InboxStatus,
    TaskDraft,
)


# ---------------------------------------------------------------------------
# 1. Add / list round-trip
# ---------------------------------------------------------------------------

class TestAddListRoundTrip:
    def test_add_creates_pending_item(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(
            source=InboxSource.CLI,
            title="Fix login bug",
            body="Login flow fails when password contains '@'",
        )
        assert isinstance(item, InboxItem)
        assert item.id.startswith("inbox-")
        assert item.source == InboxSource.CLI
        assert item.status == InboxStatus.PENDING
        assert item.draft.title == "Fix login bug"
        assert item.draft.suggested_prompt == "Login flow fails when password contains '@'"
        assert item.draft.priority == "normal"
        assert item.draft.user_edited is False
        assert item.linked_task_id is None
        assert item.rejected_reason is None
        # The inbox.json file should have been written.
        assert (tmp_path / ".hermes" / INBOX_FILENAME).exists()

    def test_list_returns_all_items(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.DESKTOP, "T2", "B2")
        mgr.add(InboxSource.FEISHU, "T3", "B3")
        items = mgr.list_items()
        assert len(items) == 3
        titles = sorted(it.draft.title for it in items)
        assert titles == ["T1", "T2", "T3"]

    def test_list_filters_by_status(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        a = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.CLI, "T2", "B2")
        mgr.convert_to_task(a.id, "task-99")
        pending = mgr.list_items(status=InboxStatus.PENDING)
        confirmed = mgr.list_items(status=InboxStatus.CONFIRMED)
        assert len(pending) == 1
        assert pending[0].draft.title == "T2"
        assert len(confirmed) == 1
        assert confirmed[0].draft.title == "T1"

    def test_list_filters_by_source(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.DESKTOP, "T2", "B2")
        cli_items = mgr.list_items(source=InboxSource.CLI)
        assert len(cli_items) == 1
        assert cli_items[0].draft.title == "T1"

    def test_list_combines_status_and_source_filters(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.DESKTOP, "T2", "B2")
        cli_pending = mgr.list_items(
            status=InboxStatus.PENDING, source=InboxSource.CLI,
        )
        assert len(cli_pending) == 1
        # CLI confirmed is empty.
        cli_confirmed = mgr.list_items(
            status=InboxStatus.CONFIRMED, source=InboxSource.CLI,
        )
        assert cli_confirmed == []

    def test_list_pending_helper(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        a = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.CLI, "T2", "B2")
        mgr.convert_to_task(a.id, "task-1")
        pending = mgr.list_pending()
        assert len(pending) == 1
        assert pending[0].draft.title == "T2"


# ---------------------------------------------------------------------------
# 2. State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_convert_to_task_marks_confirmed(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        result = mgr.convert_to_task(item.id, "task-xyz")
        assert result is not None
        assert result.status == InboxStatus.CONFIRMED
        assert result.linked_task_id == "task-xyz"
        # Re-reading from disk should reflect the new status.
        reread = mgr.get(item.id)
        assert reread.status == InboxStatus.CONFIRMED
        assert reread.linked_task_id == "task-xyz"

    def test_convert_unknown_id_returns_none(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        assert mgr.convert_to_task("does-not-exist", "task-1") is None

    def test_convert_already_confirmed_returns_existing(
        self, tmp_path: Path
    ) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.convert_to_task(item.id, "task-1")
        # Second convert should not raise; returns the existing item.
        result = mgr.convert_to_task(item.id, "task-2")
        assert result is not None
        # Original linked_task_id is preserved.
        assert result.linked_task_id == "task-1"

    def test_reject_sets_rejected_status(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.reject(item.id, "out of scope")
        reread = mgr.get(item.id)
        assert reread.status == InboxStatus.REJECTED
        assert reread.rejected_reason == "out of scope"

    def test_archive_sets_archived_status(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.archive(item.id)
        assert mgr.get(item.id).status == InboxStatus.ARCHIVED

    def test_expire_sets_expired_status(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.expire(item.id)
        assert mgr.get(item.id).status == InboxStatus.EXPIRED

    def test_reject_unknown_returns_none(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        assert mgr.reject("nope") is None
        assert mgr.archive("nope") is None
        assert mgr.expire("nope") is None


# ---------------------------------------------------------------------------
# 3. update_draft
# ---------------------------------------------------------------------------

class TestUpdateDraft:
    def test_update_title_marks_user_edited(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        assert item.draft.user_edited is False
        mgr.update_draft(item.id, title="New Title")
        reread = mgr.get(item.id)
        assert reread.draft.title == "New Title"
        assert reread.draft.user_edited is True

    def test_update_prompt_marks_user_edited(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.update_draft(item.id, prompt="New prompt body")
        reread = mgr.get(item.id)
        assert reread.draft.suggested_prompt == "New prompt body"
        assert reread.draft.user_edited is True

    def test_update_executor_does_not_set_user_edited(
        self, tmp_path: Path
    ) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.update_draft(item.id, executor="codex-cli")
        reread = mgr.get(item.id)
        assert reread.draft.suggested_executor == "codex-cli"
        # Per the source: only title/prompt changes set user_edited.
        assert reread.draft.user_edited is False

    def test_update_project_and_priority(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.update_draft(item.id, project="repo-x", priority="high")
        reread = mgr.get(item.id)
        assert reread.draft.project_hint == "repo-x"
        assert reread.draft.priority == "high"

    def test_update_unknown_returns_none(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        assert mgr.update_draft("nope", title="x") is None

    def test_no_args_keeps_existing(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(
            InboxSource.CLI, "T1", "B1",
            suggested_executor="claude-code", project_hint="repo-x", priority="high",
        )
        # Call with no overrides — all fields stay the same.
        mgr.update_draft(item.id)
        reread = mgr.get(item.id)
        assert reread.draft.title == "T1"
        assert reread.draft.suggested_executor == "claude-code"


# ---------------------------------------------------------------------------
# 4. Writeback callback
# ---------------------------------------------------------------------------

class TestWritebackCallback:
    def test_cli_source_writeback_available(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        cb = mgr.get_writeback_callback(
            item.id, run_id="run-1", summary="done",
        )
        assert isinstance(cb, InboxResultCallback)
        assert cb.inbox_item_id == item.id
        assert cb.run_id == "run-1"
        assert cb.status == "done"
        assert cb.summary == "done"
        assert cb.writeback_available is True

    def test_desktop_source_writeback_not_available(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.DESKTOP, "T1", "B1")
        cb = mgr.get_writeback_callback(
            item.id, run_id="run-1", summary="done",
        )
        assert cb is not None
        assert cb.writeback_available is False

    def test_stub_source_writeback_not_available(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        for source in (InboxSource.FEISHU, InboxSource.DISCORD, InboxSource.SCHEDULER):
            item = mgr.add(source, "T", "B")
            cb = mgr.get_writeback_callback(
                item.id, run_id="run-1", summary="done",
            )
            assert cb is not None
            assert cb.writeback_available is False, f"{source} should be unavailable"

    def test_summary_truncated_to_500_chars(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        long_summary = "x" * 2000
        cb = mgr.get_writeback_callback(
            item.id, run_id="run-1", summary=long_summary,
        )
        assert len(cb.summary) == 500

    def test_callback_for_unknown_id_returns_none(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        assert mgr.get_writeback_callback(
            "does-not-exist", run_id="r", summary="s",
        ) is None

    def test_writeback_destination_describes_path_only(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """writeback_destination returns a descriptive string. It must NOT
        actually write a file to ~/.hermes/inbox-results/.
        """
        # Redirect HOME so we can detect any writes.
        leak_root = tmp_path / "leak_check"
        leak_root.mkdir()
        leak_hermes = leak_root / ".hermes"
        monkeypatch.setenv("HOME", str(leak_root))

        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        dest = InboxManager.writeback_destination(item)
        # The string contains the home-dir-marker; we don't write, just return.
        assert "inbox-results" in dest
        # Confirm no file actually got created.
        results_dir = leak_root / ".hermes" / "inbox-results"
        assert not results_dir.exists(), (
            f"Unexpected write to {results_dir} — writeback_destination must be descriptive only"
        )

    def test_writeback_destination_per_source(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        for source, expected_substr in [
            (InboxSource.DESKTOP, "manual"),
            (InboxSource.CLI, "inbox-results"),
            (InboxSource.FEISHU, "unavailable"),
            (InboxSource.DISCORD, "unavailable"),
            (InboxSource.SCHEDULER, "unavailable"),
        ]:
            item = mgr.add(source, "T", "B")
            dest = InboxManager.writeback_destination(item)
            assert expected_substr in dest.lower(), (
                f"source {source}: dest {dest!r} missing {expected_substr!r}"
            )


# ---------------------------------------------------------------------------
# 5. Counts
# ---------------------------------------------------------------------------

class TestCounts:
    def test_count_by_status(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        a = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.CLI, "T2", "B2")
        mgr.add(InboxSource.CLI, "T3", "B3")
        mgr.convert_to_task(a.id, "task-1")
        mgr.reject(mgr.list_items(status=InboxStatus.PENDING)[0].id, "x")
        counts = mgr.count_by_status()
        assert counts["pending"] == 1
        assert counts["confirmed"] == 1
        assert counts["rejected"] == 1

    def test_count_pending_by_source(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.CLI, "T2", "B2")
        mgr.add(InboxSource.DESKTOP, "T3", "B3")
        counts = mgr.count_pending_by_source()
        assert counts["cli"] == 2
        assert counts["desktop"] == 1


# ---------------------------------------------------------------------------
# 6. Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        item = mgr.add(
            InboxSource.CLI, "Refactor auth", "Refactor the auth module",
            suggested_executor="codex-cli", project_hint="repo-x",
            priority="high",
        )
        mgr2 = InboxManager(tmp_path)
        items = mgr2.list_items()
        assert len(items) == 1
        assert items[0].id == item.id
        assert items[0].draft.title == "Refactor auth"
        assert items[0].draft.suggested_executor == "codex-cli"
        assert items[0].draft.priority == "high"

    def test_serialization_includes_all_fields(self, tmp_path: Path) -> None:
        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T", "B")
        path = tmp_path / ".hermes" / INBOX_FILENAME
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        record = data[0]
        for required in (
            "id", "source", "raw_payload", "draft", "status",
            "created_at", "expires_at", "linked_task_id", "rejected_reason",
        ):
            assert required in record, f"missing field: {required}"
        assert record["source"] == "cli"
        assert record["status"] == "pending"


# ---------------------------------------------------------------------------
# 7. Malformed JSON graceful handling
# ---------------------------------------------------------------------------

class TestMalformedJsonGraceful:
    def test_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / INBOX_FILENAME).write_text("{ not json")
        mgr = InboxManager(tmp_path)
        assert mgr.list_items() == []
        # Subsequent add should still work.
        mgr.add(InboxSource.CLI, "T1", "B1")
        assert len(mgr.list_items()) == 1

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / INBOX_FILENAME).write_text("")
        mgr = InboxManager(tmp_path)
        assert mgr.list_items() == []


# ---------------------------------------------------------------------------
# 8. Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_factory_creates_inbox_manager(self, tmp_path: Path) -> None:
        mgr = create_inbox_manager(tmp_path)
        assert isinstance(mgr, InboxManager)
        assert mgr._project_root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 9. No side effects
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    def test_does_not_write_outside_project_root(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """All inbox writes must land under tmp_path. The real HOME/HERMES_HOME
        are re-pointed to confirm no leakage to ~/.hermes/ or similar.
        """
        leak_root = tmp_path / "leak_check"
        leak_root.mkdir()
        leak_hermes = leak_root / "leak_hermes"
        monkeypatch.setenv("HOME", str(leak_root))
        monkeypatch.setenv("HERMES_HOME", str(leak_hermes))

        mgr = InboxManager(tmp_path)
        item = mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.DESKTOP, "T2", "B2")
        mgr.update_draft(item.id, title="new")
        mgr.convert_to_task(item.id, "task-1")

        # Inbox file is under the project root, not the leak detector.
        assert (tmp_path / ".hermes" / INBOX_FILENAME).exists()
        # And writeback_destination must not have written anything.
        InboxManager.writeback_destination(mgr.get(item.id))
        results_dir = leak_root / ".hermes" / "inbox-results"
        assert not results_dir.exists()
        # HERMES_HOME untouched.
        if leak_hermes.exists():
            assert not any(p.is_file() for p in leak_hermes.rglob("*"))

    def test_does_not_call_subprocess(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        popen_calls: list = []
        original = subprocess.Popen

        def tracking(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", tracking)

        mgr = InboxManager(tmp_path)
        mgr.add(InboxSource.CLI, "T1", "B1")
        mgr.add(InboxSource.DESKTOP, "T2", "B2")
        mgr.update_draft(mgr.list_items()[0].id, title="x")
        mgr.convert_to_task(mgr.list_items()[0].id, "t-1")
        mgr.reject(mgr.list_items()[0].id, "x")
        mgr.archive(mgr.list_items()[0].id)

        assert popen_calls == [], f"Unexpected subprocess: {popen_calls}"

    def test_does_not_import_worktree(self) -> None:
        import sys
        for mod_name in list(sys.modules):
            if mod_name == "executors.inbox":
                del sys.modules[mod_name]
        import executors.inbox  # noqa: F401
        assert "executors.worktree" not in sys.modules
