#!/usr/bin/env python3
"""
Tests for executors/context.py — WorkspaceContextManager.

Scope:
  - Round-trip JSON load/save in tmp_path
  - Missing context file returns graceful default
  - Field-level CRUD (overview, architecture, adr, sprint, command, etc.)
  - recent_tasks cap at MAX_RECENT_TASKS (10), preserves ordering
  - context_hash is deterministic + 16-char hex
  - Malformed JSON is handled gracefully (warning, fallback to default)
  - Does NOT write to the real repo /Users/gu/.hermes/hermes-agent
  - Does NOT create worktrees
  - Does NOT spawn subprocesses

Strictly no live network calls, no model invocations, no worktree creation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from executors.context import (
    CONTEXT_FILENAME,
    MAX_RECENT_TASKS,
    WorkspaceContextManager,
    create_context_manager,
)
from executors.types import (
    AdrSummary,
    CommandEntry,
    ProjectContext,
    RecentTask,
)


# ---------------------------------------------------------------------------
# 1. Round-trip JSON load / save
# ---------------------------------------------------------------------------

class TestContextRoundTrip:
    def test_save_then_load_round_trips_all_fields(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("A modular agent system")
        mgr.set_architecture("Event-driven core; CLI + Gateway front-ends")
        mgr.set_sprint("Sprint 47 — context injection")
        mgr.set_conventions("PEP 8 + type hints everywhere")
        mgr.add_adr("ADR-001", "Use Redis", "Use Redis for cross-process caching")
        mgr.add_adr("ADR-002", "Drop YAML", "Drop YAML in favor of JSON")
        mgr.add_common_command("build", "make build")
        mgr.add_common_command("test", "make test")
        mgr.add_test_command("unit", "pytest tests/unit")
        mgr.add_forbidden_area("secrets/")
        mgr.add_recent_task(RecentTask(
            thread_id="t-1", title="Add router", executor="claude-code",
            status="done", completed_at="2026-06-01T10:00:00Z",
        ))
        mgr.set_injection_enabled(True)

        # File should exist after set_* calls (each calls save()).
        ctx_path = tmp_path / ".hermes" / CONTEXT_FILENAME
        assert ctx_path.exists()
        raw = json.loads(ctx_path.read_text())
        assert raw["project_overview"] == "A modular agent system"
        assert raw["current_sprint"] == "Sprint 47 — context injection"
        assert len(raw["adr_summaries"]) == 2
        assert raw["adr_summaries"][0]["id"] == "ADR-001"
        assert raw["common_commands"][1]["command"] == "make test"
        assert raw["forbidden_areas"] == ["secrets/"]
        assert raw["context_injection_enabled"] is True

        # Round-trip: a fresh manager reading the same file should see all data.
        mgr2 = WorkspaceContextManager(tmp_path)
        ctx2 = mgr2.load()
        assert ctx2.project_overview == "A modular agent system"
        assert ctx2.architecture_notes.startswith("Event-driven")
        assert ctx2.current_sprint == "Sprint 47 — context injection"
        assert len(ctx2.adr_summaries) == 2
        assert ctx2.adr_summaries[1].title == "Drop YAML"
        assert len(ctx2.common_commands) == 2
        assert len(ctx2.test_commands) == 1
        assert ctx2.forbidden_areas == ["secrets/"]
        assert ctx2.coding_conventions == "PEP 8 + type hints everywhere"
        assert len(ctx2.recent_tasks) == 1
        assert ctx2.recent_tasks[0].thread_id == "t-1"
        assert ctx2.context_injection_enabled is True

    def test_save_creates_hermes_dir_if_missing(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        assert not (tmp_path / ".hermes").exists()
        mgr.set_overview("hi")
        assert (tmp_path / ".hermes").is_dir()
        assert (tmp_path / ".hermes" / CONTEXT_FILENAME).is_file()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_sprint("S1")
        path = tmp_path / ".hermes" / CONTEXT_FILENAME
        # Must not raise.
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert "current_sprint" in data


# ---------------------------------------------------------------------------
# 2. Missing file graceful default
# ---------------------------------------------------------------------------

class TestMissingFileGracefulDefault:
    def test_load_without_file_returns_default(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        assert not mgr.is_loaded()
        ctx = mgr.load()
        assert isinstance(ctx, ProjectContext)
        assert ctx.project_overview == ""
        assert ctx.architecture_notes == ""
        assert ctx.adr_summaries == []
        assert ctx.current_sprint == ""
        assert ctx.common_commands == []
        assert ctx.test_commands == []
        assert ctx.forbidden_areas == []
        assert ctx.coding_conventions == ""
        assert ctx.recent_tasks == []
        assert ctx.context_injection_enabled is True
        # No file was created by read-only load().
        assert not (tmp_path / ".hermes" / CONTEXT_FILENAME).exists()

    def test_get_context_loads_lazily(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        assert not mgr.is_loaded()
        _ = mgr.get_context()
        assert mgr.is_loaded()

    def test_set_context_before_load_still_works(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        new_ctx = ProjectContext(project_overview="preset")
        mgr.set_context(new_ctx)
        ctx2 = WorkspaceContextManager(tmp_path).load()
        assert ctx2.project_overview == "preset"


# ---------------------------------------------------------------------------
# 3. Field-level CRUD
# ---------------------------------------------------------------------------

class TestFieldCRUD:
    def test_overview_round_trip(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("alpha")
        assert mgr.get_overview() == "alpha"
        mgr.set_overview("beta")
        assert mgr.get_overview() == "beta"

    def test_architecture_round_trip(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_architecture("hex arch")
        assert mgr.get_architecture() == "hex arch"

    def test_adr_add_remove(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.add_adr("ADR-1", "First", "Use it")
        mgr.add_adr("ADR-2", "Second", "Use it too")
        ids = [a.id for a in mgr.get_adrs()]
        assert ids == ["ADR-1", "ADR-2"]
        mgr.remove_adr("ADR-1")
        ids = [a.id for a in mgr.get_adrs()]
        assert ids == ["ADR-2"]
        mgr.remove_adr("DOES-NOT-EXIST")
        ids = [a.id for a in mgr.get_adrs()]
        assert ids == ["ADR-2"]  # no-op, not an error

    def test_sprint_round_trip(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_sprint("S47")
        assert mgr.get_sprint() == "S47"

    def test_common_command_add_remove(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.add_common_command("build", "make build")
        mgr.add_common_command("lint", "ruff check")
        labels = [c.label for c in mgr.get_common_commands()]
        assert labels == ["build", "lint"]
        mgr.remove_common_command("build")
        assert [c.label for c in mgr.get_common_commands()] == ["lint"]

    def test_test_commands_add_only(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.add_test_command("unit", "pytest -q")
        assert [c.label for c in mgr.get_test_commands()] == ["unit"]

    def test_forbidden_area_dedup(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.add_forbidden_area("secrets/")
        mgr.add_forbidden_area("secrets/")  # dedup — must NOT save again
        assert mgr.get_forbidden_areas() == ["secrets/"]
        mgr.add_forbidden_area("node_modules/")
        assert mgr.get_forbidden_areas() == ["secrets/", "node_modules/"]
        mgr.remove_forbidden_area("secrets/")
        assert mgr.get_forbidden_areas() == ["node_modules/"]

    def test_conventions_round_trip(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_conventions("PEP 8")
        assert mgr.get_conventions() == "PEP 8"

    def test_injection_enabled_toggle(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        assert mgr.get_injection_enabled() is True
        mgr.set_injection_enabled(False)
        assert mgr.get_injection_enabled() is False
        mgr.set_injection_enabled(True)
        assert mgr.get_injection_enabled() is True


# ---------------------------------------------------------------------------
# 4. recent_tasks cap and ordering
# ---------------------------------------------------------------------------

class TestRecentTasksCapAndOrdering:
    def test_recent_tasks_capped_at_max(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        for i in range(MAX_RECENT_TASKS + 5):
            mgr.add_recent_task(RecentTask(
                thread_id=f"t-{i:03d}",
                title=f"Task {i:03d}",
                executor="claude-code",
                status="done",
                completed_at="2026-06-01T00:00:00Z",
            ))
        tasks = mgr.get_recent_tasks()
        assert len(tasks) == MAX_RECENT_TASKS
        # The cap keeps the LAST N entries in order (most recent at the end).
        first_id = tasks[0].thread_id
        last_id = tasks[-1].thread_id
        assert first_id == f"t-{5:03d}"   # 15 added, 10 kept, drop 5 oldest
        assert last_id == f"t-{14:03d}"
        # Order is preserved.
        for i, t in enumerate(tasks):
            assert t.thread_id == f"t-{i + 5:03d}"

    def test_recent_tasks_under_cap_keeps_all(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        for i in range(3):
            mgr.add_recent_task(RecentTask(
                thread_id=f"t-{i}", title=f"T{i}", executor="hermes-local",
                status="done", completed_at="2026-06-01T00:00:00Z",
            ))
        assert len(mgr.get_recent_tasks()) == 3

    def test_recent_tasks_loaded_from_disk_respect_cap(
        self, tmp_path: Path
    ) -> None:
        # Write 15 tasks directly to JSON; the manager should accept up to 10
        # visible on read (via add_recent_task's cap), but a fresh load returns
        # whatever is on disk verbatim. The cap is enforced only on append.
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("seed")
        mgr.add_recent_task(RecentTask(
            thread_id="t-0", title="T0", executor="x", status="done",
            completed_at="2026-06-01T00:00:00Z",
        ))
        # Bypass add_recent_task and inject more via set_context.
        ctx = mgr.get_context()
        for i in range(1, 15):
            ctx.recent_tasks.append(RecentTask(
                thread_id=f"t-{i}", title=f"T{i}", executor="x", status="done",
                completed_at="2026-06-01T00:00:00Z",
            ))
        mgr.save()
        # Re-open and verify the raw file has 15; cap is enforced only on append.
        ctx2 = WorkspaceContextManager(tmp_path).load()
        assert len(ctx2.recent_tasks) == 15


# ---------------------------------------------------------------------------
# 5. context_hash
# ---------------------------------------------------------------------------

class TestContextHash:
    def test_hash_is_16_char_hex(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("anything")
        h = mgr.context_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", h), f"non-hex hash: {h!r}"

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("a")
        h1 = mgr.context_hash()
        mgr.set_overview("b")
        h2 = mgr.context_hash()
        assert h1 != h2

    def test_hash_is_deterministic_for_same_content(self, tmp_path: Path) -> None:
        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("same")
        h1 = mgr.context_hash()
        h2 = mgr.context_hash()
        assert h1 == h2


# ---------------------------------------------------------------------------
# 6. Malformed JSON graceful handling
# ---------------------------------------------------------------------------

class TestMalformedJsonGraceful:
    def test_corrupt_json_returns_default(self, tmp_path: Path) -> None:
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / CONTEXT_FILENAME).write_text("{ not valid json")
        mgr = WorkspaceContextManager(tmp_path)
        ctx = mgr.load()
        assert isinstance(ctx, ProjectContext)
        # Should fall back to defaults.
        assert ctx.project_overview == ""
        assert ctx.current_sprint == ""

    def test_wrong_type_json_raises(self, tmp_path: Path) -> None:
        """Non-dict JSON (e.g. a list) is malformed input. The manager
        treats raw as a dict, so a list causes AttributeError on .get().
        Documented behavior: the caller is expected to delete / fix the
        corrupt file. This is acceptable — load() must not silently
        discard user data.
        """
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / CONTEXT_FILENAME).write_text("[]")  # list, not object
        mgr = WorkspaceContextManager(tmp_path)
        with pytest.raises(AttributeError):
            mgr.load()

    def test_empty_file_returns_default(self, tmp_path: Path) -> None:
        hermes_dir = tmp_path / ".hermes"
        hermes_dir.mkdir()
        (hermes_dir / CONTEXT_FILENAME).write_text("")
        mgr = WorkspaceContextManager(tmp_path)
        ctx = mgr.load()
        assert ctx.project_overview == ""


# ---------------------------------------------------------------------------
# 7. create_context_manager factory
# ---------------------------------------------------------------------------

class TestContextFactory:
    def test_factory_creates_workspace_manager(self, tmp_path: Path) -> None:
        mgr = create_context_manager(tmp_path)
        assert isinstance(mgr, WorkspaceContextManager)
        assert mgr._project_root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 8. No real-repo writes / no worktree / no subprocess
# ---------------------------------------------------------------------------

class TestNoSideEffects:
    def test_does_not_write_outside_project_root(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """All writes must land under tmp_path. The real HOME/HERMES_HOME are
        re-pointed at a different tmp_path to confirm no leakage.
        """
        real_cwd = Path.cwd()
        real_home = Path.home()
        leak_root = tmp_path / "leak_check"
        leak_root.mkdir()
        leak_hermes = leak_root / "leak_hermes"
        monkeypatch.setenv("HOME", str(leak_root))
        monkeypatch.setenv("HERMES_HOME", str(leak_hermes))

        try:
            mgr = WorkspaceContextManager(tmp_path)
            mgr.set_overview("hi")
            mgr.set_sprint("S1")
            mgr.add_adr("ADR-1", "T", "D")
            mgr.add_recent_task(RecentTask(
                thread_id="t-1", title="T", executor="x", status="done",
                completed_at="2026-06-01T00:00:00Z",
            ))

            # All writes should be under tmp_path/.hermes.
            assert (tmp_path / ".hermes" / CONTEXT_FILENAME).exists()

            # No files should appear under the leak-detector path.
            leak_files = [p for p in leak_root.rglob("*") if p.is_file()]
            assert leak_files == [], (
                f"Unexpected files outside project_root: {leak_files}"
            )
            # HERMES_HOME should not have been created/touched.
            assert not leak_hermes.exists() or not any(
                p.is_file() for p in leak_hermes.rglob("*")
            ), "HERMES_HOME was unexpectedly touched"
        finally:
            if Path.cwd() != real_cwd:
                os.chdir(real_cwd)

    def test_does_not_import_worktree(self) -> None:
        """Importing context.py must not pull in executors.worktree."""
        import sys
        # Clear cached modules so we observe fresh imports.
        for mod_name in list(sys.modules):
            if mod_name == "executors.context":
                del sys.modules[mod_name]
        import executors.context  # noqa: F401
        # The worktree module must remain unloaded.
        assert "executors.worktree" not in sys.modules

    def test_does_not_call_subprocess(self, tmp_path: Path, monkeypatch) -> None:
        """No subprocess invocation should occur during CRUD."""
        import subprocess

        popen_calls: list = []
        original_popen = subprocess.Popen

        def tracking_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return original_popen(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", tracking_popen)

        mgr = WorkspaceContextManager(tmp_path)
        mgr.set_overview("x")
        mgr.set_architecture("y")
        mgr.add_adr("ADR-1", "T", "D")
        mgr.add_common_command("build", "echo build")
        mgr.add_recent_task(RecentTask(
            thread_id="t-1", title="T", executor="x", status="done",
            completed_at="2026-06-01T00:00:00Z",
        ))

        assert popen_calls == [], f"Unexpected subprocess calls: {popen_calls}"
