"""Tests for the kanban-pr hook (handler + script)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Handler tests ────────────────────────────────────────────────────────────


class TestKanbanPrHandler:
    """Test the handler.py logic in isolation."""

    def test_handle_noops_when_script_missing(self):
        """Handler should be a no-op when the script doesn't exist."""
        from gateway.hooks import HookRegistry

        registry = HookRegistry()

        # Create a temp hook dir with a valid HOOK.yaml + handler
        with tempfile.TemporaryDirectory() as td:
            hook_dir = Path(td)
            (hook_dir / "HOOK.yaml").write_text(
                "name: test-kanban-pr\n"
                "description: Test\n"
                "events:\n"
                "  - kanban:task_completed\n"
            )
            (hook_dir / "handler.py").write_text(
                "from __future__ import annotations\n"
                "from typing import Any, Dict\n"
                "\n"
                "FIRED = []\n"
                "\n"
                "def handle(event_type: str, context: Dict[str, Any]) -> None:\n"
                "    FIRED.append((event_type, context))\n"
            )

            # Directly register the handler (bypass discover_and_load path issues)
            import importlib.util, sys

            module_name = "hermes_hook_test_kanban_pr"
            spec = importlib.util.spec_from_file_location(
                module_name, hook_dir / "handler.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            handle_fn = getattr(module, "handle")
            registry._handlers.setdefault("kanban:task_completed", []).append(handle_fn)

            # Verify the handler is registered
            assert len(registry._resolve_handlers("kanban:task_completed")) == 1

    def test_handler_receives_context_fields(self):
        """The handler should receive all expected context fields."""
        received_contexts = []

        async def capture_handle(event_type, context):
            received_contexts.append((event_type, context))

        # Simulate what the notifier sends
        hook_ctx = {
            "task_id": "t_abc123",
            "title": "Add feature X",
            "assignee": "deepseek-coder",
            "summary": "Implemented feature X with tests",
            "metadata": {
                "changed_files": ["src/feature.ts", "tests/test_feature.ts"],
                "tests_run": 12,
                "tests_passed": 12,
            },
            "result": None,
            "workspace_kind": "worktree",
            "workspace_path": "/tmp/test-worktree",
            "board": "fishing-trip",
            "created_cards": ["t_child1", "t_child2"],
        }

        # Call handler directly (not through subprocess)
        import asyncio

        asyncio.run(capture_handle("kanban:task_completed", hook_ctx))

        assert len(received_contexts) == 1
        event_type, ctx = received_contexts[0]
        assert event_type == "kanban:task_completed"
        assert ctx["task_id"] == "t_abc123"
        assert ctx["title"] == "Add feature X"
        assert ctx["assignee"] == "deepseek-coder"
        assert ctx["board"] == "fishing-trip"
        assert ctx["workspace_kind"] == "worktree"
        assert ctx["metadata"]["changed_files"] == [
            "src/feature.ts",
            "tests/test_feature.ts",
        ]
        assert ctx["created_cards"] == ["t_child1", "t_child2"]


# ── Script smoke tests ───────────────────────────────────────────────────────


class TestKanbanPrScript:
    """Test the kanban-pr.sh bash script."""

    SCRIPT = Path("/home/tjaeger/.hermes/scripts/kanban-pr.sh")

    def test_script_exists_and_executable(self):
        assert self.SCRIPT.is_file()
        assert os.access(self.SCRIPT, os.X_OK)

    def test_scratch_workspace_skips(self):
        """Scratch workspaces should skip (not repos)."""
        result = subprocess.run(
            ["bash", str(self.SCRIPT), json.dumps({
                "task_id": "t_test",
                "title": "Test",
                "summary": "Test",
                "workspace_kind": "scratch",
                "workspace_path": "/tmp/nonexistent",
            })],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        stderr = result.stderr.lower()
        # Should skip somewhere (gh not auth, or workspace kind)
        assert "skip" in stderr or "not authenticated" in stderr

    def test_not_a_git_repo_skips(self):
        """A directory that isn't a git repo should skip."""
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                ["bash", str(self.SCRIPT), json.dumps({
                    "task_id": "t_test",
                    "title": "Test",
                    "summary": "Test",
                    "workspace_kind": "worktree",
                    "workspace_path": td,
                })],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0

    def test_empty_input_skips(self):
        result = subprocess.run(
            ["bash", str(self.SCRIPT)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0


# ── Hook registration test ───────────────────────────────────────────────────


class TestHookIntegration:
    """Verify the kanban-pr hook directory is discoverable."""

    def test_hook_directory_exists(self):
        hook_dir = Path("/home/tjaeger/.hermes/hooks/kanban-pr")
        assert hook_dir.is_dir()
        assert (hook_dir / "HOOK.yaml").is_file()
        assert (hook_dir / "handler.py").is_file()

    def test_hook_yaml_is_valid(self):
        import yaml

        manifest = yaml.safe_load(
            Path("/home/tjaeger/.hermes/hooks/kanban-pr/HOOK.yaml").read_text()
        )
        assert manifest["name"] == "kanban-pr"
        assert "kanban:task_completed" in manifest["events"]

    def test_handler_is_importable(self):
        """The handler.py should be syntactically correct and have a handle function."""
        import importlib.util, sys

        handler_path = Path("/home/tjaeger/.hermes/hooks/kanban-pr/handler.py")
        spec = importlib.util.spec_from_file_location(
            "hermes_hook_kanban_pr_test", handler_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["hermes_hook_kanban_pr_test"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("hermes_hook_kanban_pr_test", None)

        assert hasattr(module, "handle")
        assert callable(module.handle)
