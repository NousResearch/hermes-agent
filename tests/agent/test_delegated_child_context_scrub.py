"""Regression tests for #87650 — HERMES_DELEGATED_CHILD_CONTEXT marker symmetric scrubbing.

Ensures:
1. scrub_kanban_env and delegated_child_subprocess_env pop the marker when not in a delegated child.
2. Long-lived process environments do not permanently retain the marker.
3. Dispatcher worker spawn explicitly removes the marker.
4. The marker scrub happens at gateway startup, never at gateway.run import time
   (#87668 review: lazy `import gateway.run` from tool code scrubbed a
   legitimate delegated child's marker).
5. A non-delegated env=None call returns a scrubbed snapshot of the inherited
   environment instead of a bare inherit.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    delegated_child_context,
    delegated_child_subprocess_env,
    is_delegated_child_context,
    is_delegated_child_process_context,
    scrub_kanban_env,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    for k in KANBAN_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_scrub_kanban_env_symmetry():
    """scrub_kanban_env removes the marker when is_delegated is False."""
    env = {
        "HERMES_KANBAN_TASK": "task-123",
        "HERMES_KANBAN_DB": "/path/to/db",
        DELEGATED_CHILD_ENV_MARKER: "1",
        "PATH": "/usr/bin",
    }
    # Delegated scrub removes kanban keys and sets marker
    scrubbed_del = scrub_kanban_env(env, is_delegated=True)
    assert "HERMES_KANBAN_TASK" not in scrubbed_del
    assert scrubbed_del[DELEGATED_CHILD_ENV_MARKER] == "1"

    # Non-delegated scrub removes both kanban keys and marker
    scrubbed_non = scrub_kanban_env(env, is_delegated=False)
    assert "HERMES_KANBAN_TASK" not in scrubbed_non
    assert DELEGATED_CHILD_ENV_MARKER not in scrubbed_non
    assert scrubbed_non["PATH"] == "/usr/bin"


def test_delegated_child_subprocess_env_cleans_non_delegated(monkeypatch):
    """When not in a delegated child, delegated_child_subprocess_env strips stale marker from env."""
    assert is_delegated_child_process_context() is False

    stale_env = {
        DELEGATED_CHILD_ENV_MARKER: "1",
        "OTHER_VAR": "abc",
    }
    result = delegated_child_subprocess_env(stale_env)
    assert result is not None
    assert DELEGATED_CHILD_ENV_MARKER not in result
    assert result["OTHER_VAR"] == "abc"


def test_delegated_child_context_lifecycle():
    """Inside delegated_child_context(), is_delegated_child_context() is True and resets upon exit."""
    assert is_delegated_child_context() is False

    with delegated_child_context("test-session"):
        assert is_delegated_child_context() is True
        env = delegated_child_subprocess_env({"CUSTOM": "1"})
        assert env[DELEGATED_CHILD_ENV_MARKER] == "1"

    assert is_delegated_child_context() is False


def test_delegated_child_subprocess_env_none_returns_snapshot(monkeypatch):
    """Non-delegated no-arg call returns a scrubbed copy of os.environ.

    Callers such as ``tools/code_execution_tool.py`` spawn with
    ``delegated_child_subprocess_env()`` (no arguments).  Returning ``None``
    meant plain inheritance, so any marker that leaked into ``os.environ``
    crossed every subprocess boundary (#87650 review, point 2).
    """
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    result = delegated_child_subprocess_env()
    assert result is not None
    assert DELEGATED_CHILD_ENV_MARKER not in result
    assert result["PATH"] == "/usr/bin:/bin"
    # The helper must never mutate the live process environment.
    assert os.environ.get("PATH") == "/usr/bin:/bin"


def test_delegated_child_subprocess_env_none_strips_stale_marker(monkeypatch):
    """A stale inherited marker never survives the env=None path."""
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    # Predicate mocked to False to exercise the non-delegated branch in
    # isolation (a real process with the marker in os.environ classifies as
    # delegated; this branch guards callers that decide delegation elsewhere).
    monkeypatch.setattr(
        "agent.delegation_context.is_delegated_child_process_context",
        lambda: False,
    )
    result = delegated_child_subprocess_env()
    assert result is not None
    assert DELEGATED_CHILD_ENV_MARKER not in result


def test_gateway_run_import_preserves_delegated_child_marker():
    """Importing gateway.run must not scrub the marker (#87668 review, point 1).

    Tool code (send_message_tool, telegram adapter, relay runtime) lazily
    imports gateway.run inside ordinary agent processes; a module-level pop
    stripped a legitimate delegated child's marker on import.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import os, sys; "
        f"os.environ[{DELEGATED_CHILD_ENV_MARKER!r}] = '1'; "
        "import gateway.run; "
        f"sys.stdout.write(os.environ.get({DELEGATED_CHILD_ENV_MARKER!r}, ''))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "1"


def test_start_gateway_scrubs_stale_marker_at_startup(monkeypatch):
    """The symmetric-scrub contract holds at real gateway startup."""
    for var in ("HERMES_EXEC_ASK", "AI_AGENT", "HERMES_AGENT", "_HERMES_GATEWAY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")

    gateway_run = pytest.importorskip("gateway.run")

    import asyncio

    import hermes_cli.resource_limits as resource_limits

    def _stop_startup():
        raise RuntimeError("startup-stop")

    # Abort start_gateway immediately after the scrub under test.
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", _stop_startup)

    with pytest.raises(RuntimeError, match="startup-stop"):
        asyncio.run(gateway_run.start_gateway())

    assert DELEGATED_CHILD_ENV_MARKER not in os.environ
