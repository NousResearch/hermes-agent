"""Gateway owner boundaries clear stale lineage; ordinary descendants retain their write fence."""

from __future__ import annotations

import os

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


def test_scrub_kanban_env_preserves_descendant_fence():
    """Worker identity is removed while board routing and the write fence survive."""
    env = {
        "HERMES_KANBAN_TASK": "task-123",
        "HERMES_KANBAN_DB": "/path/to/db",
        DELEGATED_CHILD_ENV_MARKER: "1",
        "PATH": "/usr/bin",
    }
    # Delegated scrub removes kanban keys and sets marker
    scrubbed_del = scrub_kanban_env(env)
    assert "HERMES_KANBAN_TASK" not in scrubbed_del
    assert scrubbed_del[DELEGATED_CHILD_ENV_MARKER] == "1"

    assert scrubbed_del["HERMES_KANBAN_DB"] == "/path/to/db"
    assert scrubbed_del["PATH"] == "/usr/bin"
    assert env["HERMES_KANBAN_TASK"] == "task-123"


def test_delegated_child_subprocess_env_preserves_explicit_fence():
    """An explicit child environment carries a write fence even outside a local child context."""
    assert is_delegated_child_process_context() is False

    stale_env = {
        DELEGATED_CHILD_ENV_MARKER: "1",
        "OTHER_VAR": "abc",
    }
    result = delegated_child_subprocess_env(stale_env)
    assert result is not None
    assert result[DELEGATED_CHILD_ENV_MARKER] == "1"
    assert result["OTHER_VAR"] == "abc"


def test_delegated_child_context_lifecycle():
    """Inside delegated_child_context(), is_delegated_child_context() is True and resets upon exit."""
    assert is_delegated_child_context() is False

    with delegated_child_context("test-session"):
        assert is_delegated_child_context() is True
        env = delegated_child_subprocess_env({"CUSTOM": "1"})
        assert env[DELEGATED_CHILD_ENV_MARKER] == "1"

    assert is_delegated_child_context() is False


def test_delegated_child_subprocess_env_none_keeps_plain_inheritance(monkeypatch):
    """An ordinary process without a fence keeps the upstream env=None contract."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert delegated_child_subprocess_env() is None
    assert os.environ["PATH"] == "/usr/bin:/bin"


def test_delegated_child_subprocess_env_none_preserves_inherited_fence(monkeypatch):
    """A real inherited marker survives the next spawn without mocking its predicate."""
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    assert is_delegated_child_process_context() is True
    assert is_delegated_child_process_context() is True
    result = delegated_child_subprocess_env()
    assert result is not None
    assert result[DELEGATED_CHILD_ENV_MARKER] == "1"
    assert os.environ[DELEGATED_CHILD_ENV_MARKER] == "1"


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

    import gateway.run as gateway_run

    import asyncio

    import hermes_cli.resource_limits as resource_limits

    def _stop_startup():
        raise RuntimeError("startup-stop")

    # Abort start_gateway immediately after the scrub under test.
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", _stop_startup)

    with pytest.raises(RuntimeError, match="startup-stop"):
        asyncio.run(gateway_run.start_gateway())

    assert DELEGATED_CHILD_ENV_MARKER not in os.environ


def test_restart_watcher_clears_owner_markers_without_mutating_parent(monkeypatch):
    from gateway.run_shutdown import GatewayShutdownMixin

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    watcher_env = GatewayShutdownMixin._restart_watcher_env()
    assert DELEGATED_CHILD_ENV_MARKER not in watcher_env
    assert "_HERMES_GATEWAY" not in watcher_env
    assert watcher_env["PATH"] == "/usr/bin:/bin"
    assert os.environ[DELEGATED_CHILD_ENV_MARKER] == "1"
    assert os.environ["_HERMES_GATEWAY"] == "1"
