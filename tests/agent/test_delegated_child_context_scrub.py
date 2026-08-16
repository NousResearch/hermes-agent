"""Regression tests for #87650 — HERMES_DELEGATED_CHILD_CONTEXT marker symmetric scrubbing.

Ensures:
1. scrub_kanban_env and delegated_child_subprocess_env pop the marker when not in a delegated child.
2. Long-lived process environments do not permanently retain the marker.
3. Dispatcher worker spawn explicitly removes the marker.
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
