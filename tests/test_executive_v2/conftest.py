"""Shared fixtures for Executive v2 tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def in_memory_storage():
    """In-memory ObjectiveStateStorage for tests (no state.db, no
    SessionDB)."""
    from agent.executive.state_storage import ObjectiveStateStorage

    state: dict[str, str] = {}

    class FakeDB:
        def set_meta(self, k, v):
            state[k] = v

        def get_meta(self, k):
            return state.get(k)

        def delete_meta(self, k):
            state.pop(k, None)

        def list_meta_keys(self, prefix=None):
            if prefix is None:
                return list(state.keys())
            return [k for k in state.keys() if k.startswith(prefix)]

        def close(self):
            pass

    return ObjectiveStateStorage(db_factory=lambda: FakeDB())


@pytest.fixture
def clean_env_executive(monkeypatch):
    """Ensure HERMES_EXECUTIVE_V2_ENABLED is unset unless the test sets it."""
    monkeypatch.delenv("HERMES_EXECUTIVE_V2_ENABLED", raising=False)


@pytest.fixture
def agent_stub():
    """Stub agent with default-off Executive v2."""
    a = MagicMock()
    a._executive_v2_enabled = False
    return a
