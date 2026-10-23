"""Pytest configuration for the desired-state skill test suite.

Adds `scripts/` to sys.path so tests can `from _common import ...`, and
provides a tmp store root plus a pinned clock for deterministic timestamps.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An isolated desired-state store root under tmp_path."""
    return tmp_path / "desired"


@pytest.fixture
def now() -> datetime:
    """A fixed 'now' so created_at/updated_at and pace are deterministic."""
    return datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
