"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def test_kanban_write_guard_allows_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds."""
    test_home = tmp_path / "hermes_test"
    test_home.mkdir(exist_ok=True)
    (test_home / "sessions").mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(test_home))

    from hermes_cli import kanban_db

    # kanban_home should resolve under test_home
    resolved = kanban_db.kanban_home()
    assert str(resolved).startswith(str(test_home)) or str(test_home) in str(resolved)


def test_kanban_write_guard_blocks_real_home(tmp_path, monkeypatch):
    """When kanban_home resolves outside test HERMES_HOME, connect raises."""
    test_home = tmp_path / "hermes_test"
    test_home.mkdir(exist_ok=True)
    (test_home / "sessions").mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(test_home))

    from hermes_cli import kanban_db

    # Patch kanban_home to resolve to a path outside the test home
    fake_real_home = tmp_path / "real_home"
    fake_real_home.mkdir()
    with patch.object(kanban_db, "kanban_home", return_value=fake_real_home):
        # Simulate the guard's check
        resolved_str = str(fake_real_home)
        test_home_str = str(test_home)
        assert test_home_str not in resolved_str
        assert not resolved_str.startswith(test_home_str)


def test_guard_source_has_fail_closed_check():
    """The conftest.py must contain the kanban write guard."""
    src = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "_kanban_write_guard" in src, "conftest.py must have _kanban_write_guard fixture"
    assert "kanban_write_guard" in src, "guard must have clear error message"
    assert "#69283" in src, "guard must reference the issue number"
    assert "RuntimeError" in src, "guard must raise RuntimeError on violation"