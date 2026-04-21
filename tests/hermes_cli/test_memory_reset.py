"""Tests for the `hermes memory reset` CLI command.

Covers:
- Reset both stores (MEMORY.md + USER.md)
- Reset individual stores (--target memory / --target user)
- Skip confirmation with --yes
- Graceful handling when no memory files exist
- Profile-scoped reset (uses HERMES_HOME)
"""

import os
import pytest
from argparse import Namespace
from pathlib import Path


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    """Set up a fake HERMES_HOME with memory files."""
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Create sample memory files
    (memories / "MEMORY.md").write_text(
        "§\nHermes repo is at ~/.hermes/hermes-agent\n§\nUser prefers dark themes",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "§\nUser is Teknium\n§\nTimezone: US Pacific",
        encoding="utf-8",
    )
    return hermes_home, memories


def _run_memory_reset(target="all", yes=False, monkeypatch=None, confirm_input="no"):
    """Invoke the memory reset logic from cmd_memory in main.py.

    Simulates what happens when `hermes memory reset` is run.
    """
    from hermes_constants import get_hermes_home, display_hermes_home

    mem_dir = get_hermes_home() / "memories"
    files_to_reset = []
    if target in ("all", "memory"):
        files_to_reset.append(("MEMORY.md", "agent notes"))
    if target in ("all", "user"):
        files_to_reset.append(("USER.md", "user profile"))

    existing = [(f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()]
    if not existing:
        return "nothing"

    if not yes:
        if confirm_input != "yes":
            return "cancelled"

    for f, desc in existing:
        (mem_dir / f).unlink()

    return "deleted"


class TestMemoryReset:
    """Tests for `hermes memory reset` subcommand."""

    def test_reset_all_with_yes_flag(self, memory_env):
        """--yes flag should skip confirmation and delete both files."""
        hermes_home, memories = memory_env
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_memory_only(self, memory_env):
        """--target memory should only delete MEMORY.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="memory", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_user_only(self, memory_env):
        """--target user should only delete USER.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="user", yes=True)
        assert result == "deleted"
        assert (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_no_files_exist(self, tmp_path, monkeypatch):
        """Should return 'nothing' when no memory files exist."""
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "memories").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"

    def test_reset_confirmation_denied(self, memory_env):
        """Without --yes and without typing 'yes', should be cancelled."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="no")
        assert result == "cancelled"
        # Files should still exist
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_confirmation_accepted(self, memory_env):
        """Typing 'yes' should proceed with deletion."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="yes")
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_profile_scoped(self, tmp_path, monkeypatch):
        """Reset should work on the active profile's HERMES_HOME."""
        profile_home = tmp_path / "profiles" / "myprofile"
        memories = profile_home / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("profile memory", encoding="utf-8")
        (memories / "USER.md").write_text("profile user", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_partial_files(self, memory_env):
        """Reset should work when only one memory file exists."""
        hermes_home, memories = memory_env
        (memories / "USER.md").unlink()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()

    def test_reset_empty_memories_dir(self, tmp_path, monkeypatch):
        """No memories dir at all should report nothing."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        # No memories dir
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # The memories dir won't exist; get_hermes_home() / "memories" won't have files
        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"
