"""Tests for ``remove_path_from_shell_configs`` — the uninstaller's shell-rc rewrite.

This rewrites files Hermes does not own (``~/.bashrc``, ``~/.zshrc``, ...) and
takes no backup of them, so the rewrite has to be atomic: a bare
``write_text()`` truncates the rc file before the new content lands, and the
caller wraps everything in ``except Exception: log_warn(...)``, so a partial
write is downgraded to a warning and the user's next login starts a bare shell.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hermes_cli import uninstall


ZSHRC = (
    "export EDITOR=vim\n"
    "alias ll='ls -la'\n"
    "\n"
    "# Hermes Agent\n"
    'export PATH="$HOME/.local/bin:$PATH"\n'
    "\n"
    "source ~/.work-profile\n"
)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both ``Path.home()`` and ``HERMES_HOME`` at a throwaway dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    return home


class TestHappyPath:
    def test_hermes_path_block_is_removed(self, fake_home: Path):
        rc = fake_home / ".zshrc"
        rc.write_text(ZSHRC, encoding="utf-8")

        removed = uninstall.remove_path_from_shell_configs()

        assert removed == [rc]
        text = rc.read_text(encoding="utf-8")
        assert "# Hermes Agent" not in text
        # The user's own lines are untouched.
        assert "export EDITOR=vim" in text
        assert "source ~/.work-profile" in text

    def test_untouched_rc_is_not_reported(self, fake_home: Path):
        rc = fake_home / ".zshrc"
        rc.write_text("export EDITOR=vim\n", encoding="utf-8")

        assert uninstall.remove_path_from_shell_configs() == []
        assert rc.read_text(encoding="utf-8") == "export EDITOR=vim\n"


class TestCrashDurability:
    def test_shell_config_survives_an_interrupted_rewrite(self, fake_home: Path):
        """An interrupted rewrite must leave the rc file byte-identical.

        There is no backup of the user's shell rc anywhere in this code path,
        so a truncated write is unrecoverable.
        """
        rc = fake_home / ".zshrc"
        rc.write_text(ZSHRC, encoding="utf-8")
        original = rc.read_bytes()

        def boom(fd):
            raise OSError("simulated crash mid-write")

        # Scoped context so restoring os.fsync doesn't also undo the
        # Path.home()/HERMES_HOME patches the fake_home fixture installed.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "fsync", boom)
            removed = uninstall.remove_path_from_shell_configs()

        # The write failed, so the rc must not be reported as modified...
        assert removed == []
        # ...and it must still be exactly what the user had.
        assert rc.read_bytes() == original
        # The aborted write must not leave a temp file behind in $HOME.
        assert list(fake_home.glob("*.tmp")) == []

    def test_symlinked_shell_config_stays_a_symlink(self, fake_home: Path):
        """A dotfiles-repo ``~/.zshrc`` is a symlink; replacing it with a
        regular file silently detaches the user's dotfiles."""
        dotfiles = fake_home / "dotfiles"
        dotfiles.mkdir()
        real = dotfiles / "zshrc"
        real.write_text(ZSHRC, encoding="utf-8")
        rc = fake_home / ".zshrc"
        rc.symlink_to(real)

        removed = uninstall.remove_path_from_shell_configs()

        assert removed == [rc]
        assert rc.is_symlink(), "the symlink was replaced by a regular file"
        assert "# Hermes Agent" not in real.read_text(encoding="utf-8")
        assert "export EDITOR=vim" in real.read_text(encoding="utf-8")

    def test_existing_file_mode_is_preserved(self, fake_home: Path):
        """Shell rc files are normally 0644; uninstalling must not change that."""
        rc = fake_home / ".zshrc"
        rc.write_text(ZSHRC, encoding="utf-8")
        os.chmod(rc, 0o644)

        uninstall.remove_path_from_shell_configs()

        mode = stat.S_IMODE(rc.stat().st_mode)
        assert mode == 0o644, f"mode changed to {oct(mode)}"
