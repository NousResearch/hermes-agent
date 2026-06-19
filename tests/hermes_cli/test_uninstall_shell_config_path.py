"""Tests for hermes_cli.uninstall.remove_path_from_shell_configs.

The POSIX installer (scripts/install.sh) appends a ``# Hermes Agent`` comment
followed by a GENERIC PATH line to the user's shell rc file, e.g.::

    # Hermes Agent — ensure ~/.local/bin is on PATH
    export PATH="$HOME/.local/bin:$PATH"

The export line carries no ``hermes`` substring, so the old removal heuristic
(which required ``'hermes' in line``) deleted only the comment and left the
``export PATH`` line orphaned — accumulating on every install/uninstall cycle
while the function still reported the rc file "updated". Uninstall must strip
the PATH line that sits directly under our comment too.
"""

from pathlib import Path

import pytest

import hermes_cli.uninstall as uninstall


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() so find_shell_configs() reads our temp rc files."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_strips_orphaned_local_bin_export(fake_home):
    """The `~/.local/bin` installer block (comment + bare export PATH) is gone."""
    zshrc = fake_home / ".zshrc"
    zshrc.write_text(
        "export EDITOR=vim\n"
        "\n"
        "# Hermes Agent — ensure ~/.local/bin is on PATH\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
    )

    uninstall.remove_path_from_shell_configs()

    content = zshrc.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' not in content
    assert "Hermes Agent" not in content
    # Unrelated user config must survive untouched.
    assert "export EDITOR=vim" in content


def test_strips_orphaned_usr_local_bin_export(fake_home):
    """The RHEL `/usr/local/bin` variant is stripped too."""
    bashrc = fake_home / ".bashrc"
    bashrc.write_text(
        "# Hermes Agent — ensure /usr/local/bin is on PATH (RHEL non-login shells)\n"
        'export PATH="/usr/local/bin:$PATH"\n'
        "export EDITOR=nano\n"
    )

    uninstall.remove_path_from_shell_configs()

    content = bashrc.read_text()
    assert 'export PATH="/usr/local/bin:$PATH"' not in content
    assert "Hermes Agent" not in content
    assert "export EDITOR=nano" in content


def test_leaves_unrelated_path_export_untouched(fake_home):
    """A user's own non-Hermes PATH export (no preceding comment) survives."""
    zshrc = fake_home / ".zshrc"
    zshrc.write_text(
        'export PATH="$HOME/mybin:$PATH"\n'
        "export EDITOR=vim\n"
    )

    uninstall.remove_path_from_shell_configs()

    content = zshrc.read_text()
    assert 'export PATH="$HOME/mybin:$PATH"' in content
    assert "export EDITOR=vim" in content
