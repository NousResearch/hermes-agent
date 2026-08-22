"""Regression tests for ``hermes config edit`` editor handling.

``edit_config()`` historically passed the whole ``$EDITOR``/``$VISUAL`` string
as a single argv element, so the extremely common ``EDITOR="code --wait"``
(or ``vim -f``) spelled a nonexistent executable and crashed with
``FileNotFoundError`` / WinError 2. The fix tokenizes the editor string with
``split_command_line`` (Windows-safe, unlike bare ``shlex.split``) and appends
the config path as the final argument.
"""

import os
import sys
from unittest import mock

import pytest

from hermes_cli import config as config_mod
from hermes_cli._subprocess_compat import split_command_line


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config_mod, "is_managed", lambda: False)
    return home


def _run_edit(monkeypatch, editor_value):
    calls = []
    def fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return mock.Mock(returncode=0)

    monkeypatch.setattr(os, "getenv", lambda k, d=None: editor_value if k == "EDITOR" else d)
    monkeypatch.setattr(config_mod.subprocess, "run", fake_run)
    config_mod.edit_config()
    return calls


def test_editor_with_arguments(isolated_home, monkeypatch):
    """EDITOR="code --wait" must arrive as two tokens, not one."""
    calls = _run_edit(monkeypatch, "code --wait")
    assert calls, "subprocess.run was never invoked"
    argv = calls[0]
    assert argv[:-1] == ["code", "--wait"], f"expected tokenized editor, got {argv}"
    # The config path is the final argument.
    assert argv[-1].endswith(os.sep + "config.yaml") or argv[-1].endswith("config.yaml")


def test_editor_plain(isolated_home, monkeypatch):
    """A bare editor name keeps working exactly as before."""
    calls = _run_edit(monkeypatch, "vim")
    assert calls[0][:-1] == ["vim"]


def test_editor_quoted_path_with_spaces(isolated_home, monkeypatch):
    """A quoted editor path containing spaces survives tokenization."""
    editor = '"C:\\Program Files\\My Editor\\editor.exe" -w'
    calls = _run_edit(monkeypatch, editor)
    argv = calls[0]
    assert argv[0] == "C:\\Program Files\\My Editor\\editor.exe", (
        f"quoted path mangled on {sys.platform}: {argv}"
    )
    assert argv[1] == "-w"


def test_editor_unbalanced_quotes_falls_back_to_literal(isolated_home, monkeypatch):
    """A malformed editor string still launches rather than raising."""
    calls = _run_edit(monkeypatch, 'editor "unclosed')
    assert calls, "unbalanced quotes should fall back to the literal string"


def test_split_command_line_windows_paths_preserved():
    """The tokenizer must not eat backslashes (why shlex.split is not used)."""
    tokens = split_command_line('C:\\Editors\\me.exe --flag "C:\\a b.txt"')
    assert tokens[0] == "C:\\Editors\\me.exe"
    assert tokens[1] == "--flag"
    if sys.platform == "win32":
        assert tokens[2] == "C:\\a b.txt"
