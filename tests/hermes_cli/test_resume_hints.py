"""Tests for profile-aware resume hint command builders."""

from __future__ import annotations

import sys
from pathlib import Path


def _reset_modules():
    for name in list(sys.modules):
        if name in {"hermes_constants", "hermes_cli.profiles", "hermes_cli.resume_hints"}:
            sys.modules.pop(name, None)


def test_resume_command_default_profile_has_no_profile_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes --resume sess123"
    assert build_resume_command("sess123", tui=True) == "hermes --tui --resume sess123"


def test_resume_command_named_profile_adds_global_profile_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_continue_command, build_resume_command

    assert build_resume_command("sess123") == "hermes -p coder --resume sess123"
    assert build_resume_command("sess123", tui=True) == "hermes -p coder --tui --resume sess123"
    assert build_continue_command("demo title") == 'hermes -p coder -c "demo title"'
    assert build_continue_command("demo title", tui=True) == 'hermes -p coder --tui -c "demo title"'


def test_resume_command_named_custom_profile_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "custom"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes -p custom --resume sess123"


def test_resume_command_unrecognized_custom_home_falls_back_to_plain_command(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes --resume sess123"


def test_resume_command_custom_root_named_profile_adds_profile_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / "opt" / "data" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes -p coder --resume sess123"


def test_continue_command_escapes_double_quoted_shell_expansions(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_continue_command

    command = build_continue_command(r'say "hi" \ $HOME `whoami` $(date)')

    assert command.startswith('hermes -p coder -c "')
    assert r'\"hi\"' in command
    assert r'\\' in command
    assert r'\$HOME' in command
    assert r'\`whoami\`' in command
    assert r'\$(date)' in command
