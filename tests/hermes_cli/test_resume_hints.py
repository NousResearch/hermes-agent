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
    assert build_continue_command("demo title") == "hermes -p coder -c 'demo title'"
    assert build_continue_command("demo title", tui=True) == "hermes -p coder --tui -c 'demo title'"


def test_resume_command_named_custom_profile_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "custom"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes -p custom --resume sess123"


def test_resume_command_unrecognized_custom_home_includes_hermes_home_env(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    custom_home = tmp_path / "custom home"
    custom_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_continue_command, build_resume_command

    env_prefix = f"HERMES_HOME='{custom_home}'"
    assert build_resume_command("sess123") == f"{env_prefix} hermes --resume sess123"
    assert build_resume_command("sess123", tui=True) == f"{env_prefix} hermes --tui --resume sess123"
    assert build_continue_command("demo title") == f"{env_prefix} hermes -c 'demo title'"


def test_resume_command_custom_root_named_profile_adds_profile_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / "opt" / "data" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_resume_command

    assert build_resume_command("sess123") == "hermes -p coder --resume sess123"


def test_continue_command_shell_quotes_expansion_and_history_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_modules()

    from hermes_cli.resume_hints import build_continue_command

    title = "say 'hi' $HOME `whoami` $(date)!\nnext"
    command = build_continue_command(title)

    assert command == "hermes -p coder -c 'say '\"'\"'hi'\"'\"' $HOME `whoami` $(date)! next'"
