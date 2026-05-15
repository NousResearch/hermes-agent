"""Regression tests for profile-aware classic CLI resume hints."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace


class _FakeSessionDB:
    def __init__(self, title: str | None = "demo title"):
        self.title = title

    def get_session_title(self, _session_id: str) -> str | None:
        return self.title


def _reset_profile_modules():
    for name in list(sys.modules):
        if name in {"cli", "hermes_constants", "hermes_cli.profiles", "hermes_cli.resume_hints"}:
            sys.modules.pop(name, None)


def _stub_cli(session_id: str = "20260409_000001_abc123", title: str | None = "demo title"):
    return SimpleNamespace(
        conversation_history=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        session_id=session_id,
        session_start=datetime.now() - timedelta(seconds=3),
        _session_db=_FakeSessionDB(title),
    )


def test_exit_summary_default_profile_keeps_existing_resume_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    _reset_profile_modules()

    import cli

    cli.HermesCLI._print_exit_summary(_stub_cli())
    out = capsys.readouterr().out

    assert "  hermes --resume 20260409_000001_abc123" in out
    assert '  hermes -c "demo title"' in out
    assert "hermes -p" not in out


def test_exit_summary_named_profile_includes_profile_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_profile_modules()

    import cli

    cli.HermesCLI._print_exit_summary(_stub_cli())
    out = capsys.readouterr().out

    assert "  hermes -p coder --resume 20260409_000001_abc123" in out
    assert '  hermes -p coder -c "demo title"' in out


def test_exit_summary_named_custom_profile_is_not_treated_as_unknown_custom_home(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = tmp_path / ".hermes" / "profiles" / "custom"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _reset_profile_modules()

    import cli

    cli.HermesCLI._print_exit_summary(_stub_cli())
    out = capsys.readouterr().out

    assert "  hermes -p custom --resume 20260409_000001_abc123" in out
    assert '  hermes -p custom -c "demo title"' in out


def test_exit_summary_custom_home_falls_back_to_plain_resume_hint(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    custom_home = tmp_path / "standalone-hermes-home"
    custom_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    _reset_profile_modules()

    import cli

    cli.HermesCLI._print_exit_summary(_stub_cli())
    out = capsys.readouterr().out

    assert "  hermes --resume 20260409_000001_abc123" in out
    assert '  hermes -c "demo title"' in out
    assert "hermes -p" not in out
