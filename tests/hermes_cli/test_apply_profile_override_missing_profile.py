from __future__ import annotations

import os
import sys
from pathlib import Path


def test_stale_profile_env_is_ignored_instead_of_recreating_profile(
    tmp_path, monkeypatch, capsys
):
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    missing_profile = hermes_root / "profiles" / "ghost"

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(missing_profile))
    monkeypatch.setattr(sys, "argv", ["hermes", "profile", "list"])

    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()

    assert os.environ.get("HERMES_HOME") is None
    assert not missing_profile.exists()
    assert "ignoring stale HERMES_HOME" in capsys.readouterr().err


def test_stale_active_profile_is_ignored_instead_of_exiting(
    tmp_path, monkeypatch, capsys
):
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)
    (hermes_root / "active_profile").write_text("ghost\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "profile", "list"])

    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()

    assert os.environ.get("HERMES_HOME") is None
    assert not (hermes_root / "profiles" / "ghost").exists()
    assert "ignoring stale active_profile" in capsys.readouterr().err
