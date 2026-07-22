"""#69216: uv.exe fallback search when astral installer places binary in
a non-standard location."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


def test_find_uv_fallback_returns_none_on_posix():
    """On non-Windows, _find_uv_fallback must return None."""
    with patch("platform.system", return_value="Linux"):
        from hermes_cli.managed_uv import _find_uv_fallback
        assert _find_uv_fallback() is None


def test_find_uv_fallback_finds_userprofile_local_bin(tmp_path, monkeypatch):
    """When uv.exe is in ~/.local/bin, the fallback finds it."""
    fake_home = tmp_path / "fakehome"
    fake_local = tmp_path / "fakeappdata"
    uv_dir = fake_home / ".local" / "bin"
    uv_dir.mkdir(parents=True)
    (uv_dir / "uv.exe").write_bytes(b"fake uv binary")

    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))

    with patch("platform.system", return_value="Windows"), \
         patch("hermes_cli.managed_uv.Path.home", return_value=fake_home):
        from hermes_cli.managed_uv import _find_uv_fallback
        result = _find_uv_fallback()
        assert result is not None
        assert result.endswith("uv.exe")
        assert ".local" in result


def test_find_uv_fallback_finds_astral_bin(tmp_path, monkeypatch):
    """When uv.exe is in $LOCALAPPDATA/astral/bin, the fallback finds it."""
    fake_home = tmp_path / "fakehome"
    fake_local = tmp_path / "fakeappdata"
    astral_dir = fake_local / "astral" / "bin"
    astral_dir.mkdir(parents=True)
    (astral_dir / "uv.exe").write_bytes(b"fake uv binary")

    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))

    # No uv in ~/.local/bin this time
    with patch("platform.system", return_value="Windows"), \
         patch("hermes_cli.managed_uv.Path.home", return_value=fake_home):
        from hermes_cli.managed_uv import _find_uv_fallback
        result = _find_uv_fallback()
        assert result is not None
        assert "astral" in result


def test_find_uv_fallback_returns_none_when_not_found(tmp_path, monkeypatch):
    """When uv.exe is nowhere, the fallback returns None."""
    fake_home = tmp_path / "fakehome"
    fake_local = tmp_path / "fakeappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local))

    with patch("platform.system", return_value="Windows"), \
         patch("hermes_cli.managed_uv.Path.home", return_value=fake_home):
        from hermes_cli.managed_uv import _find_uv_fallback
        assert _find_uv_fallback() is None


def test_install_ps1_has_fallback_search():
    """The install.ps1 script must include a fallback search path."""
    src = Path("scripts/install.ps1").read_text(encoding="utf-8")
    assert "fallback" in src.lower(), "install.ps1 must have fallback search"
    assert ".local\\bin" in src, "install.ps1 must search ~/.local/bin"
    assert "astral\\bin" in src, "install.ps1 must search $LOCALAPPDATA/astral/bin"