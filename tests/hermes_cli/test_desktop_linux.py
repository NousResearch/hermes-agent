"""Tests for Linux Electron sandbox fixup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import desktop_linux


def test_ensure_electron_sandbox_fixup_configures_missing_setuid(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_linux.sys, "platform", "linux")
    packaged = tmp_path / "Hermes"
    packaged.write_text("", encoding="utf-8")
    sandbox = packaged.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    sandbox.chmod(0o755)
    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.desktop_linux.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.desktop_linux.subprocess.run", return_value=ok) as mock_run:
        assert desktop_linux.ensure_electron_sandbox_fixup(packaged) is True

    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/sudo", "chown", "root:root", str(sandbox)]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/sudo", "chmod", "4755", str(sandbox)]


def test_ensure_electron_sandbox_fixup_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_linux.sys, "platform", "linux")
    packaged = tmp_path / "Hermes"
    packaged.write_text("", encoding="utf-8")
    target = tmp_path / "target"
    target.write_text("pwned", encoding="utf-8")
    sandbox = packaged.parent / "chrome-sandbox"
    sandbox.symlink_to(target)

    with patch("hermes_cli.desktop_linux.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.desktop_linux.subprocess.run") as mock_run:
        assert desktop_linux.ensure_electron_sandbox_fixup(packaged) is False

    mock_run.assert_not_called()


def test_ensure_electron_sandbox_fixup_skips_when_already_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_linux.sys, "platform", "linux")
    packaged = tmp_path / "Hermes"
    packaged.write_text("", encoding="utf-8")
    sandbox = packaged.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    import stat as stat_mod

    fake_stat = type("s", (), {"st_uid": 0, "st_mode": 0o4755 | stat_mod.S_IFREG})()
    monkeypatch.setattr(type(sandbox), "lstat", lambda self: fake_stat)

    with patch("hermes_cli.desktop_linux.subprocess.run") as mock_run:
        assert desktop_linux.ensure_electron_sandbox_fixup(packaged) is True

    mock_run.assert_not_called()
