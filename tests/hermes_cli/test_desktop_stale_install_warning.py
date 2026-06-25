"""Tests for stale installed macOS Desktop bundle detection (#52339)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import main as cli_main


def _make_built(tmp_path: Path, mtime: float | None = None) -> Path:
    exe = tmp_path / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    if mtime is not None:
        os.utime(exe, (mtime, mtime))
    return exe


def _make_installed(base: Path, mtime: float) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    exe = base / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    os.utime(exe, (mtime, mtime))
    return base / "Hermes.app"


def test_detects_stale_installed_app(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    built = _make_built(tmp_path, mtime=2000)
    apps = tmp_path / "Applications"
    installed_app = _make_installed(apps, mtime=1000)  # older than build
    result = cli_main._stale_installed_macos_desktop_app(built, search_bases=[apps])
    assert result == installed_app


def test_fresh_installed_app_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    built = _make_built(tmp_path, mtime=1000)
    apps = tmp_path / "Applications"
    _make_installed(apps, mtime=2000)  # newer than build
    assert cli_main._stale_installed_macos_desktop_app(built, search_bases=[apps]) is None


def test_no_installed_app(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    built = _make_built(tmp_path, mtime=1000)
    apps = tmp_path / "Applications"
    assert cli_main._stale_installed_macos_desktop_app(built, search_bases=[apps]) is None


def test_non_darwin_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    built = _make_built(tmp_path, mtime=2000)
    apps = tmp_path / "Applications"
    _make_installed(apps, mtime=1000)
    assert cli_main._stale_installed_macos_desktop_app(built, search_bases=[apps]) is None


def test_none_executable_returns_none(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    assert cli_main._stale_installed_macos_desktop_app(None) is None


def test_same_bundle_not_flagged(tmp_path, monkeypatch):
    """If the installed app IS the built bundle (in-place), don't warn."""
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    apps = tmp_path / "Applications"
    apps.mkdir()
    app = apps / "Hermes.app"
    exe = app / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    os.utime(exe, (1000, 1000))
    # built executable is the same path, but pretend the build refreshed mtime
    built = exe
    os.utime(built, (2000, 2000))
    assert cli_main._stale_installed_macos_desktop_app(built, search_bases=[apps]) is None
