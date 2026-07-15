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


def test_cmd_gui_build_only_prints_stale_warning(tmp_path, monkeypatch, capsys):
    """Command-path regression: the --build-only branch of cmd_gui must emit
    the stale-install warning plus replacement-safe guidance (#52339 review)."""
    import argparse

    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)

    built = _make_built(tmp_path, mtime=2000)
    installed_app = _make_installed(tmp_path / "Applications", mtime=1000)

    # Avoid any real build/logging/env work; jump straight to the build-only branch.
    monkeypatch.setattr(cli_main, "_desktop_packaged_executable", lambda d: built)
    monkeypatch.setattr(
        cli_main,
        "_stale_installed_macos_desktop_app",
        lambda exe: installed_app,
    )

    args = argparse.Namespace(
        source=False,
        skip_build=True,
        force_build=False,
        build_only=True,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )

    cli_main.cmd_gui(args)

    out = capsys.readouterr().out
    assert "was NOT replaced" in out or "NOT replaced" in out
    assert "hermes update" in out
    # Replacement-safe swap, not an in-place cp into the stale bundle's parent.
    assert ".new" in out
    assert "mv " in out
