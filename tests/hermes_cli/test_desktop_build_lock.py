"""Behavior tests for the Desktop build's cross-process lock."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main_desktop as cli_desktop
from hermes_cli.desktop_build_lock import DesktopBuildLock


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    return root


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        build_only=False,
        cwd=None,
        fake_boot=False,
        force_build=False,
        hermes_root=None,
        ignore_existing=False,
        local=False,
        skip_build=False,
        source=False,
        setup_tcc_identity=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_desktop_build_lock_is_exclusive_and_reacquirable(tmp_path):
    first = DesktopBuildLock(tmp_path)
    contender = DesktopBuildLock(tmp_path)

    assert first.acquire() is True
    try:
        assert contender.acquire() is False
    finally:
        first.release()

    assert contender.acquire() is True
    contender.release()


def test_desktop_build_lock_excludes_another_process(tmp_path):
    holder = DesktopBuildLock(tmp_path)
    assert holder.acquire() is True

    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from hermes_cli.desktop_build_lock import DesktopBuildLock\n"
        "lock = DesktopBuildLock(Path(sys.argv[1]))\n"
        "raise SystemExit(0 if lock.acquire() else 23)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            check=False,
        )
    finally:
        holder.release()

    assert result.returncode == 23


def test_desktop_build_lock_releases_after_exception(tmp_path):
    try:
        with DesktopBuildLock(tmp_path):
            raise ValueError("build failed")
    except ValueError:
        pass

    successor = DesktopBuildLock(tmp_path)
    assert successor.acquire() is True
    successor.release()


def test_desktop_build_lock_is_keyed_by_checkout(tmp_path):
    """Two different checkouts must not share one lock: node_modules and release/ are checkout-scoped."""
    other = tmp_path / "hermes-agent-2"
    other.mkdir()
    first = DesktopBuildLock(tmp_path)
    second = DesktopBuildLock(other)

    assert first.acquire() is True
    try:
        assert second.acquire() is True
    finally:
        first.release()
    second.release()
    assert first.path != second.path


def test_gui_reports_missing_source_before_constructing_build_lock(tmp_path, monkeypatch, capsys):
    root = tmp_path / "broken-hermes-agent"
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root, raising=False)

    args = argparse.Namespace()
    with patch(
        "hermes_cli.desktop_build_lock.DesktopBuildLock",
        side_effect=AssertionError("build lock constructed before source validation"),
    ), pytest.raises(SystemExit) as exc:
        cli_desktop.cmd_gui(args)

    assert exc.value.code == 1
    assert capsys.readouterr().out.strip() == (
        f"Desktop GUI source not found at: {root / 'apps' / 'desktop'}"
    )


def test_gui_refuses_contended_build_before_checking_freshness(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root, raising=False)

    holder = DesktopBuildLock(root)
    assert holder.acquire() is True
    try:
        with patch(
            "hermes_cli.main_desktop._desktop_build_needed",
            side_effect=AssertionError("freshness check ran without the build lock"),
        ), pytest.raises(SystemExit) as exc:
            cli_desktop.cmd_gui(_args())
    finally:
        holder.release()

    assert exc.value.code == 2


def test_gui_releases_lock_before_packaged_electron_handoff(tmp_path, monkeypatch):
    root = _checkout(tmp_path)
    desktop_dir = root / "apps" / "desktop"

    if sys.platform == "darwin":
        executable = desktop_dir / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif sys.platform == "win32":
        executable = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    else:
        executable = desktop_dir / "release" / "linux-unpacked" / "hermes"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root, raising=False)
    # The lock is checkout-keyed via the profile-common Hermes root; pin it so
    # the holder and the handoff probe agree regardless of the runner's home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    # The fake executable has no Electron sandbox helper; the platform launch
    # fixups are environment-dependent (sudo tty, userns policy), not the test's
    # subject — a Linux CI runner exits 1 in _packaged_desktop_launch_command.
    monkeypatch.setattr(cli_desktop, "_desktop_linux_sandbox_fixup", lambda _exe: True, raising=False)
    monkeypatch.setattr(
        cli_desktop, "_installed_desktop_launch_target", lambda _dir, exe: exe, raising=False)

    def launch_after_lock_release(*_args, **_kwargs):
        handoff_probe = DesktopBuildLock(root)
        assert handoff_probe.acquire() is True, "build lock was not released before the Electron handoff"
        handoff_probe.release()
        return subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main_desktop._desktop_launch_env", return_value=({}, [])), \
         patch("hermes_cli.main_desktop._register_linux_desktop_entry", return_value=None), \
         patch("hermes_cli.main_desktop.subprocess.run", side_effect=launch_after_lock_release), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=False), \
         pytest.raises(SystemExit) as exc:
        cli_desktop.cmd_gui(_args())

    assert exc.value.code == 0


def test_gui_releases_lock_after_build_failure(tmp_path, monkeypatch, capsys):
    """A failed build must not deadlock the next `hermes desktop`."""
    root = _checkout(tmp_path)
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root, raising=False)

    def failing_build(*_args, **_kwargs):
        raise RuntimeError("tsc missing: ENOTEMPTY node_modules race")

    with patch("hermes_cli.main_desktop._desktop_launch_env", return_value=({}, [])), \
         patch("hermes_cli.main_desktop._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main_desktop.build_prepared_desktop", side_effect=failing_build), \
         pytest.raises(SystemExit) as exc:
        cli_desktop.cmd_gui(_args(build_only=True))

    assert exc.value.code == 1
    successor = DesktopBuildLock(root)
    assert successor.acquire() is True, "build lock was not released after a failed build"
    successor.release()


def test_update_path_waits_for_a_held_lock(tmp_path, monkeypatch, capsys):
    """`hermes update`'s desktop rebuild queues behind a holder instead of failing the update."""
    from hermes_cli import source_build as source_build_mod
    from hermes_cli.main_desktop import _refresh_installed_desktop_apps

    root = _checkout(tmp_path)

    holder = DesktopBuildLock(root)
    assert holder.acquire() is True

    release_calls: list[bool] = []

    class _ReleaseOnEnter(DesktopBuildLock):
        def acquire(self, *, wait: bool = False) -> bool:
            release_calls.append(wait)
            # Simulate the holder finishing while we wait: release, then succeed.
            holder.release()
            return super().acquire(wait=wait)

    built: list[bool] = []

    def fake_build_prepared_desktop(desktop_dir, *, source_mode, npm, env, icons=None):
        built.append(True)
        return None

    monkeypatch.setattr(source_build_mod, "_install_configured_features_missing_deps", lambda *_: None, raising=False)
    monkeypatch.setattr(source_build_mod, "source_frontends", lambda _root: ("ui-tui", "web"), raising=False)
    monkeypatch.setattr(source_build_mod, "source_build_env", lambda **_kw: {"PATH": "/usr/bin"}, raising=False)
    monkeypatch.setattr(source_build_mod, "prepare_source_dependencies", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(source_build_mod, "build_source_tui", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(source_build_mod, "build_source_web", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(source_build_mod, "_refresh_installed_desktop_apps", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr("hermes_cli.desktop_build_lock.DesktopBuildLock", _ReleaseOnEnter)
    monkeypatch.setattr("hermes_cli.main_desktop.build_prepared_desktop", fake_build_prepared_desktop)
    monkeypatch.setattr("hermes_cli.update_stage.publish_stage", lambda _s: None, raising=False)

    source_build_mod.build_update_products(root, desktop=True)

    assert release_calls == [True], "update path must acquire the lock in wait mode"
    assert built == [True]
