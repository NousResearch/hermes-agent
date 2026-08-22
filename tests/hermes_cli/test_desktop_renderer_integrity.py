"""Regression coverage for packaged renderer ASAR/disk skew (#81028)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


def _packaged_linux_tree(tmp_path: Path) -> tuple[Path, Path]:
    desktop_dir = tmp_path / "apps" / "desktop"
    executable = desktop_dir / "release" / "linux-unpacked" / "hermes"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    return desktop_dir, executable


def _matching_stamp(tmp_path: Path) -> Path:
    stamp = tmp_path / "desktop-build-stamp.json"
    stamp.write_text(
        json.dumps({"contentHash": "same", "sourceMode": False}),
        encoding="utf-8",
    )
    return stamp


def test_matching_content_stamp_cannot_hide_renderer_skew(tmp_path, monkeypatch):
    desktop_dir, executable = _packaged_linux_tree(tmp_path)
    stamp = _matching_stamp(tmp_path)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    with (
        patch("hermes_cli.main._desktop_stamp_path", return_value=stamp),
        patch("hermes_cli.main._compute_desktop_content_hash", return_value="same"),
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            return_value="unpacked renderer file is absent from ASAR index",
        ) as verify,
    ):
        assert cli_main._desktop_build_needed(desktop_dir, tmp_path, source_mode=False)

    verify.assert_called_once_with(tmp_path, executable)


def test_matching_content_stamp_skips_coherent_package(tmp_path, monkeypatch):
    desktop_dir, executable = _packaged_linux_tree(tmp_path)
    stamp = _matching_stamp(tmp_path)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    with (
        patch("hermes_cli.main._desktop_stamp_path", return_value=stamp),
        patch("hermes_cli.main._compute_desktop_content_hash", return_value="same"),
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            return_value=None,
        ) as verify,
    ):
        assert not cli_main._desktop_build_needed(
            desktop_dir, tmp_path, source_mode=False
        )

    verify.assert_called_once_with(tmp_path, executable)


def test_rollback_refuses_backup_with_stale_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    unpacked = tmp_path / "release" / "win-unpacked"
    current_exe = unpacked / "Hermes.exe"
    backup_exe = tmp_path / "release" / "win-unpacked.bak" / "Hermes.exe"
    current_exe.parent.mkdir(parents=True)
    backup_exe.parent.mkdir(parents=True)
    current_exe.write_bytes(b"new")
    backup_exe.write_bytes(b"old")

    with (
        patch("hermes_cli.main._desktop_exe_integrity_error", return_value=None),
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            return_value="stale ASAR index",
        ),
    ):
        assert cli_main._rollback_desktop_from_backup(current_exe) is None

    assert current_exe.read_bytes() == b"new"
    assert backup_exe.read_bytes() == b"old"


def test_windows_gate_restores_only_verified_renderer_backup(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    unpacked = tmp_path / "release" / "win-unpacked"
    current_exe = unpacked / "Hermes.exe"
    backup_exe = tmp_path / "release" / "win-unpacked.bak" / "Hermes.exe"
    current_exe.parent.mkdir(parents=True)
    backup_exe.parent.mkdir(parents=True)
    current_exe.write_bytes(b"new")
    backup_exe.write_bytes(b"old")

    def renderer_error(_root, executable):
        return "stale ASAR index" if executable == current_exe else None

    with (
        patch("hermes_cli.main.PROJECT_ROOT", tmp_path),
        patch("hermes_cli.main._desktop_exe_integrity_error", return_value=None),
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            side_effect=renderer_error,
        ),
        patch("hermes_cli.main._desktop_stamp_path", return_value=tmp_path / "stamp"),
    ):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(
            tmp_path, current_exe
        )

    assert verified == current_exe
    assert rolled_back is True
    assert current_exe.read_bytes() == b"old"
    assert "stale ASAR index" in capsys.readouterr().out


def test_failed_windows_pack_restores_verified_renderer_backup(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    (desktop_dir / "package.json").parent.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    current_exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    backup_exe = desktop_dir / "release" / "win-unpacked.bak" / "Hermes.exe"
    current_exe.parent.mkdir(parents=True)
    backup_exe.parent.mkdir(parents=True)
    current_exe.write_bytes(b"new-skewed-package")
    backup_exe.write_bytes(b"previous-good-package")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")

    args = argparse.Namespace(
        skip_build=False,
        build_only=True,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_failed = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with (
        patch("hermes_cli.main._resolve_node_runtime_npm", return_value="npm"),
        patch(
            "hermes_cli.main._run_npm_install_deterministic",
            return_value=install_ok,
        ),
        patch("hermes_cli.main._desktop_build_needed", return_value=True),
        patch("hermes_cli.main._stop_desktop_processes_locking_build", return_value=[]),
        patch("hermes_cli.main._desktop_exe_integrity_error", return_value=None),
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            side_effect=lambda _root, exe: (
                "stale ASAR index" if exe == current_exe else None
            ),
        ),
        patch("hermes_cli.main.subprocess.run", return_value=pack_failed),
        pytest.raises(SystemExit) as exc,
    ):
        cli_main.cmd_gui(args)

    assert exc.value.code == 1
    assert current_exe.read_bytes() == b"previous-good-package"
    output = capsys.readouterr().out
    assert "restored the previous verified package" in output
    assert "Desktop GUI build failed" in output


def test_skip_build_refuses_inconsistent_renderer_package(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    (desktop_dir / "package.json").parent.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    executable = desktop_dir / "release" / "linux-unpacked" / "hermes"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    args = argparse.Namespace(
        skip_build=True,
        build_only=False,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    with (
        patch(
            "hermes_cli.main._desktop_packaged_renderer_integrity_error",
            return_value="ASAR index points to a missing renderer asset",
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli_main.cmd_gui(args)

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "--skip-build cannot use" in output
    assert "Drop --skip-build to rebuild" in output
