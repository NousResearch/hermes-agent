"""Tests for the Photon CLI setup/status helpers."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from plugins.platforms.photon import cli as photon_cli


def test_sidecar_runtime_probe_polyfills_file_before_import(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", tmp_path)
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    ok, detail = photon_cli._run_sidecar_runtime_probe("/usr/bin/node")

    assert ok is True
    assert detail == "ok"
    assert calls[0] == ["/usr/bin/node", "--check", "index.mjs"]
    assert calls[1][:3] == ["/usr/bin/node", "--input-type=module", "-e"]
    probe_script = calls[1][3]
    assert "File as NodeFile" in probe_script
    assert "await import('spectrum-ts')" in probe_script
    assert "spectrum-ts/providers/imessage" in probe_script


def test_sidecar_runtime_probe_reports_node_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", tmp_path)

    def fake_run(args, **kwargs):
        assert kwargs["timeout"] == photon_cli._SIDECAR_RUNTIME_PROBE_TIMEOUT_SECONDS
        if "--check" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="syntax bad")
        raise AssertionError("runtime import should not run after --check failure")

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    ok, detail = photon_cli._run_sidecar_runtime_probe("node")

    assert ok is False
    assert "syntax bad" in detail


def test_sidecar_runtime_probe_reports_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", tmp_path)

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    ok, detail = photon_cli._run_sidecar_runtime_probe("node")

    assert ok is False
    assert "timed out after 20s" in detail


def test_sidecar_runtime_probe_reports_os_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", tmp_path)

    def fake_run(args, **kwargs):
        raise OSError("not executable")

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)

    ok, detail = photon_cli._run_sidecar_runtime_probe("/tmp/node")

    assert ok is False
    assert "not executable" in detail


def test_resolve_node_bin_rejects_non_executable_photon_node_bin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_file = tmp_path / "node"
    node_file.write_text("not executable")
    monkeypatch.setenv("PHOTON_NODE_BIN", str(node_file))
    monkeypatch.setattr(photon_cli.shutil, "which", lambda _name: None)

    assert photon_cli._resolve_node_bin() is None


def test_resolve_node_bin_rejects_directory_photon_node_bin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    monkeypatch.setenv("PHOTON_NODE_BIN", str(node_dir))
    monkeypatch.setattr(photon_cli.shutil, "which", lambda _name: None)

    assert photon_cli._resolve_node_bin() is None


def test_resolve_node_bin_accepts_executable_photon_node_bin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_file = tmp_path / "node"
    node_file.write_text("#!/bin/sh\n")
    node_file.chmod(0o700)
    monkeypatch.setenv("PHOTON_NODE_BIN", str(node_file))
    monkeypatch.setattr(photon_cli.shutil, "which", lambda _name: None)

    assert photon_cli._resolve_node_bin() == str(node_file)


def test_install_sidecar_fails_when_runtime_probe_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(photon_cli.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(args, **kwargs):
        assert args == ["/usr/bin/npm", "install"]
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(photon_cli, "_run_sidecar_runtime_probe", lambda _node: (False, "File is not defined"))

    rc = photon_cli._install_sidecar()

    captured = capsys.readouterr()
    assert rc == 1
    assert "sidecar runtime verification failed" in captured.err
    assert "File is not defined" in captured.err


def test_install_sidecar_verifies_after_npm_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(photon_cli.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(args, **kwargs):
        assert args == ["/usr/bin/npm", "install"]
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(photon_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(photon_cli, "_run_sidecar_runtime_probe", lambda _node: (True, "ok"))

    rc = photon_cli._install_sidecar()

    captured = capsys.readouterr()
    assert rc == 0
    assert "sidecar runtime verified" in captured.out


def test_gateway_setup_propagates_setup_failure(monkeypatch) -> None:
    def fake_setup(args: argparse.Namespace) -> int:
        assert args.photon_command == "setup"
        assert args.skip_sidecar_install is False
        return 7

    monkeypatch.setattr(photon_cli, "_cmd_setup", fake_setup)

    with pytest.raises(SystemExit) as exc_info:
        photon_cli.gateway_setup()

    assert exc_info.value.code == 7


def test_status_prints_sidecar_runtime(monkeypatch, tmp_path: Path, capsys) -> None:
    sidecar_dir = tmp_path / "sidecar"
    (sidecar_dir / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(photon_cli, "_SIDECAR_DIR", sidecar_dir)
    monkeypatch.setattr(photon_cli, "_resolve_node_bin", lambda: "/usr/bin/node")
    monkeypatch.setattr(photon_cli, "_sidecar_runtime_status", lambda: "✓ import OK")
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "print_credential_summary",
        lambda emit: emit("Photon iMessage status\n  project key         : ✓ stored"),
    )

    rc = photon_cli._cmd_status(argparse.Namespace())

    captured = capsys.readouterr()
    assert rc == 0
    assert "sidecar deps        : ✓ installed" in captured.out
    assert "sidecar runtime     : ✓ import OK" in captured.out
