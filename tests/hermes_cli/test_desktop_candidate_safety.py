from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main_desktop, update_cmd, update_cmd_deps


TXID = "12345678-1234-4abc-8def-1234567890ab"


def test_update_facade_exports_candidate_receipt_writer():
    """The post-update path resolves desktop helpers through ``update_cmd._m()``."""
    assert update_cmd._m()._write_desktop_producer_receipt is main_desktop._write_desktop_producer_receipt


def _make_mac_release(root: Path, label: str) -> tuple[Path, Path]:
    app = root / "mac-arm64" / "Hermes.app"
    exe = app / "Contents" / "MacOS" / "Hermes"
    resources = app / "Contents" / "Resources"
    exe.parent.mkdir(parents=True)
    resources.mkdir(parents=True)
    exe.write_bytes(f"{label}-executable".encode())
    (resources / "app.asar").write_bytes(f"{label}-asar".encode())
    with (app / "Contents" / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleIdentifier": "com.nousresearch.hermes",
                "CFBundleShortVersionString": "1.2.3",
                "CFBundleVersion": "456",
            },
            stream,
        )
    return app, exe


def _assert_promotion_refused_unchanged(
    desktop_dir: Path,
    staging_dir: Path,
    live_exe: Path,
    live_before: bytes,
) -> None:
    with pytest.raises(SystemExit) as exc:
        main_desktop._promote_staged_desktop_app(desktop_dir, staging_dir)
    assert exc.value.code == 1
    assert live_exe.read_bytes() == live_before


@pytest.mark.macos_only
@pytest.mark.parametrize("returncode,stdout,stderr", [(1, "", "permission denied"), (0, "", "")])
def test_macos_reader_scan_treats_ambiguous_lsof_results_as_unknown(
    tmp_path, monkeypatch, returncode, stdout, stderr
):
    app, _ = _make_mac_release(tmp_path / "release", "live")
    monkeypatch.setattr(main_desktop.shutil, "which", lambda _name: "/usr/sbin/lsof")
    monkeypatch.setattr(
        main_desktop.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, stdout, stderr),
    )

    assert main_desktop._desktop_macos_reader_state(app) == "unknown"


@pytest.mark.macos_only
def test_macos_promotion_refuses_false_signing_fixup_without_touching_release(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    _, live_exe = _make_mac_release(desktop / "release", "live")
    _, _ = _make_mac_release(desktop / ".staging", "staged")
    live_before = live_exe.read_bytes()
    monkeypatch.setattr(main_desktop, "_desktop_macos_reader_state", lambda _app: "none")
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *a, **kw: False)

    _assert_promotion_refused_unchanged(desktop, desktop / ".staging", live_exe, live_before)


@pytest.mark.macos_only
def test_macos_promotion_refuses_failed_independent_strict_signature_check(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    _, live_exe = _make_mac_release(desktop / "release", "live")
    _, _ = _make_mac_release(desktop / ".staging", "staged")
    live_before = live_exe.read_bytes()
    monkeypatch.setattr(main_desktop, "_desktop_macos_reader_state", lambda _app: "none")
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *a, **kw: True)
    monkeypatch.setattr(main_desktop, "_desktop_macos_strict_signature_valid", lambda _app: False)

    _assert_promotion_refused_unchanged(desktop, desktop / ".staging", live_exe, live_before)


@pytest.mark.macos_only
@pytest.mark.parametrize("reader_state", ["present", "unknown"])
def test_macos_promotion_refuses_live_or_unknown_release_readers(tmp_path, monkeypatch, reader_state):
    desktop = tmp_path / "desktop"
    _, live_exe = _make_mac_release(desktop / "release", "live")
    _, _ = _make_mac_release(desktop / ".staging", "staged")
    live_before = live_exe.read_bytes()
    monkeypatch.setattr(main_desktop, "_desktop_macos_reader_state", lambda _app: reader_state)

    with patch.object(main_desktop, "_desktop_macos_relaunchable_fixup") as fixup:
        _assert_promotion_refused_unchanged(desktop, desktop / ".staging", live_exe, live_before)
    fixup.assert_not_called()


@pytest.mark.macos_only
def test_macos_promotion_refuses_reader_that_appears_immediately_before_swap(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    _, live_exe = _make_mac_release(desktop / "release", "live")
    _, _ = _make_mac_release(desktop / ".staging", "staged")
    live_before = live_exe.read_bytes()
    states = iter(["none", "present"])
    monkeypatch.setattr(main_desktop, "_desktop_macos_reader_state", lambda _app: next(states))
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *a, **kw: True)
    monkeypatch.setattr(main_desktop, "_desktop_macos_strict_signature_valid", lambda _app: True)

    _assert_promotion_refused_unchanged(desktop, desktop / ".staging", live_exe, live_before)


@pytest.mark.macos_only
def test_macos_producer_receipt_binds_candidate_and_artifact_facts(tmp_path, monkeypatch):
    project = tmp_path / "repo"
    desktop = project / "apps" / "desktop"
    app, exe = _make_mac_release(desktop / "release", "candidate")
    asar = app / "Contents" / "Resources" / "app.asar"
    os.utime(exe, (1_700_000_000, 1_700_000_000))
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", project)
    monkeypatch.setenv("HERMES_DESKTOP_UPDATE_TRANSACTION_ID", TXID)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["/usr/bin/lipo", "-archs"]:
            return subprocess.CompletedProcess(command, 0, "arm64\n", "")
        if command[:2] == ["/usr/bin/codesign", "-dv"]:
            return subprocess.CompletedProcess(command, 0, "", "Signature=adhoc\nIdentifier=com.nousresearch.hermes\n")
        raise AssertionError(command)

    monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)
    receipt = main_desktop._write_desktop_producer_receipt(exe, exe, rebuilt=True)

    assert receipt == desktop / "release" / "mac-arm64" / ".hermes-desktop-producer-receipt.json"
    payload = json.loads(receipt.read_text())
    assert payload == {
        "schema_version": 1,
        "transaction_id": TXID,
        "candidate_path": str(app.resolve()),
        "post_update_source_sha": "a" * 40,
        "rebuilt": True,
        "build_timestamp": "2023-11-14T22:13:20Z",
        "source_state": "clean",
        "architecture": ["arm64"],
        "bundle_id": "com.nousresearch.hermes",
        "version": "1.2.3",
        "build_version": "456",
        "signing_policy": "adhoc",
        "signing_identity": "-",
        "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "app_asar_sha256": hashlib.sha256(asar.read_bytes()).hexdigest(),
    }


@pytest.mark.macos_only
def test_reused_macos_candidate_gets_a_transaction_bound_receipt(tmp_path, monkeypatch):
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    _, executable = _make_mac_release(desktop / "release", "reused")
    facade = MagicMock()
    facade.PROJECT_ROOT = tmp_path
    facade._desktop_packaged_executable.return_value = executable
    facade._desktop_dist_exists.return_value = False
    facade._resolve_node_runtime_npm.return_value = "/usr/bin/npm"
    facade._desktop_build_needed.return_value = False
    monkeypatch.setattr(update_cmd, "_m", lambda: facade)

    assert update_cmd_deps._rebuild_desktop_after_update(
        desktop, had_desktop_app_before_update=True
    ) is True
    facade._write_desktop_producer_receipt.assert_called_once_with(
        executable, executable, rebuilt=False
    )


@pytest.mark.macos_only
def test_posix_orchestrator_creates_and_exports_a_v4_transaction_id(tmp_path):
    script = Path(__file__).parents[2] / "scripts" / "desktop-update" / "posix.sh"
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    result = subprocess.run(
        ["/bin/bash", str(script), "--install-root", str(install_root), "--self-test-transaction"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["transaction_id"] == payload["exported_transaction_id"]
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        payload["transaction_id"],
    )
