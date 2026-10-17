"""The offline installer verifies before it extracts, and never clobbers launchers.

An air-gapped install cannot re-download a damaged transfer, so the only chance
to reject a corrupt archive is on arrival. These tests run the real installer
against a small synthetic payload and assert the properties an operator relies
on: a bad digest writes nothing, a good one publishes the commands, and an
unrelated `hermes` already on PATH is left alone.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path
import shutil

import pytest

pytestmark = pytest.mark.platforms("posix")

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "offline" / "install.sh"
sys.path.insert(0, str(ROOT))

from scripts.offline import pack  # noqa: E402


def make_transfer(tmp_path: Path, *, target: str = "linux-x64") -> tuple[Path, Path]:
    """Build a real archive + manifest pair with the real packer."""
    payload = tmp_path / "payload"
    for name in ("hermes-agent", "tools", "pm-runtime", "venv", "uv-cache"):
        (payload / name).mkdir(parents=True, exist_ok=True)
    (payload / "hermes-agent" / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (payload / "bin").mkdir()
    for launcher in ("hermes", "hermes-agent", "hermes-acp"):
        shim = payload / "bin" / launcher
        # A stand-in for the payload's prebuilt shim: the installer only needs
        # it to exist and run.
        shim.write_text('#!/bin/sh\necho "fake hermes $*"\n', encoding="utf-8")
        shim.chmod(0o755)
    (payload / "manifest.json").write_text(json.dumps({
        "schema": 1, "target": target, "repo": "hermes-agent", "ref": "HEAD",
        "launchers": ["hermes", "hermes-agent", "hermes-acp"],
        "runtime": {"commands": {name: f"bin/{name}" for name in ("hermes", "hermes-agent", "hermes-acp")}},
    }), encoding="utf-8")

    transfer = tmp_path / "transfer"
    transfer.mkdir()
    archive = transfer / "hermes-offline-linux-x64.tar.gz"
    pack._archive(payload, archive)
    manifest = pack.build_manifest(payload, archive)
    manifest_path = transfer / "hermes-offline-linux-x64.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return archive, manifest_path


def run_installer(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(INSTALL_SH), *args],
        capture_output=True, text=True, timeout=180,
    )


def test_corrupt_archive_is_rejected_before_anything_is_written(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    blob = bytearray(archive.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    archive.write_bytes(bytes(blob))

    dest = tmp_path / "install"
    result = run_installer("--archive", str(archive), "--manifest", str(manifest),
                           "--dest", str(dest), "--bin-dir", str(tmp_path / "bin"), "--no-shell")

    assert result.returncode != 0, result.stdout
    assert "digest mismatch" in result.stderr
    # The failure must precede extraction: a rejected transfer leaves no payload.
    assert not (dest / "payload").exists()


def test_verified_archive_installs_and_publishes_launchers(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    dest = tmp_path / "install"
    bin_dir = tmp_path / "bin"

    result = run_installer("--archive", str(archive), "--manifest", str(manifest),
                           "--dest", str(dest), "--bin-dir", str(bin_dir), "--no-shell")

    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "payload" / "manifest.json").is_file()
    for launcher in ("hermes", "hermes-agent", "hermes-acp"):
        link = bin_dir / launcher
        assert link.is_symlink(), launcher
        # The link must point into the payload, or the install follows an
        # unrelated command on PATH.
        assert (dest / "payload") in link.resolve().parents or "payload" in str(link.readlink())


def test_existing_unowned_launcher_is_not_clobbered(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A pipx-style command the installer does not own.
    foreign = bin_dir / "hermes"
    foreign.write_text("#!/bin/sh\necho pipx\n", encoding="utf-8")
    foreign.chmod(0o755)

    result = run_installer("--archive", str(archive), "--manifest", str(manifest),
                           "--dest", str(tmp_path / "install"), "--bin-dir", str(bin_dir), "--no-shell")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not foreign.is_symlink(), "an unrelated command must not be replaced"
    assert "pipx" in foreign.read_text(encoding="utf-8")
    assert "not a symlink" in result.stdout


def test_previous_payload_is_moved_aside_not_deleted(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    dest = tmp_path / "install"
    bin_dir = tmp_path / "bin"
    args = ("--archive", str(archive), "--manifest", str(manifest),
            "--dest", str(dest), "--bin-dir", str(bin_dir), "--no-shell")

    assert run_installer(*args).returncode == 0
    (dest / "payload" / "SENTINEL").write_text("first install\n", encoding="utf-8")

    assert run_installer(*args).returncode == 0
    retired = sorted(path for path in dest.glob("payload.previous-*"))
    assert retired, "the previous payload must be kept for rollback"
    assert (retired[-1] / "SENTINEL").read_text(encoding="utf-8") == "first install\n"


def test_dry_run_changes_nothing(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    dest = tmp_path / "install"

    result = run_installer("--archive", str(archive), "--manifest", str(manifest),
                           "--dest", str(dest), "--bin-dir", str(tmp_path / "bin"),
                           "--no-shell", "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not dest.exists()


def test_archive_without_a_manifest_is_refused(tmp_path):
    archive, manifest = make_transfer(tmp_path)
    manifest.unlink()

    result = run_installer("--archive", str(archive), "--dest", str(tmp_path / "install"),
                           "--bin-dir", str(tmp_path / "bin"), "--no-shell")

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()
