"""Native-Windows witness for the Desktop package rollback transaction.

The JS builder boundary and the Python launchability gate are intentionally
separate authorities. These tests run only on a real Windows runner and drive
the same on-disk ``win-unpacked`` / ``win-unpacked.bak`` generations through
both layers. No Electron or npm mock stands in for the filesystem transaction.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main

PE_AMD64 = 0x8664
PE_ARM64 = 0xAA64


def _make_pe(
    path: Path,
    machine: int,
    *,
    marker: bytes,
    truncate_to: int | None = None,
) -> Path:
    """Write a minimal PE whose section table is accepted by both verifiers."""
    buf = bytearray(0x400)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x80)
    buf[0x80:0x84] = b"PE\x00\x00"
    struct.pack_into("<HHIIIHH", buf, 0x84, machine, 1, 0, 0, 0, 0, 0x0002)
    section_off = 0x98
    struct.pack_into("<II", buf, section_off + 16, 0x200, 0x200)
    buf[0x200 : 0x200 + len(marker)] = marker
    data = bytes(buf if truncate_to is None else buf[:truncate_to])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _settle_apparent_builder_success(release_dir: Path, session_id: str) -> dict:
    node = shutil.which("node")
    assert node is not None, "Windows transaction witness requires Node on PATH"

    transaction_module = (
        Path(__file__).resolve().parents[2]
        / "apps"
        / "desktop"
        / "scripts"
        / "desktop-pack-transaction.mjs"
    )
    assert transaction_module.is_file()

    script = r"""
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [modulePath, releaseDir, sessionId] = process.argv.slice(1)
const {
  settleDesktopPack,
  writeRollbackSession
} = await import(pathToFileURL(modulePath).href)

const backupDir = path.join(releaseDir, 'win-unpacked.bak')
writeRollbackSession(backupDir, sessionId)
const result = settleDesktopPack({
  releaseDir,
  builderSucceeded: true,
  sessionId
})
process.stdout.write(JSON.stringify(result))
"""
    completed = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            script,
            str(transaction_module),
            str(release_dir),
            session_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"Node settlement failed with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


@pytest.mark.windows_only
def test_valid_candidate_crosses_js_settlement_and_python_gate(tmp_path):
    """A plausible replacement keeps rollback authority until Python accepts it."""
    desktop_dir = tmp_path / "apps" / "desktop"
    release_dir = desktop_dir / "release"
    candidate = _make_pe(
        release_dir / "win-unpacked" / "Hermes.exe",
        PE_AMD64,
        marker=b"accepted-candidate",
    )
    backup = _make_pe(
        release_dir / "win-unpacked.bak" / "Hermes.exe",
        PE_AMD64,
        marker=b"previous-generation",
    )
    candidate_bytes = candidate.read_bytes()
    backup_bytes = backup.read_bytes()

    settlement = _settle_apparent_builder_success(release_dir, "accepted-session")

    assert settlement["ok"] is True
    assert len(settlement["retained"]) == 1
    assert settlement["restored"] == []
    assert candidate.read_bytes() == candidate_bytes
    assert backup.read_bytes() == backup_bytes

    with patch(
        "hermes_cli.main._expected_windows_pe_machines",
        return_value={PE_AMD64},
    ):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(
            desktop_dir, candidate
        )

    assert verified == candidate
    assert rolled_back is False
    assert candidate.read_bytes() == candidate_bytes
    # The accepted generation cannot retroactively destroy rollback material;
    # the next package generation replaces it transactionally.
    assert backup.read_bytes() == backup_bytes


@pytest.mark.windows_only
def test_wrong_arch_candidate_is_rolled_back_by_python_after_js_retains_it(tmp_path):
    """The exact former blocker: a structurally valid wrong-arch PE reaches Python."""
    desktop_dir = tmp_path / "apps" / "desktop"
    release_dir = desktop_dir / "release"
    candidate = _make_pe(
        release_dir / "win-unpacked" / "Hermes.exe",
        PE_ARM64,
        marker=b"wrong-arch-candidate",
    )
    backup = _make_pe(
        release_dir / "win-unpacked.bak" / "Hermes.exe",
        PE_AMD64,
        marker=b"known-good-backup",
    )
    corrupt_bytes = candidate.read_bytes()
    backup_bytes = backup.read_bytes()

    settlement = _settle_apparent_builder_success(release_dir, "rollback-session")

    assert settlement["ok"] is True
    assert len(settlement["retained"]) == 1
    assert settlement["restored"] == []
    assert backup.exists()

    with (
        patch(
            "hermes_cli.main._expected_windows_pe_machines",
            return_value={PE_AMD64},
        ),
        patch("hermes_cli.main._purge_electron_build_cache", return_value=[]),
        patch(
            "hermes_cli.main._desktop_stamp_path",
            return_value=tmp_path / "desktop-build-stamp.json",
        ),
    ):
        verified, rolled_back = cli_main._ensure_desktop_exe_launchable(
            desktop_dir, candidate
        )

    assert verified == candidate
    assert rolled_back is True
    assert candidate.read_bytes() == backup_bytes
    assert not backup.exists()
    assert (
        release_dir / "win-unpacked.corrupt" / "Hermes.exe"
    ).read_bytes() == corrupt_bytes


@pytest.mark.windows_only
def test_structurally_truncated_candidate_restores_before_python_gate(tmp_path):
    """A false builder success cannot leave an incomplete PE as the active app."""
    release_dir = tmp_path / "apps" / "desktop" / "release"
    candidate = _make_pe(
        release_dir / "win-unpacked" / "Hermes.exe",
        PE_AMD64,
        marker=b"truncated-candidate",
        truncate_to=0x300,
    )
    backup = _make_pe(
        release_dir / "win-unpacked.bak" / "Hermes.exe",
        PE_AMD64,
        marker=b"known-good-backup",
    )
    backup_bytes = backup.read_bytes()

    settlement = _settle_apparent_builder_success(release_dir, "truncated-session")

    assert settlement["ok"] is False
    assert len(settlement["restored"]) == 1
    assert settlement["retained"] == []
    assert settlement["failures"]
    assert candidate.read_bytes() == backup_bytes
    assert not backup.exists()
