"""Offline payload packing: the archive must describe and carry real bytes.

These are unit tests over a small synthetic payload. They assert the artifact
contract an air-gapped transfer depends on — the manifest identifies the
target, the recorded digest matches the archive actually written, an incomplete
payload is refused before it becomes a plausible-looking archive, and a
tampered archive is detected rather than silently accepted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.offline import pack


def make_payload(root: Path, *, target: str = "linux-x64", launchers=("hermes", "hermes-agent")) -> Path:
    """A minimal payload shaped like `hermes pm bundle --out` output."""
    payload = root / "payload"
    for name in pack.REQUIRED_ENTRIES:
        if name == "manifest.json":
            continue
        (payload / name).mkdir(parents=True, exist_ok=True)
    (payload / "bin").mkdir(exist_ok=True)
    commands = {}
    for launcher in launchers:
        shim = payload / "bin" / launcher
        shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        shim.chmod(0o755)
        commands[launcher] = f"bin/{launcher}"
    (payload / "manifest.json").write_text(json.dumps({
        "schema": 1, "target": target, "repo": "hermes-agent", "ref": "HEAD",
        "launchers": list(launchers),
        "runtime": {"repoDir": "hermes-agent", "commands": commands},
    }), encoding="utf-8")
    (payload / "hermes-agent" / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return payload


def test_manifest_records_target_revision_and_matching_archive_digest(tmp_path):
    payload = make_payload(tmp_path, target="linux-arm64")
    out = tmp_path / "transfer"
    out.mkdir()
    archive = out / "hermes-offline-linux-arm64.tar.gz"
    pack._archive(payload, archive)

    manifest = pack.build_manifest(payload, archive)

    assert manifest["kind"] == "hermes-offline-payload"
    assert manifest["schema"] == pack.PACK_SCHEMA
    assert manifest["target"] == "linux-arm64"
    assert manifest["ref"] == "HEAD"
    assert manifest["payloadRoot"] == payload.name
    # The recorded digest must describe the archive actually written, or the
    # air-gapped machine rejects a good transfer (or accepts a bad one).
    assert manifest["archiveSha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["archiveBytes"] == archive.stat().st_size
    assert manifest["launchers"] == ["hermes", "hermes-agent"]


def test_archive_stores_a_single_relocatable_top_level_directory(tmp_path):
    payload = make_payload(tmp_path)
    archive = tmp_path / "payload.tar.gz"
    pack._archive(payload, archive)

    members = pack.tarfile.open(archive).getnames()

    assert all(name == payload.name or name.startswith(f"{payload.name}/") for name in members), members
    assert f"{payload.name}/manifest.json" in members


def test_digest_changes_when_payload_bytes_change(tmp_path):
    payload = make_payload(tmp_path)
    archive = tmp_path / "payload.tar.gz"
    pack._archive(payload, archive)
    before = pack.build_manifest(payload, archive)["archiveSha256"]

    (payload / "hermes-agent" / "pyproject.toml").write_text("[project]\nname='y'\n", encoding="utf-8")
    archive.unlink()
    pack._archive(payload, archive)

    assert pack.build_manifest(payload, archive)["archiveSha256"] != before


def test_tampered_archive_fails_the_recorded_digest(tmp_path):
    payload = make_payload(tmp_path)
    archive = tmp_path / "payload.tar.gz"
    pack._archive(payload, archive)
    expected = pack.build_manifest(payload, archive)["archiveSha256"]

    blob = bytearray(archive.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    archive.write_bytes(bytes(blob))

    assert hashlib.sha256(archive.read_bytes()).hexdigest() != expected


def test_incomplete_payload_is_rejected_before_packing(tmp_path):
    payload = make_payload(tmp_path)
    (payload / "uv-cache").rmdir()

    with pytest.raises(SystemExit, match="incomplete"):
        pack._validate_payload(payload)


def test_manifest_without_a_target_is_rejected(tmp_path):
    payload = make_payload(tmp_path)
    manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("target")
    (payload / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="no target"):
        pack._validate_payload(payload)


def test_missing_launcher_is_rejected(tmp_path):
    payload = make_payload(tmp_path)
    (payload / "bin" / "hermes-agent").unlink()

    with pytest.raises(SystemExit, match="launcher is missing"):
        pack._validate_payload(payload)


def test_a_directory_shaped_payload_is_rejected(tmp_path):
    payload = make_payload(tmp_path)
    (payload / "manifest.json").unlink()
    (payload / "manifest.json").mkdir()

    with pytest.raises(SystemExit, match="manifest is unreadable"):
        pack._validate_payload(payload)
