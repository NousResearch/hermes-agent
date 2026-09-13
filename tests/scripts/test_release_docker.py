"""Tests for scripts/releases/docker.py — the staged Docker release contract."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

docker = importlib.import_module("scripts.releases.docker")

TAG = "v1.2.3"
COMMIT = "a" * 40
DIGESTS = {"amd64": "b" * 64, "arm64": "c" * 64}


@pytest.fixture
def module_root() -> Path:
    return Path(docker.__file__).resolve().parent.parent.parent


def test_build_manifest_roundtrip_and_identity_rejection() -> None:
    manifest = docker.build_manifest(TAG, COMMIT, DIGESTS)
    parsed = docker.parse_manifest(json.dumps(manifest).encode())
    docker.verify_manifest(parsed, TAG, COMMIT)
    with pytest.raises(docker.DockerReleaseError):
        docker.verify_manifest(parsed, "v1.2.4", COMMIT)
    with pytest.raises(docker.DockerReleaseError):
        docker.verify_manifest(parsed, TAG, "b" * 40)


def test_manifest_requires_both_arches() -> None:
    with pytest.raises(docker.DockerReleaseError):
        docker.build_manifest(TAG, COMMIT, {"amd64": DIGESTS["amd64"]})
    bad = dict(DIGESTS)
    bad["arm64"] = "z" * 64
    with pytest.raises(docker.DockerReleaseError):
        docker.build_manifest(TAG, COMMIT, bad)


def test_manifest_archive_hashes_optional_but_covering() -> None:
    manifest = docker.build_manifest(TAG, COMMIT, DIGESTS, {"amd64": "d" * 64, "arm64": "e" * 64})
    assert manifest["archives"] == {"amd64": "d" * 64, "arm64": "e" * 64}
    with pytest.raises(docker.DockerReleaseError):
        docker.build_manifest(TAG, COMMIT, DIGESTS, {"amd64": "d" * 64})
    with pytest.raises(docker.DockerReleaseError):
        docker.build_manifest(TAG, COMMIT, DIGESTS, {"amd64": "d" * 64, "riscv64": "e" * 64})


def test_parse_manifest_rejects_garbage() -> None:
    with pytest.raises(docker.DockerReleaseError):
        docker.parse_manifest(b"not json")
    with pytest.raises(docker.DockerReleaseError):
        docker.parse_manifest(json.dumps({"schema": 2}).encode())


def test_sha256_file(tmp_path: Path) -> None:
    blob = tmp_path / "image.tar"
    blob.write_bytes(b"payload")
    assert docker.sha256_file(str(blob)) == __import__("hashlib").sha256(b"payload").hexdigest()


def test_cli_manifest_and_verify(tmp_path: Path, module_root: Path) -> None:
    out = tmp_path / "manifest.json"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.releases.docker", "manifest",
         "--tag", TAG, "--commit", COMMIT,
         "--digest-amd64", DIGESTS["amd64"], "--digest-arm64", DIGESTS["arm64"]],
        cwd=module_root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out.write_text(proc.stdout)
    parsed = json.loads(proc.stdout)
    assert parsed["digests"] == DIGESTS

    verify = subprocess.run(
        [sys.executable, "-m", "scripts.releases.docker", "verify",
         "--tag", TAG, "--commit", COMMIT, str(out)],
        cwd=module_root, capture_output=True, text=True,
    )
    assert verify.returncode == 0, verify.stderr

    bad = subprocess.run(
        [sys.executable, "-m", "scripts.releases.docker", "verify",
         "--tag", "v9.9.9", "--commit", COMMIT, str(out)],
        cwd=module_root, capture_output=True, text=True,
    )
    assert bad.returncode == 1
    assert "::error::" in bad.stderr
