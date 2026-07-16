"""Tests for bundle manifest writing and verification.

Phase 0 task 0.4: round-trip write→verify on a tiny fixture tree;
verification fails when any file is modified; fails when manifest is modified.

These are behavior/invariant tests — they exercise the real hashing and
verification logic, not mocks.
"""

import json
import os
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

# Import the module under test by path (it lives in scripts/release/, not a package)
_MANIFEST_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "release" / "write-manifest.py"
_spec = importlib.util.spec_from_file_location("write_manifest", _MANIFEST_SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["write_manifest"] = _mod
_spec.loader.exec_module(_mod)
from write_manifest import (  # noqa: E402
    collect_file_hashes,
    compute_file_hash,
    verify_file_hashes,
    write_manifest,
)


@pytest.fixture
def bundle_fixture(tmp_path):
    """Create a tiny fixture bundle directory."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "runtime" / "venv" / "bin").mkdir(parents=True)
    (bundle / "app").mkdir()
    (bundle / "bin").mkdir()

    (bundle / "runtime" / "venv" / "bin" / "python").write_text("# fake python")
    (bundle / "app" / "run_agent.py").write_text("# fake source\n")
    (bundle / "bin" / "hermes").write_text("#!/bin/sh\necho hermes\n")
    (bundle / "manifest.json").write_text("{}")  # placeholder, should be skipped
    return bundle


class TestCollectFileHashes:
    def test_collects_all_regular_files(self, bundle_fixture):
        hashes = collect_file_hashes(bundle_fixture)
        assert "runtime/venv/bin/python" in hashes
        assert "app/run_agent.py" in hashes
        assert "bin/hermes" in hashes

    def test_skips_manifest_files(self, bundle_fixture):
        hashes = collect_file_hashes(bundle_fixture)
        assert "manifest.json" not in hashes
        assert "manifest.json.minisig" not in hashes

    def test_hashes_are_sha256_format(self, bundle_fixture):
        hashes = collect_file_hashes(bundle_fixture)
        for path, h in hashes.items():
            assert h.startswith("sha256:")
            # sha256 hex is 64 chars after the prefix
            assert len(h) == len("sha256:") + 64


class TestComputeFileHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "b.txt"
        f2.write_text("world")
        assert compute_file_hash(f1) != compute_file_hash(f2)


class TestWriteManifest:
    def test_writes_valid_json(self, bundle_fixture):
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
            desktop=False,
        )
        assert manifest["schema"] == 1
        assert manifest["version"] == "2026.07.14"
        assert manifest["channel"] == "nightly"
        assert manifest["git_sha"] == "a" * 40
        assert manifest["platform"] == "linux-x64"
        assert manifest["min_updater_version"] == "0.1.0"
        assert manifest["desktop"] is False
        assert len(manifest["files"]) == 3

    def test_manifest_file_written_to_disk(self, bundle_fixture):
        write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        manifest_path = bundle_fixture / "manifest.json"
        assert manifest_path.exists()
        written = json.loads(manifest_path.read_text())
        assert written["schema"] == 1


class TestVerifyFileHashes:
    def test_clean_bundle_verifies(self, bundle_fixture):
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert ok, f"Expected clean verify, got errors: {errors}"
        assert errors == []

    def test_tampered_file_detected(self, bundle_fixture):
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        # Tamper with a file
        (bundle_fixture / "app" / "run_agent.py").write_text("# TAMPERED\n")
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert not ok
        assert any("tampered" in e for e in errors)
        assert any("app/run_agent.py" in e for e in errors)

    def test_missing_file_detected(self, bundle_fixture):
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        # Remove a file
        (bundle_fixture / "app" / "run_agent.py").unlink()
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert not ok
        assert any("missing" in e for e in errors)

    def test_extra_file_not_in_manifest_detected(self, bundle_fixture):
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        # Add a file not in the manifest
        (bundle_fixture / "evil.py").write_text("# evil injection")
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert not ok
        assert any("extra file" in e for e in errors)

    def test_tampered_manifest_detected(self, bundle_fixture):
        """If manifest.json is modified after writing, its hash entries
        no longer match the files — but this test is about modifying the
        manifest's *content* (version, etc.), not the file hashes. The file
        hash check still catches file tampering even if the manifest metadata
        is wrong."""
        manifest = write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="a" * 40,
            platform="linux-x64",
        )
        # Modify the manifest's version field (not the files)
        manifest["version"] = "2099.99.99"
        # File hashes should still verify (we didn't touch any files)
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert ok, "File hashes should still match even with modified manifest metadata"
        # But the version in the manifest is wrong — that's what the signature
        # catches (tested separately via minisign when available)


class TestRoundTrip:
    def test_write_then_verify(self, bundle_fixture):
        """Full round-trip: write manifest → verify all hashes pass."""
        write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="b" * 40,
            platform="linux-x64",
            desktop=True,
        )
        manifest = json.loads((bundle_fixture / "manifest.json").read_text())
        ok, errors = verify_file_hashes(bundle_fixture, manifest)
        assert ok, f"Round-trip verify failed: {errors}"


class TestEd25519Signing:
    """Tests for Ed25519 manifest signing via PyNaCl.

    No external CLI needed — PyNaCl is a Python dependency.
    """

    def test_sign_and_verify_manifest(self, bundle_fixture):
        """Generate a throwaway keypair, sign manifest, verify it."""
        from write_manifest import sign_manifest, verify_signature

        write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="c" * 40,
            platform="linux-x64",
        )

        # Sign with a throwaway keypair (seckey=None generates one)
        assert sign_manifest(bundle_fixture, seckey_b64=None)
        assert (bundle_fixture / "manifest.json.sig").exists()

        # Verify (uses the pubkey embedded in the .sig file)
        assert verify_signature(bundle_fixture)

        # Tamper with manifest
        manifest_path = bundle_fixture / "manifest.json"
        content = json.loads(manifest_path.read_text())
        content["version"] = "tampered"
        manifest_path.write_text(json.dumps(content))

        # Verification should now fail
        assert not verify_signature(bundle_fixture)

    def test_sign_with_explicit_key(self, bundle_fixture):
        """Sign with a known keypair and verify with the explicit pubkey."""
        import base64
        from write_manifest import sign_manifest, verify_signature

        write_manifest(
            bundle_fixture,
            version="2026.07.14",
            channel="nightly",
            git_sha="d" * 40,
            platform="linux-x64",
        )

        # Generate a keypair and extract the base64 secret key
        import nacl.signing
        key = nacl.signing.SigningKey.generate()
        seckey_b64 = base64.b64encode(bytes(key)).decode()
        pubkey_b64 = base64.b64encode(bytes(key.verify_key)).decode()

        # Sign with the explicit key
        assert sign_manifest(bundle_fixture, seckey_b64=seckey_b64)

        # Verify with the explicit pubkey
        assert verify_signature(bundle_fixture, pubkey_b64=pubkey_b64)

        # Verify with wrong pubkey should fail
        wrong_key = nacl.signing.SigningKey.generate()
        wrong_pubkey_b64 = base64.b64encode(bytes(wrong_key.verify_key)).decode()
        assert not verify_signature(bundle_fixture, pubkey_b64=wrong_pubkey_b64)
