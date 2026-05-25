"""Tests for hermes secrets CLI — fixture-based, no real ~/.hermes dependency."""
import json
import os
import tempfile
import subprocess
import textwrap
from unittest import mock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_home():
    """Temporary HERMES_HOME with manifest, .env, and sidecar."""
    with tempfile.TemporaryDirectory() as tmp:
        hermes_home = os.path.join(tmp, ".hermes")
        os.makedirs(hermes_home, exist_ok=True)
        yield hermes_home


@pytest.fixture
def manifest_path(temp_home):
    """Create a minimal test manifest."""
    path = os.path.join(temp_home, "secrets-manifest.yaml")
    manifest = {
        "schema_version": 1,
        "bws_project": {"name": "test", "id": "fake-project-id"},
        "targets": [
            {
                "name": "root/default",
                "path": os.path.join(temp_home, ".env"),
                "kind": "runtime-env",
                "mode": "0600",
                "keys": [
                    {
                        "canonical": "TEST_TOKEN",
                        "projection": "TEST_TOKEN",
                        "source": {"type": "bws", "project": "test", "key": "TEST_TOKEN"},
                        "classification": "secret",
                        "required": True,
                    },
                    {
                        "canonical": "TEST_KEY",
                        "projection": "TEST_KEY",
                        "source": {"type": "bws", "project": "test", "key": "TEST_KEY"},
                        "classification": "secret",
                        "required": False,
                    },
                ],
            }
        ],
        "provider_bindings": [
            {"profile": "arisu", "provider": "custom", "base_url": "https://api.deepseek.com",
             "credential_source": "ARISU_DEEPSEEK_API_KEY"}
        ],
    }
    with open(path, "w") as f:
        yaml.dump(manifest, f)
    return path


@pytest.fixture
def env_path(temp_home):
    """Create a test .env file."""
    path = os.path.join(temp_home, ".env")
    with open(path, "w") as f:
        f.write("TEST_TOKEN=real-secret-value-123\n")
        f.write("TEST_KEY=another-secret-456\n")
        f.write("UNMANAGED_CONFIG=some_value\n")
    return path


@pytest.fixture
def sidecar_path(env_path):
    """Create a sidecar with matching fingerprints."""
    import hashlib
    env = {}
    with open(env_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v

    sidecar = {
        "generated_by": "hermes secrets sync",
        "manifest_sha256": hashlib.sha256(open(env_path.replace(".env", "secrets-manifest.yaml") if False else "").read().encode()).hexdigest() if False else "fake-hash",
        "target": "root/default",
        "generated_at": "2026-05-25T00:00:00+00:00",
        "bws_project": "test",
        "keys": [
            {"canonical": "TEST_TOKEN", "projection": "TEST_TOKEN",
             "fingerprint": hashlib.sha256(env["TEST_TOKEN"].encode()).hexdigest()[:16],
             "status": "MATCH"},
            {"canonical": "TEST_KEY", "projection": "TEST_KEY",
             "fingerprint": hashlib.sha256(env["TEST_KEY"].encode()).hexdigest()[:16],
             "status": "MATCH"},
        ],
        "manual_edits_forbidden": True,
    }
    sp = f"{env_path}.hermes-projection.json"
    with open(sp, "w") as f:
        json.dump(sidecar, f)
    return sp


def _run_secrets(temp_home: str, *args: str) -> subprocess.CompletedProcess:
    """Run hermes secrets CLI with temporary home."""
    env = os.environ.copy()
    env["HERMES_HOME"] = temp_home
    env["HOME"] = os.path.expanduser("~")  # real home for bws
    return subprocess.run(
        ["python", "-m", "hermes_cli.main", "secrets"] + list(args),
        capture_output=True, text=True, timeout=30,
        env=env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSecretsCheck:
    def test_check_passes_with_matching_sidecar(self, temp_home, manifest_path, env_path, sidecar_path):
        """check passes when sidecar fingerprints match current .env values."""
        # Rewrite sidecar with correct manifest hash
        import hashlib
        mh = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["manifest_sha256"] = mh
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        result = _run_secrets(temp_home, "check")
        assert result.returncode == 0
        assert "MATCH" in result.stdout

    def test_check_fails_on_fingerprint_drift(self, temp_home, manifest_path, env_path, sidecar_path):
        """check fails when .env value differs from sidecar fingerprint."""
        import hashlib
        mh = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["manifest_sha256"] = mh
        # Tamper with .env
        sidecar["keys"][0]["fingerprint"] = "0000000000000000"  # wrong fingerprint
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        result = _run_secrets(temp_home, "check")
        assert result.returncode != 0
        assert "DRIFT" in result.stdout


class TestSecretsPreflight:
    def test_preflight_fails_on_drift(self, temp_home, manifest_path, env_path, sidecar_path):
        """preflight fails when fingerprint drift detected."""
        import hashlib
        mh = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["manifest_sha256"] = mh
        sidecar["keys"][0]["fingerprint"] = "0000000000000000"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        result = _run_secrets(temp_home, "preflight")
        assert result.returncode != 0
        assert "drift" in result.stdout.lower()

    def test_preflight_passes_clean(self, temp_home, manifest_path, env_path, sidecar_path):
        """preflight passes when everything matches."""
        import hashlib
        mh = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["manifest_sha256"] = mh
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        result = _run_secrets(temp_home, "preflight")
        assert "preflight passed" in result.stdout


class TestSecretsBreakGlass:
    def test_break_glass_ttl_is_future(self, temp_home, manifest_path):
        """break-glass expires_at should be in the future."""
        result = _run_secrets(temp_home, "break-glass", "--target", "root/default",
                              "--reason", "test", "--ttl", "3600")
        assert result.returncode == 0

        # Verify expires_at is future
        from datetime import datetime, timezone
        marker_dir = os.path.join(temp_home, "break-glass")
        marker_file = os.path.join(marker_dir, "root-default.json")
        assert os.path.exists(marker_file)
        with open(marker_file) as f:
            marker = json.load(f)
        expires = datetime.fromisoformat(marker["expires_at"])
        now = datetime.now(timezone.utc)
        assert expires > now


class TestNoSecretLeak:
    def test_no_secret_values_in_output(self, temp_home, manifest_path, env_path, sidecar_path):
        """No command should leak raw secret values."""
        import hashlib
        mh = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["manifest_sha256"] = mh
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

        commands = [
            ["check"],
            ["preflight"],
            ["provider-check", "--profile", "default"],
        ]
        for cmd in commands:
            result = _run_secrets(temp_home, *cmd)
            output = result.stdout + result.stderr
            # Real secret values from fixture must not appear
            assert "real-secret-value-123" not in output, f"Secret leaked in {cmd}"
            assert "another-secret-456" not in output, f"Secret leaked in {cmd}"
