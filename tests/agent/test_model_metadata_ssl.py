"""Tests for _resolve_requests_verify() env var precedence.

Verifies that custom provider `/models` fetches honour the three supported
CA bundle env vars (HERMES_CA_BUNDLE, REQUESTS_CA_BUNDLE, SSL_CERT_FILE)
in the documented priority order, and that non-existent paths are
skipped gracefully rather than breaking the request.

No filesystem or network I/O required — we use tmp_path to create real
CA bundle stand-in files and monkeypatch env vars.
"""

from pathlib import Path


import pytest

from agent.model_metadata import _resolve_requests_verify


_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


@pytest.fixture
def clean_env(monkeypatch):
    """Clear all three SSL env vars so each test starts from a known state."""
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def bundle_file(tmp_path: Path) -> str:
    """Create a placeholder CA bundle file and return its absolute path."""
    path = tmp_path / "ca.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\nstub\n-----END CERTIFICATE-----\n")
    return str(path)


class TestResolveRequestsVerify:
    def test_no_env_returns_true(self, clean_env):
        assert _resolve_requests_verify() is True




    def test_priority_hermes_over_requests(self, clean_env, tmp_path, bundle_file):
        other = tmp_path / "other.pem"
        other.write_text("stub")
        clean_env.setenv("HERMES_CA_BUNDLE", bundle_file)
        clean_env.setenv("REQUESTS_CA_BUNDLE", str(other))
        assert _resolve_requests_verify() == bundle_file




