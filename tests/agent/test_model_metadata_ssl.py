"""Tests for _resolve_requests_verify() env var precedence.

Verifies that custom provider `/models` fetches honour the four supported
CA bundle env vars (HERMES_CA_BUNDLE, SSL_CERT_FILE, REQUESTS_CA_BUNDLE,
CURL_CA_BUNDLE) in the documented priority order, and that non-existent
paths are skipped gracefully rather than breaking the request.

The order mirrors ``agent.ssl_verify.resolve_httpx_verify`` so the
requests-based and httpx-based callsites cannot disagree about which
bundle wins inside a single process.

No filesystem or network I/O required — we use tmp_path to create real
CA bundle stand-in files and monkeypatch env vars.
"""

from pathlib import Path


import pytest

from agent.model_metadata import _resolve_requests_verify


_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Clear every SSL env var so each test starts from a known state."""
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





    @pytest.mark.parametrize("env_var", _CA_ENV_VARS)
    def test_each_supported_var_is_honoured(self, clean_env, bundle_file, env_var):
        """Every documented var must resolve — CURL_CA_BUNDLE was ignored
        despite `requests` honouring it natively and the docstring listing it."""
        clean_env.setenv(env_var, bundle_file)
        assert _resolve_requests_verify() == bundle_file

    def test_full_precedence_chain(self, clean_env, tmp_path):
        """Dropping the current winner must hand off to the next var in order."""
        paths = {}
        for var in _CA_ENV_VARS:
            path = tmp_path / f"{var.lower()}.pem"
            path.write_text("stub")
            paths[var] = str(path)
            clean_env.setenv(var, paths[var])

        for expected_winner in _CA_ENV_VARS:
            assert _resolve_requests_verify() == paths[expected_winner]
            clean_env.delenv(expected_winner)

        assert _resolve_requests_verify() is True

    def test_nonexistent_path_falls_through_to_next_var(
        self, clean_env, tmp_path, bundle_file
    ):
        clean_env.setenv("HERMES_CA_BUNDLE", str(tmp_path / "missing.pem"))
        clean_env.setenv("CURL_CA_BUNDLE", bundle_file)
        assert _resolve_requests_verify() == bundle_file
