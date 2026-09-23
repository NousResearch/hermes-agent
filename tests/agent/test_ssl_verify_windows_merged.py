"""Tests for agent.ssl_verify.resolve_httpx_verify with windows_merged_ca_bundle."""

import ssl
import certifi
import pytest

from agent.ssl_verify import resolve_httpx_verify, _context_for_ca_bundle

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


@pytest.fixture
def clean_ca_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_merged_bundle_default_returns_context(clean_ca_env, monkeypatch, tmp_path):
    """When windows_merged_ca_bundle returns a real PEM file, resolve_httpx_verify returns an SSLContext."""
    bundle_path = tmp_path / "merged_ca_bundle.pem"
    # Create a minimal valid PEM file (certifi's bundle content works)
    certifi_path_str = certifi.where()
    bundle_path.write_text(__import__("pathlib").Path(certifi_path_str).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: str(bundle_path))
    result = resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)
    # Identity contract: same path => same shared SSLContext
    assert result is _context_for_ca_bundle(str(bundle_path))


def test_explicit_env_wins_over_merged_bundle(clean_ca_env, monkeypatch, tmp_path):
    """Explicit HERMES_CA_BUNDLE (certifi) takes priority over merged bundle hook."""
    env_bundle = certifi.where()
    monkeypatch.setenv("HERMES_CA_BUNDLE", env_bundle)
    other_bundle = tmp_path / "other_merged.pem"
    import pathlib
    certifi_path_obj = pathlib.Path(certifi.where())
    other_bundle.write_text(certifi_path_obj.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: str(other_bundle))
    result = resolve_httpx_verify()
    # Must be the context for the ENV path, not the merged hook path
    assert result is _context_for_ca_bundle(env_bundle)


def test_merged_none_keeps_today_behavior(clean_ca_env, monkeypatch):
    """When windows_merged_ca_bundle returns None, result stays True (current behavior)."""
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: None)
    result = resolve_httpx_verify()
    assert result is True


def test_ssl_verify_false_wins_over_merged_bundle(clean_ca_env, monkeypatch, tmp_path):
    """Explicit ssl_verify: false (insecure mode) beats the merged Windows bundle —
    the merged bundle is a DEFAULT, never an override."""
    bundle_path = tmp_path / "merged_ca_bundle.pem"
    import pathlib
    bundle_path.write_text(pathlib.Path(certifi.where()).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: str(bundle_path))
    assert resolve_httpx_verify(ssl_verify=False) is False


def test_seam_fail_open_when_merged_hook_raises(clean_ca_env, monkeypatch):
    """The merged-bundle lookup must never break client construction: any exception
    from the lookup falls through to True (fail-open contract)."""
    def _boom():
        raise RuntimeError("simulated bundle failure")

    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", _boom)
    assert resolve_httpx_verify() is True


def test_requests_seam_fail_open_when_merged_hook_raises(clean_ca_env, monkeypatch):
    """Mirror of the ssl_verify fail-open contract for the requests-probe seam."""
    from agent.model_metadata import _resolve_requests_verify

    def _boom():
        raise RuntimeError("simulated bundle failure")

    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", _boom)
    assert _resolve_requests_verify() is True
