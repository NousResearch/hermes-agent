"""Tests for agent.model_metadata._resolve_requests_verify with windows_merged_ca_bundle."""

import certifi
import pytest

from agent.model_metadata import _resolve_requests_verify

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


@pytest.fixture
def clean_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_merged_bundle_default_returns_path(clean_env, monkeypatch, tmp_path):
    bundle_path = tmp_path / "merged_ca_bundle.pem"
    import pathlib
    certifi_path_str = certifi.where()
    bundle_path.write_text(pathlib.Path(certifi_path_str).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: str(bundle_path))
    result = _resolve_requests_verify()
    assert result == str(bundle_path)


def test_explicit_env_wins(clean_env, monkeypatch):
    env_bundle = certifi.where()
    monkeypatch.setenv("HERMES_CA_BUNDLE", env_bundle)
    other_bundle = "/fake/other.pem"
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: other_bundle)
    result = _resolve_requests_verify()
    assert result == env_bundle


def test_merged_none_keeps_true(clean_env, monkeypatch):
    monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: None)
    assert _resolve_requests_verify() is True
