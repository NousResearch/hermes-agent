"""Tests for agent.ssl_compat — SSL/TLS compatibility layer."""

from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path
from unittest import mock

import pytest

from agent.ssl_compat import (
    _any_ca_env_var_set,
    _certifi_path,
    _detect_macos_high_sierra,
    resolve_httpx_verify,
    resolve_requests_verify,
    setup_ssl_compat,
)


@pytest.fixture(autouse=True)
def _clean_env():
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HERMES_CA_BUNDLE"):
        os.environ.pop(var, None)


@pytest.fixture
def fake_ca_bundle(tmp_path: Path) -> str:
    bundle = tmp_path / "ca-bundle.crt"
    try:
        import certifi
        content = Path(certifi.where()).read_text()
    except ImportError:
        content = "# Empty CA bundle\n"
    bundle.write_text(content)
    return str(bundle)


# =============================================================================
# macOS detection
# =============================================================================


@pytest.mark.parametrize(
    ("platform", "mac_ver", "expected"),
    [
        ("linux", ("", "", ""), False),
        ("darwin", ("14.5.0", ("", "", ""), "arm64"), False),
        ("darwin", ("10.13.6", ("", "", ""), "x86_64"), True),
        ("darwin", ("10.12.0", ("", "", ""), "x86_64"), True),
        ("darwin", ("", ("", "", ""), "arm64"), False),
    ],
)
def test_detect_macos_high_sierra(platform, mac_ver, expected):
    with (
        mock.patch.object(sys, "platform", platform),
        mock.patch("platform.mac_ver", return_value=mac_ver),
    ):
        assert _detect_macos_high_sierra() is expected


# =============================================================================
# env var helpers
# =============================================================================


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"SSL_CERT_FILE": "/no/such/file.pem"}, False),
        ({"HERMES_CA_BUNDLE": ""}, False),
    ],
)
def test_any_ca_env_var_not_set(env, expected):
    os.environ.update(env)
    assert _any_ca_env_var_set() is expected


def test_any_ca_env_var_set_true(fake_ca_bundle):
    os.environ["SSL_CERT_FILE"] = fake_ca_bundle
    assert _any_ca_env_var_set() is True


# =============================================================================
# certifi path
# =============================================================================


def test_certifi_path():
    path = _certifi_path()
    if path:
        assert os.path.isfile(path)
    else:
        assert path is None


def test_certifi_path_missing():
    with mock.patch.dict("sys.modules", {"certifi": None}):
        with mock.patch("builtins.__import__", side_effect=ImportError):
            assert _certifi_path() is None


# =============================================================================
# setup_ssl_compat
# =============================================================================


def test_skips_when_env_var_already_set(fake_ca_bundle):
    os.environ["SSL_CERT_FILE"] = fake_ca_bundle
    setup_ssl_compat()
    assert os.environ["SSL_CERT_FILE"] == fake_ca_bundle


def test_respects_custom_ca_bundle_config(fake_ca_bundle):
    setup_ssl_compat({"tls": {"ca_bundle": fake_ca_bundle}})
    assert os.environ["SSL_CERT_FILE"] == fake_ca_bundle


def test_custom_ca_bundle_not_found_does_not_crash():
    setup_ssl_compat({"tls": {"ca_bundle": "/no/such/file.pem"}})


def test_insecure_config_returns_without_setting_env():
    setup_ssl_compat({"tls": {"insecure": True}})
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HERMES_CA_BUNDLE"):
        assert not os.environ.get(var)


def test_macos_10_13_no_certifi_does_not_crash():
    with (
        mock.patch("agent.ssl_compat._detect_macos_high_sierra", return_value=True),
        mock.patch("agent.ssl_compat._certifi_path", return_value=None),
    ):
        setup_ssl_compat()
        assert not any(os.environ.get(v) for v in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HERMES_CA_BUNDLE"))


def test_sets_env_vars_when_certifi_available():
    """setup_ssl_compat() sets env vars only on macOS 10.13 when certifi is installed."""
    certifi_path = _certifi_path()
    if not certifi_path:
        pytest.skip("certifi not installed")
    setup_ssl_compat()
    if _detect_macos_high_sierra():
        assert os.environ.get("SSL_CERT_FILE") == certifi_path
        assert os.environ.get("REQUESTS_CA_BUNDLE") == certifi_path
    else:
        assert os.environ.get("SSL_CERT_FILE") != certifi_path
        assert os.environ.get("REQUESTS_CA_BUNDLE") != certifi_path


# =============================================================================
# resolve_requests_verify
# =============================================================================


def test_resolve_requests_default():
    result = resolve_requests_verify()
    assert result is True or (isinstance(result, str) and os.path.isfile(result))


def test_resolve_requests_insecure():
    assert resolve_requests_verify({"tls": {"insecure": True}}) is False


def test_resolve_requests_ca_bundle(fake_ca_bundle):
    assert resolve_requests_verify({"tls": {"ca_bundle": fake_ca_bundle}}) == fake_ca_bundle


def test_resolve_requests_env_var(fake_ca_bundle):
    os.environ["REQUESTS_CA_BUNDLE"] = fake_ca_bundle
    assert resolve_requests_verify() == fake_ca_bundle


# =============================================================================
# resolve_httpx_verify
# =============================================================================


def test_resolve_httpx_default():
    result = resolve_httpx_verify()
    assert result is True or isinstance(result, ssl.SSLContext)


def test_resolve_httpx_insecure():
    assert resolve_httpx_verify({"tls": {"insecure": True}}) is False


def test_resolve_httpx_ca_bundle(fake_ca_bundle):
    result = resolve_httpx_verify({"tls": {"ca_bundle": fake_ca_bundle}})
    assert isinstance(result, ssl.SSLContext)


def test_resolve_httpx_env_var(fake_ca_bundle):
    os.environ["SSL_CERT_FILE"] = fake_ca_bundle
    result = resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)
