"""Probe-error redaction for ``hermes doctor`` (regression for #144458).

``hermes doctor`` used to interpolate raw HTTP-client exception text into
diagnostics. A key with an embedded line break (bad paste) makes httpx raise
with the full value in the message, which doctor then printed — and operators
paste diagnostics into issues and screenshots. These tests drive the real
header validation against loopback for the Bearer / Key / x-api-key / xi-api-key
paths and assert the value never reaches the row or the collected issues.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import doctor_connectivity as dc
from hermes_cli import doctor_live as dl

# Distinctive synthetic credential: the alnum stem must never appear in output.
_STEM = "OpaqueSyntheticCredentialA1B2C3D4E5"
_BAD_PASTE = _STEM + "\nmultiline-paste"


@pytest.fixture
def loopback_url():
    """A bound-but-never-served loopback URL: the kernel accepts the TCP
    handshake, then real httpx/httpcore header validation raises."""
    with HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler) as server:
        yield f"http://127.0.0.1:{server.server_port}/models"


def _row_detail(result):
    ((_, _, detail),) = result.lines
    return detail


def test_apikey_probe_malformed_key_names_format_without_value(monkeypatch, loopback_url):
    monkeypatch.setenv("AUDIT_API_KEY", _BAD_PASTE)
    result = dc._probe_apikey_provider("Audit", ("AUDIT_API_KEY",), loopback_url, None, True)
    detail = _row_detail(result)
    assert _STEM not in detail and _STEM not in "".join(result.issues)
    assert "AUDIT_API_KEY" in detail and "line break" in detail


def test_openrouter_probe_malformed_key_redacted(monkeypatch, loopback_url):
    monkeypatch.setenv("OPENROUTER_API_KEY", _BAD_PASTE)
    monkeypatch.setattr(dc, "OPENROUTER_MODELS_URL", loopback_url)
    result = dc._probe_openrouter()
    detail = _row_detail(result)
    assert _STEM not in detail and _STEM not in "".join(result.issues)
    assert "OPENROUTER_API_KEY" in detail


def test_github_probe_malformed_token_redacted(monkeypatch, tmp_path, loopback_url):
    from urllib.parse import urlsplit

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GITHUB_TOKEN", _BAD_PASTE)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    path = urlsplit(dc.GITHUB_API_PROBE_URL).path
    monkeypatch.setattr(dc, "GITHUB_API_PROBE_URL", loopback_url + path)
    result = dc._probe_github_token()
    detail = _row_detail(result)
    assert _STEM not in detail and _STEM not in "".join(result.issues)


def test_anthropic_probe_malformed_key_redacted(monkeypatch, loopback_url):
    import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", _BAD_PASTE)
    real_get = httpx.get

    def fake_get(url, **kwargs):
        return real_get(loopback_url, **kwargs)

    monkeypatch.setattr(httpx, "get", fake_get)
    result = dc._probe_anthropic()
    detail = _row_detail(result)
    assert _STEM not in detail


def test_ordinary_probe_error_keeps_useful_context(monkeypatch):
    import httpx

    monkeypatch.setenv("AUDIT_API_KEY", "ordinary-key-without-newline")

    def fake_get(url, **kwargs):
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(httpx, "get", fake_get)
    result = dc._probe_apikey_provider("Audit", ("AUDIT_API_KEY",), "http://127.0.0.1:9/models", None, True)
    assert _row_detail(result) == "(connection reset by peer)"


def test_scrub_replaces_exact_secret_value():
    err = RuntimeError(f"reset while sending Bearer {_STEM}")
    assert dc.scrub_probe_error(err, _STEM) == "(reset while sending Bearer ***)"


def test_scrub_ignores_empty_secrets():
    assert dc.scrub_probe_error(RuntimeError("connection reset"), "", None) == "(connection reset)"


def test_keyed_probe_malformed_key_redacted(monkeypatch, loopback_url):
    monkeypatch.setenv("FAL_KEY", _BAD_PASTE)
    result = dl._keyed_probe("FAL", loopback_url, "FAL_KEY", "Key", 5.0)
    assert result.status == "fail"
    assert _STEM not in result.detail


def test_audio_probe_malformed_key_redacted(monkeypatch, loopback_url):
    monkeypatch.setenv("ELEVENLABS_API_KEY", _BAD_PASTE)
    monkeypatch.setattr(dl, "_AUDIO_PROBES", {"elevenlabs": (loopback_url, "ELEVENLABS_API_KEY", "xi")})
    result = dl._probe_audio("elevenlabs", {"elevenlabs": {"provider": "elevenlabs"}}, 5.0)
    assert result.status == "fail"
    assert _STEM not in result.detail


def test_run_one_scrubs_known_live_secret(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-live-secret-value")

    def boom():
        raise RuntimeError("boom while sending fc-live-secret-value")

    result = dl._run_one("Firecrawl", boom, [])
    assert result.status == "fail"
    assert "fc-live-secret-value" not in result.detail


def test_run_one_ordinary_error_unchanged(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

    def boom():
        raise RuntimeError("connection reset")

    result = dl._run_one("Firecrawl", boom, [])
    assert result.detail == "(connection reset)"