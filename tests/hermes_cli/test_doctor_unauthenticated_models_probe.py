"""Unauthenticated /models baseline for generic API-key doctor probes.

Some providers (e.g. alibaba-coding-plan) serve GET /models with HTTP 200
without checking the key. An authenticated 200 alone must not be reported
as ✓; a second GET with credential headers stripped detects that case.

See: NousResearch/hermes-agent#107473
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx

from hermes_cli.doctor_connectivity import _GLYPH, _probe_apikey_provider

_WARN = _GLYPH["warn"][0]
_OK = _GLYPH["ok"][0]
_FAIL = _GLYPH["fail"][0]

_PNAME = "DeepSeek"
_ENV = ("DEEPSEEK_API_KEY",)
_URL = "https://api.deepseek.com/v1/models"
_BASE_ENV = "DEEPSEEK_BASE_URL"


def _probe(monkeypatch, *, supports_health_check=True):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    return _probe_apikey_provider(_PNAME, _ENV, _URL, _BASE_ENV, supports_health_check)


def _glyph(result):
    return result.lines[0][0]


def _detail(result):
    return result.lines[0][2]


def test_unauthenticated_models_200_is_warn_not_ok(monkeypatch):
    """Detection: /models returns 200 with or without Authorization → warn."""
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, dict(headers or {}), timeout))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _probe(monkeypatch)

    assert _WARN in _glyph(result), f"expected warn glyph, got {_glyph(result)!r} detail={_detail(result)!r}"
    assert _OK not in _glyph(result)
    detail = _detail(result).lower()
    assert "not verified" in detail or "unauthenticated" in detail or "does not authenticate" in detail
    assert any("Authorization" not in headers and "x-goog-api-key" not in headers for _, headers, _ in calls)


def test_authenticated_200_unauth_401_stays_ok(monkeypatch):
    """Fail-open CONTROL: endpoint that does authenticate stays ✓."""

    def fake_get(url, headers=None, timeout=None):
        headers = headers or {}
        if "Authorization" in headers or "x-goog-api-key" in headers:
            return SimpleNamespace(status_code=200)
        return SimpleNamespace(status_code=401)

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _probe(monkeypatch)

    assert _OK in _glyph(result)
    assert _WARN not in _glyph(result)
    assert result.issues == []


def test_authenticated_401_still_fail_invalid_key(monkeypatch):
    """Existing fail path: authenticated 401 is ✗, no baseline required."""
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(dict(headers or {}))
        return SimpleNamespace(status_code=401)

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _probe(monkeypatch)

    assert _FAIL in _glyph(result)
    assert "(invalid API key)" in _detail(result)
    assert any("Authorization" in headers or "x-goog-api-key" in headers for headers in calls)


def test_supports_health_check_false_skips_http(monkeypatch):
    """Presence-only rows stay ✓ (key configured) with no HTTP."""
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("supports_health_check=False must not probe /models")

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _probe(monkeypatch, supports_health_check=False)

    assert _OK in _glyph(result)
    assert "(key configured)" in _detail(result)
    assert calls == []
