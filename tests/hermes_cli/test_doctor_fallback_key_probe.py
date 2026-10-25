"""Regression coverage for region-specific API-key doctor probes (#136791)."""

from __future__ import annotations

from hermes_cli import doctor_connectivity as dc


def test_apikey_probe_skips_profile_when_only_fallback_key_is_configured(monkeypatch):
    """A runtime fallback key must not make doctor probe another region's endpoint."""
    monkeypatch.delenv("REGION_API_KEY", raising=False)
    monkeypatch.setenv("SHARED_API_KEY", "international-key")

    result = dc._probe_apikey_provider(
        "Regional provider",
        ("REGION_API_KEY", "SHARED_API_KEY"),
        "https://region.example.test/v1/models",
        None,
        True,
        False,
    )

    assert result.lines == []
    assert result.issues == []


def test_apikey_probe_names_fallback_key_in_invalid_key_hint(monkeypatch):
    monkeypatch.delenv("PRIMARY_API_KEY", raising=False)
    monkeypatch.setenv("FALLBACK_API_KEY", "invalid-key")

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: type("Response", (), {"status_code": 401})())
    result = dc._probe_apikey_provider(
        "Aliased provider",
        ("PRIMARY_API_KEY", "FALLBACK_API_KEY"),
        "https://provider.example.test/v1/models",
        None,
        True,
    )

    assert result.issues == ["Check FALLBACK_API_KEY in .env"]
