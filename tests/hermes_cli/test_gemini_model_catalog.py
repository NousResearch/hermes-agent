"""Gemini (AI Studio) live picker catalog: native ``models.list`` filtered to agent chat models."""

from __future__ import annotations

import io
import json
import urllib.parse

import pytest

from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids
from providers import get_provider_profile

_PAGES = {
    "": {"models": [
        {"name": "models/gemini-4-argon", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/gemini-3.8-flash-tts", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-3.1-flash-image", "supportedGenerationMethods": ["generateContent"]},
    ], "nextPageToken": "p2"},
    "p2": {"models": [
        {"name": "models/gemini-pro-latest", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemma-4-31b-it", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/lyria-3.5", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/veo-3.1-generate-preview", "supportedGenerationMethods": ["predictLongRunning"]},
    ]},
}


def _serve(monkeypatch, *, fail_page: str | None = None):
    seen = []

    def fake_open(req, *, timeout, **_kwargs):
        url = urllib.parse.urlsplit(req.full_url)
        token = urllib.parse.parse_qs(url.query).get("pageToken", [""])[0]
        seen.append({"url": req.full_url, "key_header": req.get_header("X-goog-api-key")})
        if token == fail_page:
            raise OSError("page failed")
        return io.BytesIO(json.dumps(_PAGES[token]).encode())

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
    return seen


def test_native_catalog_lists_every_page_of_agent_chat_models(monkeypatch):
    seen = _serve(monkeypatch)
    profile = get_provider_profile("gemini")

    ids = profile.fetch_models(api_key="fixture-key", base_url=profile.base_url)

    # Media/TTS/embedding families drop out even though Google marks some generateContent; Gemma
    # stays out of the picker by design (free-tier-only on this API).
    assert ids == ["gemini-4-argon", "gemini-pro-latest"]
    assert [s["url"].split("?")[0] for s in seen] == [profile.base_url + "/models"] * 2
    # Header auth only: the key never rides in a URL that error text or logs could echo.
    assert all(s["key_header"] == "fixture-key" and "fixture-key" not in s["url"] for s in seen)


@pytest.mark.parametrize("fail_page", ["", "p2"])
def test_failed_catalog_page_keeps_the_curated_picker(monkeypatch, fail_page):
    _serve(monkeypatch, fail_page=fail_page)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider_id: {"provider": provider_id, "api_key": "fixture-key", "base_url": "", "source": "GEMINI_API_KEY"},
    )

    rows = provider_model_ids("gemini", force_refresh=True)

    assert set(_PROVIDER_MODELS["gemini"]) <= set(rows)
    assert "gemini-pro-latest" not in rows  # a truncated live list never half-replaces the catalog
