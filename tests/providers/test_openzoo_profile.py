"""Tests for the bundled openzoo provider plugin.

openzoo is a local proxy (``npx openzoo``) that pays for inference per call
over x402; Hermes talks to it as a plain OpenAI-compatible endpoint on
localhost. No live network calls — the catalog fetch is exercised through a
stubbed ``open_credentialed_url``.
"""

import io
import json

import pytest

from providers import get_provider_profile


def _profile():
    p = get_provider_profile("openzoo")
    assert p is not None
    return p


class TestOpenZooProfile:
    def test_profile_registered(self):
        p = _profile()
        assert p.name == "openzoo"
        assert p.base_url == "http://localhost:8402/v1"
        assert p.auth_type == "api_key"
        assert p.api_mode == "chat_completions"
        assert p.env_vars == ("OPENZOO_API_KEY", "OPENZOO_BASE_URL")
        assert p.default_aux_model == "openzoo/auto"
        assert p.fallback_models[0] == "openzoo/auto"
        assert "anthropic/claude-sonnet-5" in p.fallback_models

    @pytest.mark.parametrize("alias", ["open-zoo", "zoo"])
    def test_aliases_resolve(self, alias):
        assert get_provider_profile(alias) is _profile()

    def test_base_url_is_local_proxy_not_gateway(self):
        """Only the local proxy can settle the gateway's 402s."""
        assert get_provider_profile("openzoo").get_hostname() == "localhost"


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _stub_catalog(monkeypatch, payload, *, seen=None):
    import hermes_cli.urllib_security as sec

    def fake_open(request, *, timeout, **kwargs):
        if seen is not None:
            seen.append(request)
        return _FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(sec, "open_credentialed_url", fake_open)


class TestFetchModels:
    def test_skips_media_rows(self, monkeypatch):
        seen = []
        _stub_catalog(
            monkeypatch,
            {
                "data": [
                    {"id": "openzoo/auto", "name": None, "context_length": 128000000},
                    {"id": "anthropic/claude-sonnet-5"},
                    {"id": "black-forest-labs/flux-2", "kind": "image"},
                    {"id": "google/veo-3.1", "kind": "video"},
                    {"id": ""},
                    "not-a-row",
                ]
            },
            seen=seen,
        )
        models = _profile().fetch_models(api_key="sk-openzoo")
        assert models == ["openzoo/auto", "anthropic/claude-sonnet-5"]
        assert seen[0].full_url == "http://localhost:8402/v1/models"
        # The catalog is free — no bearer is sent for it.
        assert not seen[0].has_header("Authorization")

    def test_honours_custom_base_url(self, monkeypatch):
        seen = []
        _stub_catalog(monkeypatch, {"data": [{"id": "x-ai/grok-4.6"}]}, seen=seen)
        models = _profile().fetch_models(
            api_key="sk-openzoo", base_url="http://127.0.0.1:8412/v1/"
        )
        assert models == ["x-ai/grok-4.6"]
        assert seen[0].full_url == "http://127.0.0.1:8412/v1/models"

    def test_fetch_failure_returns_none(self, monkeypatch):
        import hermes_cli.urllib_security as sec

        def boom(request, *, timeout, **kwargs):
            raise ConnectionRefusedError("proxy not running")

        monkeypatch.setattr(sec, "open_credentialed_url", boom)
        assert _profile().fetch_models(api_key="sk-openzoo") is None
