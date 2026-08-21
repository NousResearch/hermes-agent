"""Tests for the Inworld LLM router provider profile."""

import json
import sys
from unittest.mock import patch

import pytest

from providers import get_provider_profile


@pytest.fixture
def profile():
    """The registered Inworld profile (discovery runs on first lookup)."""
    return get_provider_profile("inworld")


@pytest.fixture
def inworld(profile):
    """The plugin module, reached through the profile the registry loaded."""
    module = sys.modules[type(profile).__module__]
    module.clear_cache()
    yield module
    module.clear_cache()


def _entry(provider="openai", model="gpt-5.2", **spec):
    """Build a catalog entry with an overridable ``spec``."""
    base = {
        "contextLength": 272000,
        "maxCompletionTokens": 128000,
        "inputModalities": ["text", "image"],
        "outputModalities": ["text"],
        "capabilities": {"functionCalling": True, "reasoning": True},
    }
    base.update(spec)
    return {"model": model, "provider": provider, "isSupported": True, "spec": base}


class _FakeResponse:
    """Minimal context manager returning a fixed JSON body."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return self._body


class TestRegistration:
    def test_profile_is_registered(self, profile):
        assert profile is not None
        assert profile.name == "inworld"

    def test_aliases_resolve(self, profile):
        assert get_provider_profile("inworld-ai") is profile
        assert get_provider_profile("inworld-router") is profile

    def test_openai_compatible_endpoint(self, profile):
        assert profile.base_url == "https://api.inworld.ai/v1"
        assert profile.api_mode == "chat_completions"
        assert profile.get_hostname() == "api.inworld.ai"

    def test_router_model_is_the_offline_fallback(self, profile, inworld):
        """The picker stays usable when the catalog is unreachable."""
        assert profile.fallback_models == (inworld.ROUTER_MODEL,)

    def test_aux_model_is_a_qualified_catalog_id(self, profile):
        """Aux resolution is synchronous, so this id is sent as-is."""
        assert profile.default_aux_model == "inworld/models/gemma-4-26b-a4b-it"


class TestParseCatalog:
    def test_ids_join_provider_and_model(self, inworld):
        catalog = inworld.parse_catalog(
            {
                "models": [
                    _entry(provider="deepinfra", model="deepseek-ai/DeepSeek-V3.2"),
                    _entry(provider="inworld", model="models/gemma-4-31b-it"),
                ]
            }
        )

        assert catalog == [
            "deepinfra/deepseek-ai/DeepSeek-V3.2",
            "inworld/models/gemma-4-31b-it",
        ]

    def test_unsupported_entries_are_skipped(self, inworld):
        entry = _entry()
        entry["isSupported"] = False

        assert inworld.parse_catalog({"models": [entry]}) == []

    def test_models_without_tool_calling_are_skipped(self, inworld):
        """A model that cannot call tools cannot drive the agent."""
        catalog = inworld.parse_catalog(
            {"models": [_entry(capabilities={"functionCalling": False})]}
        )

        assert catalog == []

    def test_unstated_tool_calling_is_not_guessed(self, inworld):
        catalog = inworld.parse_catalog({"models": [_entry(capabilities={})]})

        assert catalog == []

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"models": None}, {"models": [None]}, {"models": [{}]}],
    )
    def test_malformed_payloads_yield_no_models(self, inworld, payload):
        assert inworld.parse_catalog(payload) == []


class TestFetchModels:
    def _fetch(self, profile, payload, **kwargs):
        """Run ``fetch_models`` against a stubbed catalog, returning the request."""
        captured = {}

        def _open(request, *, timeout):
            captured["request"] = request
            return _FakeResponse(payload)

        with patch("hermes_cli.urllib_security.open_credentialed_url", _open):
            models = profile.fetch_models(api_key="k", **kwargs)
        return models, captured.get("request")

    def test_catalog_path_replaces_the_v1_prefix(self, profile, inworld):
        """The catalog sits outside ``/v1``, so its path replaces base_url's."""
        _, request = self._fetch(profile, {"models": [_entry()]})

        assert request.full_url == "https://api.inworld.ai/llm/v1alpha/models"

    def test_custom_base_url_follows_its_own_host(self, profile, inworld):
        _, request = self._fetch(
            profile, {"models": [_entry()]}, base_url="https://staging.example.com/v1"
        )

        assert request.full_url == "https://staging.example.com/llm/v1alpha/models"

    def test_catalog_uses_basic_auth(self, profile, inworld):
        """Bearer is built by the OpenAI client on the completions path only."""
        _, request = self._fetch(profile, {"models": [_entry()]})

        assert request.get_header("Authorization") == "Basic k"

    def test_router_model_leads_the_list(self, profile, inworld):
        models, _ = self._fetch(profile, {"models": [_entry()]})

        assert models == [inworld.ROUTER_MODEL, "openai/gpt-5.2"]

    def test_missing_credential_skips_the_fetch(self, profile, inworld):
        """The catalog requires auth; an anonymous probe would only 401."""
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=AssertionError("network touched"),
        ):
            assert profile.fetch_models(api_key=None) is None

    def test_transport_failure_degrades_to_fallback(self, profile, inworld):
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=OSError("unreachable"),
        ):
            assert profile.fetch_models(api_key="k") is None

    def test_empty_catalog_degrades_to_fallback(self, profile, inworld):
        models, _ = self._fetch(profile, {"models": []})

        assert models is None

    def test_result_is_cached_per_url(self, profile, inworld):
        first, _ = self._fetch(profile, {"models": [_entry()]})

        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=AssertionError("refetched"),
        ):
            second = profile.fetch_models(api_key="k")

        assert second == first

    def test_cache_expires_so_new_models_appear_without_a_restart(
        self, profile, inworld, monkeypatch
    ):
        clock = [1000.0]
        monkeypatch.setattr(inworld.time, "monotonic", lambda: clock[0])

        self._fetch(profile, {"models": [_entry()]})
        clock[0] += inworld._CACHE_TTL_SECONDS + 1
        refreshed, _ = self._fetch(
            profile, {"models": [_entry(model="gpt-6")]}
        )

        assert refreshed == [inworld.ROUTER_MODEL, "openai/gpt-6"]

    @pytest.mark.parametrize("key", ["abc\r\nX-Injected: 1", "abc def", "ké"])
    def test_keys_that_cannot_form_a_header_are_rejected(
        self, profile, inworld, key
    ):
        """A malformed key is named, not sent as an opaque failed request."""
        with patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=AssertionError("network touched"),
        ):
            assert profile.fetch_models(api_key=key) is None

    def test_empty_catalog_warns_while_transport_failure_does_not(
        self, profile, inworld, caplog
    ):
        """An operator can tell 'no usable models' from 'network down'."""
        with caplog.at_level("WARNING"):
            self._fetch(profile, {"models": []})
        assert any("no tool-calling models" in r.message for r in caplog.records)

        caplog.clear()
        with caplog.at_level("WARNING"), patch(
            "hermes_cli.urllib_security.open_credentialed_url",
            side_effect=OSError("unreachable"),
        ):
            profile.fetch_models(api_key="k")
        assert not caplog.records


class TestResolveAuxModel:
    def _fetch(self, profile, payload):
        def _open(request, *, timeout):
            return _FakeResponse(payload)

        with patch("hermes_cli.urllib_security.open_credentialed_url", _open):
            profile.fetch_models(api_key="k")

    def test_no_opinion_without_a_cached_catalog(self, profile, inworld):
        """Falls through to default_aux_model rather than guessing."""
        assert profile.resolve_aux_model() == ""

    def test_no_opinion_while_the_pinned_model_is_live(self, profile, inworld):
        self._fetch(
            profile,
            {"models": [_entry(provider="inworld", model="models/gemma-4-26b-a4b-it")]},
        )

        assert profile.resolve_aux_model() == ""

    def test_retired_aux_model_falls_back_to_the_router(self, profile, inworld):
        """Aux calls survive Inworld retiring the pinned model."""
        self._fetch(profile, {"models": [_entry()]})

        assert profile.resolve_aux_model() == inworld.ROUTER_MODEL
