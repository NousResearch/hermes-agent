"""``options.num_ctx`` is Ollama's own request extension - only an Ollama endpoint may receive it.

A local compatibility gateway answers ``/api/tags`` and ``/api/show`` in Ollama's shape (dotted
``general.*`` keys included), so a probe-based verdict marks it Ollama and the field then reaches
a relayed non-Ollama model - Gemini rejects the unknown field with a 400 on EVERY request
(#142239). The endpoint's SIGNATURE is the authority; an explicit ``model.ollama_num_ctx`` is the
operator's own opt-in and wins over it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """The registered custom profile (plugin discovery runs on the import)."""
    import model_tools
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


_OLLAMA_URL = "http://127.0.0.1:11434/v1"
_RELAY_URL = "http://127.0.0.1:8080/v1"


class TestOllamaOptionsInjectionGate:
    def test_a_relay_endpoint_gets_no_ollama_options(self, custom_profile):
        """The reproduction: a local gateway that only imitates Ollama must not receive the field."""
        eb, _tl = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=1048576, model="gemini-3-pro", base_url=_RELAY_URL,
        )
        assert eb == {}, "options.num_ctx is Ollama-only; a relay rejects the unknown field"

    def test_an_ollama_endpoint_still_gets_the_options(self, custom_profile):
        """A real Ollama (the default port) keeps the window it was asked to serve."""
        eb, _tl = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=65536, model="llama3.1:8b", base_url=_OLLAMA_URL,
        )
        assert eb == {"options": {"num_ctx": 65536}}

    def test_an_explicit_window_is_the_operators_opt_in(self, custom_profile):
        """``model.ollama_num_ctx`` is explicit: it goes out even to an endpoint we cannot place."""
        eb, _tl = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=65536, ollama_num_ctx_explicit=True,
            model="llama3.1:8b", base_url=_RELAY_URL,
        )
        assert eb == {"options": {"num_ctx": 65536}}

    def test_without_a_base_url_the_previous_behaviour_is_kept(self, custom_profile):
        """An endpoint we cannot place (no base_url) keeps the previous behaviour."""
        eb, _tl = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=4096, model="llama3.1:8b",
        )
        assert eb == {"options": {"num_ctx": 4096}}

    def test_no_window_means_no_options(self, custom_profile):
        """The control: nothing to send, nothing sent."""
        eb, _tl = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=None, model="llama3.1:8b", base_url=_OLLAMA_URL,
        )
        assert eb == {}
