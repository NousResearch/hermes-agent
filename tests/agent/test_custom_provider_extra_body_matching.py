"""Tests for custom-provider model matching (extra_body / service_tier drop bug).

July 2026 incident: a custom provider with a multi-model catalog
(``models: {gpt-5.5: {}, gpt-5.6-terra: {}, ...}``) and a ``model``/
``default_model`` differing from the session model failed
``_custom_provider_model_matches``, so ``extra_body: {service_tier: flex}``
was silently dropped — every request billed at standard tier (~2.3x).
"""

from agent.agent_init import (
    _custom_provider_extra_body_for_agent,
    _custom_provider_model_matches,
)

BASE = "https://api.openai.com/v1"


def _entry(**over):
    e = {
        "name": "openai",
        "base_url": BASE,
        "extra_body": {"service_tier": "flex"},
    }
    e.update(over)
    return e


class TestModelMatches:
    def test_models_dict_catalog_matches_session_model(self):
        e = _entry(model="gpt-5.5", models={"gpt-5.5": {}, "gpt-5.6-terra": {}})
        assert _custom_provider_model_matches("gpt-5.6-terra", e)

    def test_models_list_catalog_matches(self):
        e = _entry(model="gpt-5.5", models=["gpt-5.5", "gpt-5.6-sol"])
        assert _custom_provider_model_matches("gpt-5.6-sol", e)

    def test_catalog_miss_falls_back_to_model_field(self):
        e = _entry(model="gpt-5.5", models={"gpt-5.5": {}})
        assert _custom_provider_model_matches("gpt-5.5", e)
        assert not _custom_provider_model_matches("gpt-4o", e)

    def test_no_model_no_catalog_matches_everything(self):
        e = _entry()
        assert _custom_provider_model_matches("anything", e)

    def test_catalog_case_insensitive(self):
        e = _entry(models={"GPT-5.6-Terra": {}})
        assert _custom_provider_model_matches("gpt-5.6-terra", e)


class TestExtraBodyResolution:
    def test_multi_model_provider_yields_extra_body(self):
        # The exact sweeper-profile shape that failed in production.
        entry = _entry(
            model="gpt-5.5",
            models={"gpt-5.5": {}, "gpt-5.6-sol": {}, "gpt-5.6-terra": {}},
        )
        got = _custom_provider_extra_body_for_agent(
            provider="custom", model="gpt-5.6-terra",
            base_url=BASE, custom_providers=[entry],
        )
        assert got == {"service_tier": "flex"}

    def test_non_catalog_model_gets_no_override(self):
        entry = _entry(model="gpt-5.5", models={"gpt-5.5": {}})
        got = _custom_provider_extra_body_for_agent(
            provider="custom", model="gpt-4o",
            base_url=BASE, custom_providers=[entry],
        )
        assert got is None

    def test_non_custom_provider_unaffected(self):
        entry = _entry(models={"gpt-5.6-terra": {}})
        got = _custom_provider_extra_body_for_agent(
            provider="openrouter", model="gpt-5.6-terra",
            base_url=BASE, custom_providers=[entry],
        )
        assert got is None
