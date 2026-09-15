"""Metadata drives UI choices, selection validation and both Gemini wires."""
import io
import json

import pytest

from agent import models_dev
from agent.gemini_catalog_reasoning import describe_thinking_control
from hermes_constants import parse_reasoning_effort
from providers import get_provider_profile
from providers.reasoning import resolve_provider_reasoning_config

from agent.transports.chat_completions import ChatCompletionsTransport


def select_reasoning(model, effort):
    return resolve_provider_reasoning_config(
        "gemini", model, parse_reasoning_effort(effort), explicit=True)


@pytest.fixture
def catalog(monkeypatch):
    entries = {
        "gemini-future-levels": [{"type": "effort", "values": ["low", "high"]}],
        "gemini-future-budget": [{"type": "toggle"}, {"type": "budget_tokens", "min": 512, "max": 24576}],
        "gemini-fixed": [], "gemini-unknown": None,
        "gemini-mandatory-budget": [{"type": "budget_tokens", "min": 128, "max": 32768}],
    }
    models = {key: {"reasoning": True, "reasoning_options": value, "tool_call": True,
                    "modalities": {"output": ["text"]}} for key, value in entries.items()}
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **kwargs: {"google": {"models": models}})
    return models


def test_metadata_drives_both_wires_and_rejects_impossible_choices(catalog):
    profile = get_provider_profile("gemini")
    descriptor = describe_thinking_control("gemini-future-levels")
    assert descriptor["reasoning_efforts"] == ["low", "high"]
    assert not descriptor["can_disable_reasoning"]
    for effort in ("low", "high"):
        config = select_reasoning("gemini-future-levels", effort)
        assert profile.build_extra_body(model="gemini-future-levels", reasoning_config=config) == {
            "thinking_config": {"includeThoughts": True, "thinkingLevel": effort}}
    with pytest.raises(ValueError):
        select_reasoning("gemini-future-levels", "none")
    for effort in ("budget:-1", "budget:512", "budget:4096", "budget:24576"):
        config = select_reasoning("gemini-future-budget", effort)
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="gemini-future-budget", messages=[{"role": "user", "content": "hi"}],
            provider_profile=profile, provider_name="gemini", base_url=profile.base_url + "/openai",
            reasoning_config=config)
        assert kwargs["extra_body"]["extra_body"]["google"]["thinking_config"] == {
            "include_thoughts": True, "thinking_budget": int(effort[7:])}
    for effort in ("budget:0", "budget:511", "budget:24577", "high"):
        with pytest.raises(ValueError):
            select_reasoning("gemini-future-budget", effort)
    assert profile.build_extra_body(model="gemini-future-budget", reasoning_config=parse_reasoning_effort("none")) == {
        "thinking_config": {"includeThoughts": False, "thinkingBudget": 0}}
    assert describe_thinking_control("gemini-mandatory-budget")["can_disable_reasoning"] is False
    assert describe_thinking_control("gemini-fixed")["reasoning_control"] == "default"
    assert describe_thinking_control("gemini-unknown")["reasoning_control"] == "unknown"
    catalog["gemma-4-fixture"] = {"reasoning": True, "reasoning_options": [{"type": "toggle"}]}
    assert profile.build_extra_body(model="gemma-4-fixture", reasoning_config={"enabled": False}) == {}
    assert profile.build_extra_body(model="gemini-unknown", reasoning_config={"enabled": False}) == {}
    # A metadata update changes options without a release or family-name guesses.
    catalog["gemini-future-levels"]["reasoning_options"][0]["values"] = ["medium"]
    assert profile.supported_reasoning_efforts("gemini-future-levels") == ("medium",)


def test_native_catalog_pagination_filters_and_never_returns_partial(catalog, monkeypatch):
    import agent.gemini_model_catalog as native
    monkeypatch.setattr(native, "fetch_models_dev", models_dev.fetch_models_dev)
    catalog["gemini-future-computer-use-preview"] = catalog["gemini-future-levels"]
    pages = [
        {"models": [{"name": "models/gemini-future-levels", "supportedGenerationMethods": ["generateContent"]}],
         "nextPageToken": "opaque +/token"},
        {"models": [{"name": "models/gemini-future-budget", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/gemini-future-computer-use-preview", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/gemini-fixed", "supportedGenerationMethods": ["embedContent"]}]},
    ]
    requests = []
    def open_url(req, **kwargs):
        requests.append(req)
        return io.StringIO(json.dumps(pages[len(requests) - 1]))
    monkeypatch.setattr(native, "open_credentialed_url", open_url)
    assert native.fetch_models("fixture-secret") == ["gemini-future-levels", "gemini-future-budget"]
    assert "pageToken=opaque+%2B%2Ftoken" in requests[1].full_url
    assert "fixture-secret" not in requests[0].full_url
    assert requests[0].get_header("X-goog-api-key") == "fixture-secret"
    requests.clear()
    pages[1] = {"error": "unavailable"}
    assert native.fetch_models("fixture-secret") is None


def test_successful_native_list_does_not_reintroduce_curated_ids(monkeypatch):
    from hermes_cli import models
    profile = get_provider_profile("gemini")
    monkeypatch.setattr(models, "_api_key_credentials", lambda name: ("fixture-key", profile.base_url))
    monkeypatch.setattr(profile, "fetch_models", lambda **kwargs: ["gemini-current-fixture"])
    assert models._profile_live_catalog("gemini") == ["gemini-current-fixture"]
    monkeypatch.setattr(profile, "fetch_models", lambda **kwargs: [])
    assert models._profile_live_catalog("gemini") == []


def test_inventory_publishes_the_same_controls_as_the_wire(catalog):
    from hermes_cli.inventory import _apply_capabilities
    from tui_gateway.contracts.config_free_tier_control import ModelCapabilities
    rows = [{"slug": "gemini", "models": list(catalog)}]
    _apply_capabilities(rows)
    caps = rows[0]["capabilities"]
    assert caps["gemini-future-levels"]["reasoning_efforts"] == ["low", "high"]
    assert caps["gemini-future-levels"]["fast"] is False
    assert caps["gemini-future-budget"]["reasoning_budget"] == {
        "min": 512, "max": 24576, "dynamic": True}
    assert caps["gemini-unknown"]["reasoning_control"] == "unknown"
    for value in caps.values():
        ModelCapabilities.model_validate(value)


@pytest.mark.parametrize("options", [None, {}, [{"type": "new"}],
    [{"type": "effort", "values": ["bogus"]}],
    [{"type": "budget_tokens", "min": True, "max": 100}],
    [{"type": "budget_tokens", "min": 101, "max": 100}]])
def test_invalid_metadata_is_unknown(options):
    assert models_dev._parse_reasoning_options(options) is None
