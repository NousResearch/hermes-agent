"""ACP labels may change; routing IDs, descriptions and row membership must not."""

from unittest.mock import patch

import pytest

from acp_adapter import model_catalog


ENDPOINT = "https://gw.example.com/v1"
CATALOG = ["antigravity/gemini-3.8-flash-tiered", "openrouter/openai/gpt-5.6-sol"]


def _entry(**overrides):
    entry = {
        "name": "Gateway",
        "base_url": ENDPOINT,
        "provider_key": "gateway",
        "api_key": "sk-test",
    }
    entry.update(overrides)
    return entry


def _catalogs(names_result):
    entry = _entry()
    with patch("hermes_cli.config.load_config",
               return_value={"custom_providers": [entry]}), \
         patch("hermes_cli.config.get_compatible_custom_providers",
               return_value=[entry]), \
         patch("hermes_cli.model_switch_providers._fetch_picker_live_models",
               return_value=list(CATALOG)), \
         patch("hermes_cli.models_catalog.fetch_api_model_names", **names_result):
        return model_catalog._named_custom_provider_catalogs()


def _rows(catalogs, display_names=None):
    cat = model_catalog._ModelCatalog(
        normalize_provider=lambda p: p, current_model="", current_choice_provider="",
        current_base_url="", display_names=display_names or {})
    cat.add_named_catalogs(catalogs, current_choice_provider="")
    return cat.models


def _by_id(rows):
    return {r.model_id.split(":", 2)[-1]: r for r in rows}


class TestNamedEndpointRows:

    def test_published_name_becomes_the_label(self):
        rows = _by_id(_rows(_catalogs({"return_value": {
            "antigravity/gemini-3.8-flash-tiered": "Gemini 3.8 Flash Tiered"}})))
        assert rows["antigravity/gemini-3.8-flash-tiered"].name == "Gemini 3.8 Flash Tiered"

    def test_label_is_not_rewritten(self):
        raw = "cu/Gemini 3.7 Flash High"
        rows = _by_id(_rows(_catalogs({"return_value": {
            "antigravity/gemini-3.8-flash-tiered": raw}})))
        assert rows["antigravity/gemini-3.8-flash-tiered"].name == raw

    def test_unnamed_model_falls_back_to_its_id(self):
        rows = _by_id(_rows(_catalogs({"return_value": {
            "antigravity/gemini-3.8-flash-tiered": "Named"}})))
        assert rows["openrouter/openai/gpt-5.6-sol"].name == "openrouter/openai/gpt-5.6-sol"

    def test_models_dev_fills_in_when_the_endpoint_publishes_no_name(self):
        rows = _by_id(_rows(_catalogs({"return_value": {}}), display_names={
            "openrouter/openai/gpt-5.6-sol": "GPT-5.6 Sol"}))
        assert rows["openrouter/openai/gpt-5.6-sol"].name == "GPT-5.6 Sol"

    def test_endpoint_name_wins_over_models_dev(self):
        rows = _by_id(_rows(_catalogs({"return_value": {
            "openrouter/openai/gpt-5.6-sol": "Gateway's Sol"}}),
            display_names={"openrouter/openai/gpt-5.6-sol": "models.dev Sol"}))
        assert rows["openrouter/openai/gpt-5.6-sol"].name == "Gateway's Sol"


class TestFailedLookupKeepsRows:

    def test_failed_lookup_keeps_every_row(self):
        from hermes_cli.models_catalog import fetch_api_model_names

        with patch("hermes_cli.models.probe_api_models", side_effect=RuntimeError("endpoint down")):
            catalogs = _catalogs({"wraps": fetch_api_model_names})
        assert [mid for mid, _ in catalogs[0][2]] == CATALOG

    def test_declared_only_endpoint_keeps_its_rows(self):
        entry = _entry()
        entry.pop("api_key")
        entry["model"] = CATALOG[0]
        with patch("hermes_cli.config.load_config",
                   return_value={"custom_providers": [entry]}), \
             patch("hermes_cli.config.get_compatible_custom_providers",
                   return_value=[entry]), \
             patch("hermes_cli.models_catalog.fetch_api_model_names",
                   side_effect=RuntimeError("no network")) as fetch_names:
            catalogs = model_catalog._named_custom_provider_catalogs()
        assert catalogs[0][2] == [(CATALOG[0], "")]
        fetch_names.assert_not_called()

    def test_disabled_discovery_does_not_fetch_labels(self):
        entry = _entry(discover_models=False, model=CATALOG[0])
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch("hermes_cli.config.get_compatible_custom_providers", return_value=[entry]), \
             patch("hermes_cli.models_catalog.fetch_api_model_names") as fetch_names:
            catalogs = model_catalog._named_custom_provider_catalogs()
        assert catalogs[0][2] == [(CATALOG[0], "")]
        fetch_names.assert_not_called()


class TestCanonicalProviderRows:
    """Built-in providers get real names too, via the models.dev layer."""

    @pytest.mark.parametrize("model, names, expected", [
        ("x/y", {"x/y": "Friendly Model"}, "Friendly Model"),
        ("x/y", {}, "Provider · x/y"),
        ({"id": "x/y", "name": "Row Name"}, {"x/y": "Cached Name"}, "Row Name"),
        ({"id": "x/y", "name": "x/y"}, {"x/y": "Friendly Model"}, "Friendly Model"),
    ])
    def test_inventory_label_priority(self, model, names, expected):
        cat = model_catalog._ModelCatalog(
            normalize_provider=lambda p: p, current_model="", current_choice_provider="",
            current_base_url="", display_names=names)
        cat.add_inventory_rows(
            [{"slug": "p", "name": "Provider", "models": [model]}], lambda pid: pid)
        assert cat.models[0].name == expected


class TestRoutingIdentityUnchanged:

    def test_ids_ignore_the_display_name(self):
        with_names = _rows(_catalogs({"return_value": {
            "antigravity/gemini-3.8-flash-tiered": "Gemini 3.8 Flash Tiered"}}))
        without = _rows(_catalogs({"return_value": {}}))
        assert [r.model_id for r in with_names] == [r.model_id for r in without]
        assert [r.description for r in with_names] == [r.description for r in without]

    def test_each_row_keeps_a_provider_prefixed_id(self):
        rows = _rows(_catalogs({"return_value": {}}))
        for row in rows:
            assert row.model_id.startswith("custom:gateway:")
            assert row.model_id.endswith(tuple(CATALOG))
