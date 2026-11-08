"""ACP rows show a provider's own model name; ids and routing are untouched.

ACP documents ``ModelInfo.name`` as the "human-readable name of the model". Every custom endpoint
rendered its routing id instead, because discovery kept only ``data[].id``. These pin the ACP
half for BOTH row kinds — named endpoints (providers:/custom_providers:) and canonical inventory
rows — plus the invariant that a failed name lookup never changes which rows exist.
"""

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
         patch("hermes_cli.models.fetch_api_model_names", **names_result):
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
        catalogs = _catalogs({"side_effect": RuntimeError("endpoint down")})
        assert [mid for mid, _ in catalogs[0][2]] == CATALOG

    def test_empty_lookup_keeps_every_row(self):
        catalogs = _catalogs({"return_value": {}})
        assert [mid for mid, _ in catalogs[0][2]] == CATALOG

    def test_declared_only_endpoint_keeps_its_rows(self):
        entry = _entry()
        entry.pop("api_key")
        entry["model"] = CATALOG[0]
        with patch("hermes_cli.config.load_config",
                   return_value={"custom_providers": [entry]}), \
             patch("hermes_cli.config.get_compatible_custom_providers",
                   return_value=[entry]), \
             patch("hermes_cli.models.fetch_api_model_names",
                   side_effect=RuntimeError("no network")):
            catalogs = model_catalog._named_custom_provider_catalogs()
        assert catalogs[0][2] == [(CATALOG[0], "")]


class TestCanonicalProviderRows:
    """Built-in providers get real names too, via the models.dev layer."""

    def test_inventory_row_uses_the_name_from_models_dev(self):
        cat = model_catalog._ModelCatalog(
            normalize_provider=lambda p: p, current_model="", current_choice_provider="",
            current_base_url="", display_names={"gpt-5.6-sol": "GPT-5.6 Sol"})
        cat.add_inventory_rows(
            [{"slug": "openai", "name": "OpenAI", "models": ["gpt-5.6-sol"]}],
            lambda pid: pid.upper())
        assert cat.models[0].name == "GPT-5.6 Sol"

    def test_inventory_row_keeps_provider_prefixed_id_when_unnamed(self):
        cat = model_catalog._ModelCatalog(
            normalize_provider=lambda p: p, current_model="", current_choice_provider="",
            current_base_url="", display_names={})
        cat.add_inventory_rows(
            [{"slug": "openai", "name": "OpenAI", "models": ["mystery/model"]}],
            lambda pid: pid.upper())
        assert cat.models[0].name == "OpenAI · mystery/model"

    def test_dict_rows_carry_their_own_name(self):
        cat = model_catalog._ModelCatalog(
            normalize_provider=lambda p: p, current_model="", current_choice_provider="",
            current_base_url="", display_names={})
        cat.add_inventory_rows(
            [{"slug": "p", "name": "P", "models": [{"id": "x/y", "name": "Row Name"}]}],
            lambda pid: pid)
        assert cat.models[0].name == "Row Name"


class TestRoutingIdentityUnchanged:

    def test_ids_ignore_the_display_name(self):
        with_names = _rows(_catalogs({"return_value": {
            "antigravity/gemini-3.8-flash-tiered": "Gemini 3.8 Flash Tiered"}}))
        without = _rows(_catalogs({"return_value": {}}))
        assert [r.model_id for r in with_names] == [r.model_id for r in without]

    def test_each_row_keeps_a_provider_prefixed_id(self):
        rows = _rows(_catalogs({"return_value": {}}))
        for row in rows:
            assert row.model_id.startswith("custom:gateway:")
            assert row.model_id.endswith(tuple(CATALOG))
