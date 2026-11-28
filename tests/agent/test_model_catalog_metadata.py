"""The remote model catalog fills in metadata and context windows for models the install does not know.

A model published after a release has no row in the built-in tables, so it resolved to the generic
256K window and to unknown capabilities. The cached manifest now supplies both. A row the install
ships always keeps its own value: the manifest fills gaps, it does not correct.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent import model_metadata
from agent.bedrock_adapter import BEDROCK_CONTEXT_LENGTHS, bedrock_static_context_length
from agent.context_pin import advertised_context_length
from agent.usage_pricing import get_pricing_entry
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT, get_model_context_length
from agent.models_dev import get_model_capabilities
from hermes_cli import model_catalog

NEW_MODEL = "novelfamily-test-9"
SHIPPED_MODEL = "gpt-6.2-sol"
SHIPPED_BEDROCK_KEY = next(iter(BEDROCK_CONTEXT_LENGTHS))


def _manifest() -> dict:
    return {
        "version": 1,
        "providers": {},
        "context_lengths": {
            "default": {NEW_MODEL: 777_000, SHIPPED_MODEL: 5},
            "bedrock": {"novelfamily-test-9": 333_000, SHIPPED_BEDROCK_KEY: 5},
        },
        "model_metadata": {
            "deepseek": {
                "deepseek-test-9": {
                    "limit": {"context": 500_000, "output": 64_000},
                    "modalities": {"input": ["text", "image"], "output": ["text"]},
                    "tool_call": True,
                    "reasoning": True,
                },
                "deepseek-flash": {"limit": {"context": 5}, "tool_call": False},
            },
        },
    }


@pytest.fixture
def cached_manifest():
    """Write the manifest into this test's ``HERMES_HOME`` cache, as an earlier fetch would have."""
    model_catalog.reset_cache()
    path = Path(os.environ["HERMES_HOME"]) / "cache" / "model_catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    yield
    model_catalog.reset_cache()


def test_a_model_without_a_built_in_row_gets_its_window_from_the_manifest(cached_manifest):
    assert get_model_context_length(NEW_MODEL) == 777_000


def test_a_model_no_catalog_knows_gets_its_capabilities_from_the_manifest(cached_manifest):
    caps = get_model_capabilities("deepseek", "deepseek-test-9")

    assert caps.context_window == 500_000
    assert caps.max_output_tokens == 64_000
    assert caps.supports_vision is True
    assert caps.supports_reasoning is True


def _recache(data: dict) -> None:
    Path(os.environ["HERMES_HOME"], "cache", "model_catalog.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_manifest_window_reaches_every_fallback_that_read_the_built_in_table(cached_manifest, monkeypatch):
    data = _manifest()
    data["context_lengths"]["codex_oauth"] = {NEW_MODEL: 444_000}
    del data["context_lengths"]["bedrock"]["novelfamily-test-9"]
    _recache(data)
    monkeypatch.setattr(model_metadata, "_resolve_endpoint_context_length", lambda *a, **kw: None)
    monkeypatch.setattr(model_metadata, "_query_ollama_api_show", lambda *a, **kw: None)

    assert get_model_context_length(NEW_MODEL, "https://proxy.example", provider="custom") == 777_000
    assert advertised_context_length(NEW_MODEL) == 777_000
    assert bedrock_static_context_length(f"us.openai.{NEW_MODEL}-v1:0") == 777_000
    assert model_metadata._resolve_codex_oauth_context_length_with_source(f"openai/{NEW_MODEL}") == (444_000, "fallback")


def test_a_published_mixed_case_key_matches_the_lowercased_model_id(cached_manifest):
    data = _manifest()
    data["context_lengths"]["default"] = {"NovelVendor/NovelFamily-Mixed-9": 65_536}
    _recache(data)

    assert advertised_context_length("novelvendor/novelfamily-mixed-9") == 65_536


def test_a_case_variant_of_a_shipped_key_does_not_replace_it(cached_manifest):
    data = _manifest()
    data["context_lengths"]["default"] = {SHIPPED_MODEL.upper(): 5}
    _recache(data)

    assert advertised_context_length(SHIPPED_MODEL) == DEFAULT_CONTEXT_LENGTHS[SHIPPED_MODEL]


@pytest.mark.parametrize("row", [{"": 32_000}, {"  ": 32_000}, {NEW_MODEL: True}, {NEW_MODEL: 0}, {NEW_MODEL: "777000"}])
def test_a_malformed_context_row_is_ignored(cached_manifest, row):
    data = _manifest()
    data["context_lengths"]["default"] = row
    _recache(data)

    assert get_model_context_length(NEW_MODEL) == DEFAULT_FALLBACK_CONTEXT


@pytest.mark.parametrize("row", ["invalid", {}, {"limit": []}, {"limit": {"context": "500000"}},
                                 {"limit": {"context": 500_000, "output": True}}])
def test_a_malformed_metadata_entry_leaves_the_model_unknown(cached_manifest, row):
    data = _manifest()
    data["model_metadata"]["deepseek"]["deepseek-test-9"] = row
    _recache(data)

    assert get_model_capabilities("deepseek", "deepseek-test-9") is None


@pytest.mark.parametrize("cost", [{"input": 1, "output": 2}, {"input": "unknown"}])
def test_a_metadata_entry_never_supplies_a_price(cached_manifest, cost):
    data = _manifest()
    data["model_metadata"]["deepseek"]["deepseek-test-9"]["cost"] = cost
    _recache(data)

    assert get_pricing_entry("deepseek-test-9", "deepseek") is None


def test_the_dashboards_model_info_reads_the_requested_profiles_catalog(cached_manifest, monkeypatch):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers.models import get_model_info

    monkeypatch.setattr(model_metadata, "_resolve_endpoint_context_length", lambda *a, **kw: None)
    monkeypatch.setattr(model_metadata, "_query_ollama_api_show", lambda *a, **kw: None)
    b_home = profiles_mod.get_profile_dir("b")
    (b_home / "cache").mkdir(parents=True, exist_ok=True)
    (b_home / "config.yaml").write_text(
        f"model:\n  default: {NEW_MODEL}\n  provider: custom\n  base_url: https://proxy.example/v1\n", encoding="utf-8")
    data = _manifest()
    data["context_lengths"]["default"][NEW_MODEL] = 333_000  # the launch profile's manifest says 777,000
    (b_home / "cache" / "model_catalog.json").write_text(json.dumps(data), encoding="utf-8")

    assert get_model_info(profile="b")["auto_context_length"] == 333_000
