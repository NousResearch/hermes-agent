"""Regression tests for section-3 (``providers:``) same-endpoint grouping in
``list_authenticated_providers`` and for ``format_model_for_display``.

Salvaged with PR #36998 (@antydizajn): section 3 folds ``providers:`` entries
that share (api_url, credential, api_mode, extra_headers) into one picker row,
mirroring section 4's grouping for ``custom_providers:``. These are invariant
tests — grouping identity, header-routed separation, list-of-dict model
declarations, and display-only RID stripping.
"""

import pytest

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import (
    format_model_for_display,
    strip_bedrock_profile_prefix_for_display,
    list_authenticated_providers,
)


def _providers(monkeypatch, user_providers):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *a, **k: [])
    return list_authenticated_providers(
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )


def _user_rows(rows):
    return [p for p in rows if p.get("source") == "user-config"]


def test_same_endpoint_same_credential_entries_fold_to_one_row(monkeypatch):
    """Two providers: entries differing only by model id collapse into one
    picker row carrying both models (the Palantir Foundry case)."""
    rows = _user_rows(_providers(monkeypatch, {
        "palantir-claude46": {
            "name": "Palantir Claude 4.6 Opus",
            "base_url": "https://foundry.example.com/anthropic",
            "key_env": "PALANTIR_TOKEN",
            "api_mode": "anthropic_messages",
            "model": "ri.language-model-service..language-model.anthropic-claude-4-6-opus",
        },
        "palantir-claude47": {
            "name": "Palantir Claude 4.7 Opus",
            "base_url": "https://foundry.example.com/anthropic",
            "key_env": "PALANTIR_TOKEN",
            "api_mode": "anthropic_messages",
            "model": "ri.language-model-service..language-model.anthropic-claude-4-7-opus",
        },
    }))
    assert len(rows) == 1
    row = rows[0]
    assert row["slug"] == "palantir-claude46"  # first member's slug wins
    assert row["name"] == "Palantir Claude"    # version suffix stripped
    assert len(row["models"]) == 2


def test_different_extra_headers_keep_distinct_rows(monkeypatch):
    """Header-routed tenants behind one proxy URL are distinct endpoints —
    extra_headers is part of the group identity (mirrors section 4)."""
    rows = _user_rows(_providers(monkeypatch, {
        "tenant-a": {
            "name": "Tenant A",
            "base_url": "https://proxy.example.com/v1",
            "key_env": "PROXY_TOKEN",
            "api_mode": "openai_chat",
            "extra_headers": {"X-Tenant": "a"},
            "model": "model-a",
        },
        "tenant-b": {
            "name": "Tenant B",
            "base_url": "https://proxy.example.com/v1",
            "key_env": "PROXY_TOKEN",
            "api_mode": "openai_chat",
            "extra_headers": {"X-Tenant": "b"},
            "model": "model-b",
        },
    }))
    assert len(rows) == 2


class TestFormatModelForDisplay:
    def test_palantir_rid_stripped_to_trailing_slug(self):
        rid = "ri.language-model-service..language-model.anthropic-claude-4-7-opus"
        assert format_model_for_display(rid) == "anthropic-claude-4-7-opus"


class TestStripBedrockProfilePrefixForDisplay:
    """Every profile ID below is a real one, confirmed against
    ``ListInferenceProfiles``. That matters more than it looks: a string
    transform passes just as happily on an ID no region serves, so an invented
    one would quietly stop being evidence that the helper handles the shapes
    Bedrock actually returns."""

    @pytest.mark.parametrize("profile,expected", [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0",
         "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
         "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("global.anthropic.claude-sonnet-4-5-20250929-v1:0",
         "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("apac.anthropic.claude-sonnet-4-20250514-v1:0",
         "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("us.meta.llama4-scout-17b-instruct-v1:0",
         "meta.llama4-scout-17b-instruct-v1:0"),
        ("us.deepseek.r1-v1:0", "deepseek.r1-v1:0"),
    ])
    def test_geo_prefix_stripped(self, profile, expected):
        assert strip_bedrock_profile_prefix_for_display(profile) == expected

    @pytest.mark.parametrize("model_id", [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "meta.llama4-scout-17b-instruct-v1:0",
        "mistral.pixtral-large-2502-v1:0",
        "deepseek.v3.2",
        "moonshotai.kimi-k2.5",
        "amazon.nova-pro-v1:0",
        "claude-sonnet-4-20250514",
        "meta-llama/Llama-3.3-70B-Instruct",
        "gpt-5-4",
    ])
    def test_non_profile_ids_are_untouched(self, model_id):
        assert strip_bedrock_profile_prefix_for_display(model_id) == model_id

    def test_vendor_is_never_eaten_when_no_dotted_tail_remains(self):
        """The guard that matters: a name whose first dotted token collides with
        a geography token must not lose it, because a real profile ID always has
        a ``vendor.model`` tail left over and this one does not."""
        assert strip_bedrock_profile_prefix_for_display("me.some-model") == "me.some-model"
        assert strip_bedrock_profile_prefix_for_display("ca.thing-v1:0") == "ca.thing-v1:0"

    def test_empty_and_prefix_only_are_safe(self):
        assert strip_bedrock_profile_prefix_for_display("") == ""
        assert strip_bedrock_profile_prefix_for_display("us.") == "us."

    def test_only_the_leading_prefix_goes(self):
        """Strip once, not repeatedly -- the second token is the vendor."""
        assert strip_bedrock_profile_prefix_for_display(
            "us.us.anthropic.claude-sonnet-4-5") == "us.anthropic.claude-sonnet-4-5"

    def test_the_switch_note_path_is_deliberately_untouched(self):
        """``format_model_for_display`` also feeds the model-switch note, where
        collapsing ``us.X`` to ``X`` would render as "switched from X to X"."""
        profile = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        assert format_model_for_display(profile) == profile
