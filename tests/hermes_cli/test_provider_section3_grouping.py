"""Regression tests for section-3 (``providers:``) same-endpoint grouping in
``list_authenticated_providers`` and for ``format_model_for_display``.

Salvaged with PR #36998 (@antydizajn): section 3 folds ``providers:`` entries
that share (api_url, credential, api_mode, extra_headers) into one picker row,
mirroring section 4's grouping for ``custom_providers:``. These are invariant
tests — grouping identity, header-routed separation, list-of-dict model
declarations, and display-only RID stripping.
"""

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import (
    format_model_for_display,
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


def test_different_api_mode_keeps_distinct_rows(monkeypatch):
    """Same host + credential but a different wire protocol must not fold."""
    rows = _user_rows(_providers(monkeypatch, {
        "proxy-claude": {
            "name": "Proxy Claude",
            "base_url": "https://proxy.example.com/v1",
            "key_env": "PROXY_TOKEN",
            "api_mode": "anthropic_messages",
            "model": "claude-opus-4.6",
        },
        "proxy-gpt": {
            "name": "Proxy GPT",
            "base_url": "https://proxy.example.com/v1",
            "key_env": "PROXY_TOKEN",
            "api_mode": "openai_chat",
            "model": "gpt-5.4",
        },
    }))
    assert len(rows) == 2


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


def test_list_of_dict_model_declarations_are_honored(monkeypatch):
    """``models: [{"id": ...}]`` rows go through _declared_model_ids — the
    grouped path must not regress that contract."""
    rows = _user_rows(_providers(monkeypatch, {
        "dictrows": {
            "name": "Dict Rows",
            "base_url": "https://dictrows.example.com/v1",
            "key_env": "DICTROWS_TOKEN",
            "models": [{"id": "model-x"}, {"id": "model-y"}],
        },
    }))
    assert len(rows) == 1
    assert rows[0]["models"] == ["model-x", "model-y"]


def test_single_word_group_name_not_over_trimmed(monkeypatch):
    """Version-token stripping only applies when the prefix keeps >= 2 words."""
    rows = _user_rows(_providers(monkeypatch, {
        "gpt54-a": {
            "name": "GPT 5.4",
            "base_url": "https://single.example.com/v1",
            "key_env": "SINGLE_TOKEN",
            "model": "gpt-5.4",
        },
    }))
    assert rows[0]["name"] == "GPT 5.4"


class TestFormatModelForDisplay:
    def test_palantir_rid_stripped_to_trailing_slug(self):
        rid = "ri.language-model-service..language-model.anthropic-claude-4-7-opus"
        assert format_model_for_display(rid) == "anthropic-claude-4-7-opus"

    def test_plain_names_pass_through(self):
        for name in (
            "claude-opus-4.6",
            "gpt-5.4",
            "meta-llama/Llama-3.3-70B-Instruct",
            "some-model.gguf",
            "",
        ):
            assert format_model_for_display(name) == name

    def test_prefix_only_edge_preserved(self):
        """A bare prefix with no trailing slug must not become empty."""
        assert format_model_for_display("ri.language-model-service..language-model.") != ""
