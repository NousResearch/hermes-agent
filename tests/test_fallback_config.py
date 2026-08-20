"""Tests for :mod:`hermes_cli.fallback_config`.

Covers both on-disk shapes of the fallback chain:

* dict entries — the canonical form written by ``hermes fallback`` / the CLI.
* ``"provider:model"`` strings — the shape produced by the desktop
  *Model → Fallback Models* settings field (a generic comma-separated ``list``
  editor whose help text reads "Backup provider:model entries ...").
"""
from __future__ import annotations

from hermes_cli.fallback_config import _iter_fallback_entries, get_fallback_chain


class TestIterFallbackEntriesDicts:
    def test_single_dict(self):
        assert _iter_fallback_entries({"provider": "custom", "model": "m1"}) == [
            {"provider": "custom", "model": "m1"}
        ]

    def test_list_of_dicts_preserves_order(self):
        raw = [
            {"provider": "a", "model": "m1"},
            {"provider": "b", "model": "m2"},
        ]
        out = _iter_fallback_entries(raw)
        assert [(e["provider"], e["model"]) for e in out] == [("a", "m1"), ("b", "m2")]

    def test_dict_missing_provider_or_model_is_skipped(self):
        assert _iter_fallback_entries({"model": "m1"}) == []
        assert _iter_fallback_entries({"provider": "a"}) == []

    def test_base_url_is_normalized(self):
        out = _iter_fallback_entries(
            {"provider": "a", "model": "m", "base_url": "https://x.test/v1/  "}
        )
        assert out[0]["base_url"] == "https://x.test/v1"

    def test_non_iterable_returns_empty(self):
        assert _iter_fallback_entries(None) == []
        assert _iter_fallback_entries(123) == []


class TestIterFallbackEntriesStrings:
    """The desktop settings field persists ``provider:model`` strings.

    String coercion is opt-in via ``allow_strings`` (only ``fallback_providers``
    enables it), so these direct-helper cases pass the flag explicitly.
    """

    def test_simple_provider_model_string(self):
        assert _iter_fallback_entries(
            ["openrouter:anthropic/claude-3.5"], allow_strings=True
        ) == [{"provider": "openrouter", "model": "anthropic/claude-3.5"}]

    def test_qualified_custom_provider_keeps_full_provider(self):
        # custom:<name>:<model> is special-cased so the qualified custom
        # endpoint id keeps its "custom:<name>" prefix.
        assert _iter_fallback_entries(
            ["custom:my-endpoint:claude-sonnet-4.6"], allow_strings=True
        ) == [{"provider": "custom:my-endpoint", "model": "claude-sonnet-4.6"}]

    def test_model_id_with_colon_suffix_keeps_full_model(self):
        # First-colon split: the model half may itself contain colons
        # (e.g. an Ollama Cloud size suffix), which must survive intact rather
        # than being swallowed by a last-colon split.
        assert _iter_fallback_entries(
            ["ollama-cloud:nemotron-3-nano:30b"], allow_strings=True
        ) == [{"provider": "ollama-cloud", "model": "nemotron-3-nano:30b"}]

    def test_openrouter_free_suffix_keeps_full_model(self):
        assert _iter_fallback_entries(
            ["openrouter:qwen/qwen3.6-plus:free"], allow_strings=True
        ) == [{"provider": "openrouter", "model": "qwen/qwen3.6-plus:free"}]

    def test_whitespace_is_trimmed(self):
        assert _iter_fallback_entries([" openai : gpt-5.2 "], allow_strings=True) == [
            {"provider": "openai", "model": "gpt-5.2"}
        ]

    def test_bare_model_without_provider_is_skipped(self):
        # No colon -> the provider can't be inferred -> dropped, matching the
        # "provider:model" contract advertised by the field's help text.
        assert _iter_fallback_entries(["gpt-5.2"], allow_strings=True) == []

    def test_top_level_string(self):
        assert _iter_fallback_entries("custom:m1", allow_strings=True) == [
            {"provider": "custom", "model": "m1"}
        ]

    def test_strings_dropped_when_not_allowed(self):
        # Default (allow_strings=False) is used for the dict-only fallback_model
        # key -- strings must be ignored there.
        assert _iter_fallback_entries(["a:m1"]) == []
        assert _iter_fallback_entries("a:m1") == []

    def test_mixed_strings_and_dicts(self):
        raw = ["a:m1", {"provider": "b", "model": "m2"}]
        out = _iter_fallback_entries(raw, allow_strings=True)
        assert [(e["provider"], e["model"]) for e in out] == [("a", "m1"), ("b", "m2")]


class TestGetFallbackChain:
    def test_string_entries_now_participate(self):
        # Regression: string entries from the desktop field used to be silently
        # dropped, leaving the effective chain empty.
        cfg = {
            "fallback_providers": [
                "custom:llmhub:claude-sonnet-4.6",
                "openai:gpt-5.2",
            ]
        }
        chain = get_fallback_chain(cfg)
        assert [(e["provider"], e["model"]) for e in chain] == [
            ("custom:llmhub", "claude-sonnet-4.6"),
            ("openai", "gpt-5.2"),
        ]

    def test_colon_suffix_model_string_participates(self):
        # First-colon split end to end: an Ollama Cloud size suffix survives.
        cfg = {"fallback_providers": ["ollama-cloud:nemotron-3-nano:30b"]}
        chain = get_fallback_chain(cfg)
        assert [(e["provider"], e["model"]) for e in chain] == [
            ("ollama-cloud", "nemotron-3-nano:30b"),
        ]

    def test_string_fallback_model_is_not_coerced(self):
        # fallback_model is dict-only (documented + validated). A bare string
        # there must NOT be coerced -- string coercion is scoped to the
        # fallback_providers desktop list field.
        cfg = {"fallback_model": "openai:gpt-5.2"}
        assert get_fallback_chain(cfg) == []

    def test_providers_stay_before_legacy_fallback_model(self):
        cfg = {
            "fallback_providers": [{"provider": "a", "model": "m1"}],
            "fallback_model": {"provider": "b", "model": "m2"},
        }
        chain = get_fallback_chain(cfg)
        assert [(e["provider"], e["model"]) for e in chain] == [("a", "m1"), ("b", "m2")]

    def test_duplicate_route_is_deduped_across_keys(self):
        cfg = {
            "fallback_providers": [{"provider": "a", "model": "m1"}],
            "fallback_model": {"provider": "a", "model": "m1"},
        }
        assert len(get_fallback_chain(cfg)) == 1

    def test_empty_config(self):
        assert get_fallback_chain({}) == []
        assert get_fallback_chain(None) == []
