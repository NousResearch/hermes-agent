"""Regression tests for slash-qualified model ids in context-length lookup (#47782).

``get_model_context_length`` returned different windows for the same model
depending on spelling: ``"qwen3.7-plus"`` with ``provider="opencode-go"``
resolved via the provider-aware models.dev lookup (1,000,000), while
``"opencode-go/qwen3.7-plus"`` kept the prefixed id, whose ``qwen``
substring matched the generic hardcoded ``DEFAULT_CONTEXT_LENGTHS`` entry
(131,072) before the per-model lookup scored. The fix parses the
slash prefix once — but only when the caller supplied no explicit provider —
strips it from the model id, and propagates it into the provider-aware
lookup. OpenRouter slug handling and explicitly-supplied providers are
untouched.
"""

from unittest.mock import patch

import agent.model_metadata as mm


def _with_models_dev(monkeypatch, entries):
    """Point lookup_models_dev_context at a fake catalog keyed (provider, model)."""
    calls = []

    def fake_lookup(provider, model):
        calls.append((provider, model))
        return entries.get((provider, model))

    monkeypatch.setattr(
        "agent.models_dev.lookup_models_dev_context", fake_lookup, raising=True
    )
    return calls


class TestSlashQualifiedResolution:
    def test_prefixed_and_bare_with_provider_agree(self, monkeypatch):
        calls = _with_models_dev(
            monkeypatch, {("opencode-go", "qwen3.7-plus"): 1_000_000}
        )
        monkeypatch.setattr(
            mm, "get_model_context_length_async", mm.get_model_context_length
        )

        via_bare = mm.get_model_context_length(
            "qwen3.7-plus", provider="opencode-go"
        )
        via_prefixed = mm.get_model_context_length("opencode-go/qwen3.7-plus")

        assert via_prefixed == via_bare == 1_000_000
        assert ("opencode-go", "qwen3.7-plus") in calls, (
            "the parsed prefix must reach the provider-aware lookup (#47782)"
        )

    def test_parsed_provider_beats_generic_substring_default(self, monkeypatch):
        # The generic {"qwen": 131072} entry must NOT win when the slash
        # prefix resolves to a real provider with a per-model entry.
        _with_models_dev(
            monkeypatch, {("opencode-go", "qwen3.5-plus"): 262_144}
        )
        assert (
            mm.get_model_context_length("opencode-go/qwen3.5-plus") == 262_144
        )

    def test_explicit_provider_leaves_server_side_spelling_whole(
        self, monkeypatch
    ):
        # With an explicit provider the id is a server-side spelling and is
        # NOT rewritten: local/custom endpoints know org-qualified names
        # like "NousResearch/Hermes-…" verbatim, and NVIDIA NIM uses
        # "deepseek-ai/…" vendor ids. The strip only happens when the
        # caller supplied no provider at all.
        calls = _with_models_dev(
            monkeypatch, {("custom-provider", "NousResearch/Hermes-3-70B"): 65_536}
        )
        assert (
            mm.get_model_context_length(
                "NousResearch/Hermes-3-70B", provider="custom-provider"
            )
            == 65_536
        )
        assert ("custom-provider", "NousResearch/Hermes-3-70B") in calls

    def test_unknown_prefix_left_untouched(self, monkeypatch):
        # A slash prefix that is NOT a registered provider is left as part
        # of the model id (existing consumers of qualified forms keep
        # working; the fallback path is unchanged).
        calls = _with_models_dev(monkeypatch, {})
        with patch.dict(mm.DEFAULT_CONTEXT_LENGTHS, {"notaprovider/odd-model": 777}):
            assert mm.get_model_context_length("notaprovider/odd-model") == 777

    def test_openrouter_slug_not_stripped_with_explicit_provider(
        self, monkeypatch
    ):
        # OR slugs ("anthropic/claude-fable-5") are model ids, not
        # provider prefixes: with an explicit OpenRouter provider the id
        # must reach the OR catalog verbatim. Pin via a fake OR metadata
        # entry keyed by the full slug — if the prefix were stripped the
        # lookup would miss and fall through to a different value.
        monkeypatch.setattr(
            mm,
            "fetch_model_metadata",
            lambda: {"anthropic/claude-fable-5": {"context_length": 999_999}},
        )
        assert (
            mm.get_model_context_length(
                "anthropic/claude-fable-5", provider="openrouter"
            )
            == 999_999
        )

    def test_url_inference_still_beats_parsed_prefix(self, monkeypatch):
        calls = _with_models_dev(
            monkeypatch, {("from-url", "m1"): 123_456}
        )
        # A provider inferred from the endpoint URL is more authoritative
        # than one parsed off the model id (opencode-go is registered, so
        # the parse succeeds and would otherwise claim the lookup).
        with patch.object(
            mm, "_infer_provider_from_url", return_value="from-url"
        ):
            assert (
                mm.get_model_context_length(
                    "opencode-go/m1", base_url="https://api.example/v1"
                )
                == 123_456
            )
        assert ("from-url", "m1") in calls
        assert ("opencode-go", "m1") not in calls

    def test_openrouter_endpoint_keeps_slug_whole(self, monkeypatch):
        # Same OR-slug protection as the explicit-provider test above, but
        # for the caller that supplied NO provider and only the OpenRouter
        # base URL: the endpoint makes the whole id an OR slug, and the
        # first segment ("anthropic") names a bundled provider, so without
        # the endpoint guard the strip would rewrite the id to
        # ("anthropic", "claude-fable-5") and miss the OR catalog entry.
        monkeypatch.setattr(
            mm,
            "fetch_model_metadata",
            lambda: {"anthropic/claude-fable-5": {"context_length": 999_999}},
        )
        assert (
            mm.get_model_context_length(
                "anthropic/claude-fable-5",
                base_url="https://openrouter.ai/api/v1",
            )
            == 999_999
        )

    def test_user_plugin_prefix_not_stripped(self, monkeypatch):
        # The strip must key off BUNDLED provider names only: the registry
        # is extended by user plugins, and a user-only registration must
        # not change how a slash-qualified id resolves. "anthropic" is
        # bundled, "acme-inference" is not.
        calls = _with_models_dev(monkeypatch, {})
        monkeypatch.setattr(
            "providers.is_bundled_provider", lambda name: name == "anthropic"
        )
        with patch.dict(mm.DEFAULT_CONTEXT_LENGTHS, {"acme-inference/m9": 555}):
            assert mm.get_model_context_length("acme-inference/m9") == 555
        assert not calls
