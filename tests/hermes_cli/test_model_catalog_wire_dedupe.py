"""Wire-spelling dedupe in provider catalog merges (#anthropic claude-fable-5.1 duplicate)."""

from hermes_cli.models import _merge_unique, _model_dedup_key


def test_dedup_key_folds_a_providers_wire_spelling():
    """Anthropic rewrites dots to hyphens on the wire, so the public slug and the wire id are ONE
    model. Without the fold the picker lists `claude-fable-5.1` and `claude-fable-5-1` separately."""
    slug_key = _model_dedup_key("claude-fable-5.1", "anthropic")
    wire_key = _model_dedup_key("claude-fable-5-1", "anthropic")

    assert slug_key == wire_key

    # Distinct models must NOT collapse into each other.
    assert _model_dedup_key("claude-fable-5", "anthropic") != slug_key
    assert _model_dedup_key("claude-opus-5", "anthropic") != slug_key

    # Without a provider the key cannot know about the rewrite — the two stay distinct, which is
    # why the merge call sites must pass the provider.
    assert _model_dedup_key("claude-fable-5.1") != _model_dedup_key("claude-fable-5-1")


def test_merge_keeps_one_entry_per_model_and_prefers_the_public_slug():
    """A curated/live merge of the same model under both spellings yields one row, spelled as the
    curated public slug — the picker renders that, and saved selections are keyed on it."""
    curated = ["claude-fable-5.1", "claude-opus-5"]
    live = ["claude-fable-5-1", "claude-opus-5", "claude-haiku-4-5"]

    merged = _merge_unique(curated, live, key=lambda m: _model_dedup_key(m, "anthropic"))

    assert merged.count("claude-fable-5.1") == 1
    assert "claude-fable-5-1" not in merged
    # Live-only models still arrive — dedupe must not become a filter.
    assert "claude-haiku-4-5" in merged
