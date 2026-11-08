"""Tests for the /model picker fuzzy filter (C-01).

The filter narrows a provider's concrete model list as the user types, but
selection must still resolve to exactly ONE real model — never an ambiguous
or fuzzy resolution (the "claude → claude-sonnet-3" footgun). These pin the
index-preserving contract of ``_filter_model_picker_entries``.
"""

from cli import HermesCLI


MODELS = [
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.5",
    "x-ai/grok-4.6",
    "deepseek/deepseek-v4-flash",
]


def test_empty_query_returns_all_with_original_indices():
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "")
    assert pairs == list(enumerate(MODELS))


def test_filter_narrows_and_preserves_original_index():
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "grok")
    # Only the grok row matches, and it carries its ORIGINAL index (4) so the
    # selection handler resolves the exact concrete model.
    assert pairs == [(4, "x-ai/grok-4.6")]
    idx, label = pairs[0]
    assert MODELS[idx] == label  # index maps back to the real entry


def test_subsequence_match_case_insensitive():
    # "cs46" is a subsequence of "anthropic/claude-sonnet-4.6"
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "CS46")
    assert ("anthropic/claude-sonnet-4.6") in [e for _i, e in pairs]


def test_no_match_returns_empty():
    assert HermesCLI._filter_model_picker_entries(MODELS, "zzzznope") == []


def test_filter_does_not_reorder_or_pick_a_default():
    # Typing "claude" narrows to the three claude rows in ORIGINAL order — it
    # never silently resolves to one (the anti-ambiguity guarantee). The user
    # still explicitly selects among the concrete matches.
    pairs = HermesCLI._filter_model_picker_entries(MODELS, "claude")
    labels = [e for _i, e in pairs]
    assert labels == [
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
    ]
    # indices are the originals, in order
    assert [i for i, _e in pairs] == [0, 1, 2]


def test_whitespace_query_is_treated_as_empty():
    assert HermesCLI._filter_model_picker_entries(MODELS, "   ") == list(enumerate(MODELS))
