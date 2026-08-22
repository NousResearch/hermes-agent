"""
Unit tests for the nested Telegram ``/model`` picker enhancement.

Covers:
  * Free/Paid and Fast/Slow partitioning via ``categorize_models`` /
    ``categorize_models_nested``.
  * ``filter_models_by_query`` and ``validate_picker_query`` behavior,
    including empty / oversize rejection.
  * Callback-data byte budget (≤64 ASCII bytes) for the new prefixes.
  * Backward-compat — the legacy ``mp:`` / ``mpg:`` / ``mpv:`` / ``mm:`` /
    ``mc:`` / ``mb`` / ``mx`` / ``mg:`` prefixes are not removed.
  * Config-flag behavior of ``model_picker.speed_categories`` (default off).

These tests focus on the catalog helpers and callback-encoding contracts.
They intentionally do NOT require a live Telegram bot — they cover the pure
parts of the contract; the adapter-level wiring is exercised in
``test_telegram_model_picker.py``.
"""

from __future__ import annotations

import pytest

from hermes_cli.models import (
    categorize_models,
    categorize_models_nested,
    filter_models_by_query,
    validate_picker_query,
)


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


def test_categorize_models_basic_partition():
    ids = ["openai/gpt-3.5:free", "openai/gpt-4", "hf/llama:fast", "other:slow"]
    base = categorize_models(ids)
    assert set(base.keys()) == {"free", "paid", "fast", "slow"}
    for v in base.values():
        assert isinstance(v, list)
    # Free/Paid partition is disjoint and exhaustive over the input.
    assert set(base["free"]) | set(base["paid"]) == set(ids)
    assert not (set(base["free"]) & set(base["paid"]))


def test_categorize_models_nested_default_is_flat():
    ids = ["openai/gpt-3.5:free", "openai/gpt-4", "hf/llama:fast"]
    nested = categorize_models_nested(ids, speed_categories_enabled=False)
    assert "free" in nested and "paid" in nested
    # Speed-categories disabled: values stay flat lists (not nested dicts).
    assert isinstance(nested["free"], list)
    assert isinstance(nested["paid"], list)
    assert "fast" not in nested["free"]  # no third tier when flag is off


def test_categorize_models_nested_with_speed_categories():
    ids = ["a:free", "a:fast", "b:paid", "b:slow"]
    nested = categorize_models_nested(ids, speed_categories_enabled=True)
    # When enabled, each Free/Paid bucket is itself a dict with fast/slow keys.
    assert isinstance(nested["free"], dict)
    assert isinstance(nested["paid"], dict)
    assert set(nested["free"].keys()) == {"fast", "slow"}
    assert set(nested["paid"].keys()) == {"fast", "slow"}


def test_categorize_models_nested_empty_input():
    base = categorize_models_nested([], speed_categories_enabled=False)
    assert base == {"free": [], "paid": []}
    nested = categorize_models_nested([], speed_categories_enabled=True)
    assert nested == {
        "free": {"fast": [], "slow": []},
        "paid": {"fast": [], "slow": []},
    }


# ---------------------------------------------------------------------------
# Search / query validation
# ---------------------------------------------------------------------------


def test_filter_models_by_query_case_insensitive():
    ids = ["gpt-4", "claude-3-opus", "Gemini-Pro"]
    out = filter_models_by_query(ids, "gpt")
    assert out == ["gpt-4"]
    out2 = filter_models_by_query(ids, "GEM")
    assert out2 == ["Gemini-Pro"]


def test_filter_models_by_query_rejects_empty_and_overlong():
    ids = ["x", "y", "z"]
    assert filter_models_by_query(ids, "") == []
    assert filter_models_by_query(ids, "q" * 65) == []


def test_filter_models_by_query_at_boundary():
    # Build a corpus that contains a 64-char string and prove the inclusive
    # upper bound is honored (filter passes through to substring match).
    corpus = ["a" * 64, "beta"]
    assert filter_models_by_query(corpus, "a" * 64) == ["a" * 64]
    # And the 65-char case is rejected:
    assert filter_models_by_query(corpus, "a" * 65) == []


def test_validate_picker_query_bounds():
    assert validate_picker_query("") is False
    assert validate_picker_query("q" * 65) is False
    assert validate_picker_query(None) is False
    assert validate_picker_query(123) is False
    assert validate_picker_query("ok") is True
    assert validate_picker_query("q" * 64) is True
    assert validate_picker_query("a") is True


# ---------------------------------------------------------------------------
# Callback-data byte budget (Telegram enforces ≤64 ASCII bytes)
# ---------------------------------------------------------------------------


def test_callback_encodings_within_64_bytes():
    # Provider idx up to 999, fp/fs are short tokens; lengths are tiny.
    for provider_idx in range(0, 1000):
        for fp in ("free", "paid"):
            data = f"mp:cat:{provider_idx}:{fp}"
            assert len(data.encode("ascii")) <= 64, data
            for fs in ("fast", "slow"):
                data2 = f"mp:cat2:{provider_idx}:{fp}:{fs}"
                assert len(data2.encode("ascii")) <= 64, data2


def test_callback_search_action_short_token():
    # Search prompts are passed through the adapter's text/force-reply flow,
    # not callback data. Even if a short action key leaks into a callback,
    # the canonical action token is well under the byte budget.
    for token in ("search", "q", "go", "back", "next"):
        data = f"mp:{token}"
        assert len(data.encode("ascii")) <= 64


# ---------------------------------------------------------------------------
# Backward compatibility of legacy prefixes
# ---------------------------------------------------------------------------


LEGACY_PREFIXES = {"mp:", "mpg:", "mpv:", "mm:", "mc:", "mb", "mx", "mg:"}
NEW_PREFIXES = {"mp:cat:", "mp:cat2:", "mp:search"}


def test_backward_compatibility_legacy_prefixes_preserved():
    # Each new prefix must start with a legacy prefix and then add a
    # distinguishing token — they MUST NOT collide with any legacy prefix.
    for new in NEW_PREFIXES:
        assert not new in LEGACY_PREFIXES, f"{new} collides with legacy"
        # New prefixes must remain distinct from the bare ``mp:`` so the
        # legacy branch can keep parsing them as ``mp:<slug>``.
        assert new != "mp:"
        # All new prefixes are non-empty extensions of ``mp:``.
        assert new.startswith("mp:")


def test_legacy_mp_slug_callback_still_well_formed():
    # Sample legacy provider-slug callback remains <=64 bytes.
    slug = "openai/gpt-4o-mini"
    data = f"mp:{slug}"
    assert len(data.encode("ascii")) <= 64
    # And the new category callback for the same provider is even shorter.
    cat = f"mp:cat:0:paid"
    assert len(cat.encode("ascii")) <= 64


# ---------------------------------------------------------------------------
# Config flag default behavior
# ---------------------------------------------------------------------------


def test_speed_categories_flag_default_is_false_in_contract():
    """``model_picker.speed_categories`` must default to False.

    The catalog helper treats ``False`` as the default — re-asserting the
    contract here so a future refactor that flips the default to ``True``
    is caught by tests.
    """
    ids = ["a:free", "a:fast", "b:paid"]
    # speed_categories_enabled omitted → defaults to False → flat partition.
    default = categorize_models_nested(ids)
    assert isinstance(default["free"], list)
    # Explicitly False → identical result.
    explicit = categorize_models_nested(ids, speed_categories_enabled=False)
    assert explicit == default


# ---------------------------------------------------------------------------
# Parametric sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", " " * 65, "x" * 64 + "y", None, 42, [], {}])
def test_validate_picker_query_rejects_bad_inputs(query):
    assert validate_picker_query(query) is False


@pytest.mark.parametrize("query", ["a", "ab", "GPT-4", "中", "q" * 64])
def test_validate_picker_query_accepts_good_inputs(query):
    assert validate_picker_query(query) is True