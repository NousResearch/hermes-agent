"""Tests for hindsight recall dedup — collapse restated facts, don't merge contradictions."""

import json
from types import SimpleNamespace

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _RecallResult,
    _dedup_recalled_texts,
    _recall_should_collapse,
    _recall_similarity,
)


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------


class TestRecallSimilarity:
    def test_identical_facts_are_maximally_similar(self):
        assert _recall_similarity("user prefers dark mode", "user prefers dark mode") == 1.0

    def test_reordering_is_still_similar(self):
        assert _recall_similarity("dark mode user prefers", "user prefers dark mode") == 1.0

    def test_different_facts_are_clearly_dissimilar(self):
        assert _recall_similarity("user prefers dark mode", "the project is written in Python") < 0.5


# ---------------------------------------------------------------------------
# Pure dedup helper
# ---------------------------------------------------------------------------


class TestDedupRecalledTexts:
    def test_exact_restatements_collapse_to_first(self):
        out = _dedup_recalled_texts([
            "user prefers dark mode",
            "user prefers dark mode",
            "user prefers dark mode",
        ])
        assert out == ["user prefers dark mode"]

    def test_filler_only_rephrasing_collapses(self):
        # Differing tokens are all stopwords/filler -> restatement, collapse.
        out = _dedup_recalled_texts([
            "user prefers dark mode",
            "the user prefers dark mode",
        ])
        assert out == ["user prefers dark mode"]

    def test_content_word_rephrasing_keeps_both(self):
        # 'color scheme' vs 'mode' differ by a content word -> update, keep both.
        out = _dedup_recalled_texts([
            "user prefers dark mode",
            "user prefers the dark color scheme",
        ])
        assert len(out) == 2

    def test_distinct_facts_are_all_kept(self):
        out = _dedup_recalled_texts([
            "user prefers dark mode",
            "the project uses a SQLite database",
            "team meets on wednesdays",
        ])
        assert len(out) == 3

    def test_contradictory_near_duplicates_are_both_kept(self):
        # Near-identical but opposite polarity — a contradiction, not a restatement.
        out = _dedup_recalled_texts([
            "user prefers dark mode",
            "user does not prefer dark mode",
        ])
        assert len(out) == 2

    def test_negation_spelling_variants_do_not_merge(self):
        out = _dedup_recalled_texts([
            "user likes dark mode",
            "user doesn't like dark mode",
        ])
        assert len(out) == 2

    def test_order_is_preserved_first_wins(self):
        out = _dedup_recalled_texts([
            "first fact about cats",
            "second fact about dogs",
            "first fact about cats",  # duplicate of first, later — dropped
        ])
        assert out == ["first fact about cats", "second fact about dogs"]


# ---------------------------------------------------------------------------
# Integration: dedup runs on both recall paths
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    from pathlib import Path

    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user-home"))

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")

    from unittest.mock import AsyncMock

    client = SimpleNamespace()
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[
                SimpleNamespace(text="user prefers dark mode"),
                SimpleNamespace(text="user prefers dark mode"),
                SimpleNamespace(text="user prefers dark mode"),
            ]
        )
    )
    p._client = client
    return p


class TestReviewRegressions:
    """Pins the failure shapes from the AI review of this PR.

    Facts differing in a number, date, proper noun, or rare content word are
    UPDATES, not restatements — they must never collapse. Double-negation
    pairs over different targets must also stay distinct."""

    def test_number_swapped_facts_both_kept(self):
        out = _dedup_recalled_texts([
            "user's daughter is 7",
            "user's daughter is 8",
        ])
        assert len(out) == 2

    def test_day_swapped_facts_both_kept(self):
        out = _dedup_recalled_texts([
            "meeting moved to tuesday",
            "meeting moved to wednesday",
        ])
        assert len(out) == 2

    def test_secret_id_swap_both_kept(self):
        out = _dedup_recalled_texts([
            "deploy key is abc123",
            "deploy key is abc456",
        ])
        assert len(out) == 2

    def test_double_negation_different_targets_both_kept(self):
        # Both sides carry negation markers, but over DIFFERENT targets —
        # the old polarity-only guard wrongly merged these.
        out = _dedup_recalled_texts([
            "user doesn't like dark mode",
            "user doesn't like light mode",
        ])
        assert len(out) == 2

    def test_double_negation_same_target_collapses(self):
        # Same negated target, only filler differs -> collapse is safe.
        out = _dedup_recalled_texts([
            "never deploy on friday",
            "never not deploy on the friday",
        ])
        assert len(out) == 1

    def test_proper_noun_swap_both_kept(self):
        out = _dedup_recalled_texts([
            "Alice owns the deploy pipeline",
            "Bob owns the deploy pipeline",
        ])
        assert len(out) == 2

    def test_true_restatement_still_collapses_with_numbers(self):
        # Same numbers + same content words, only filler differs -> collapse.
        out = _dedup_recalled_texts([
            "the build takes 42 seconds on the main runner",
            "build takes 42 seconds on main runner",
        ])
        assert len(out) == 1

    def test_similarity_threshold_documented(self):
        # Two collapsible facts may differ ONLY by stopwords/filler; any
        # single content-token difference blocks collapse. This documents
        # the effective minimum edit distance: one content word = keep both.
        assert not _recall_should_collapse(
            "user prefers dark mode", "user prefers dark theme"
        )
        assert _recall_similarity("user prefers dark mode", "user prefers dark theme") >= 0.75


class TestRecallToolDedup:
    def test_recall_tool_dedupes_restated_facts(self, provider):
        result = json.loads(provider.handle_tool_call("hindsight_recall", {"query": "dark"}))
        assert result["result"] == "1. user prefers dark mode"
        assert result["result"].count("dark mode") == 1


class TestRecallAutoInjectionDedup:
    def test_auto_injection_dedupes_and_collapses_other_facts(self, provider):
        # Rework the mock to return two distinct + one duplicate.
        from unittest.mock import AsyncMock

        provider._client.arecall = AsyncMock(
            return_value=SimpleNamespace(
                results=[
                    SimpleNamespace(text="fact alpha"),
                    SimpleNamespace(text="fact alpha"),
                    SimpleNamespace(text="fact beta"),
                ]
            )
        )
        res = provider._do_recall("alpha")
        assert isinstance(res, _RecallResult)
        assert res.count == 2  # fact alpha collapsed to 1, fact beta kept
        assert res.text.count("fact alpha") == 1
        assert res.text.count("fact beta") == 1

    def test_auto_injection_keeps_contradictions(self, provider):
        from unittest.mock import AsyncMock

        provider._client.arecall = AsyncMock(
            return_value=SimpleNamespace(
                results=[
                    SimpleNamespace(text="user prefers dark mode"),
                    SimpleNamespace(text="user does not prefer dark mode"),
                ]
            )
        )
        res = provider._do_recall("dark")
        assert res.count == 2  # contradiction preserved — no merge
        assert "does not prefer" in res.text
        assert "prefers dark" in res.text
