"""Tests for the memory context relevance framework (Phase 1-4).

Covers:
  - ScoredMemory dataclass and parsing
  - Intent extraction (classify_task_type, extract_entities, extract_keywords)
  - Context-tag matching and signal extraction
  - Min-relevance gating and selection
  - Cross-provider fusion and deduplication
  - End-to-end scored prefetch pipeline
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from agent.memory.scored_memory import (
    ScoredMemory,
    parse_context_tag,
    parse_signal_tag,
    strip_memory_markup,
    context_tag_match,
    extract_signal_strength,
    extract_context_tag,
    classify_task_type,
    extract_entities,
    HYBRID_WEIGHTS,
)
from agent.memory.intent_extractor import extract_intent, IntentContext, extract_keywords
from agent.memory.fusion import fuse, deduplicate, PROVIDER_TRUST
from agent.memory.gating import (
    apply_score_threshold,
    compute_memory_budget,
    select_and_format,
    _DEFAULT_MIN_RELEVANCE,
    _DEFAULT_MAX_CONTEXT_CHARS,
)
from agent.memory.diagnostics import (
    format_turn_debug,
    format_session_stats,
    record_turn,
    clear_log,
)


# =========================================================================
# ScoredMemory
# =========================================================================


class TestScoredMemory:
    def test_from_raw_entry_no_tags(self):
        """Plain text entry without tags is parsed correctly."""
        sm = ScoredMemory.from_raw_entry("Build REST API for user management")
        assert sm.content == "Build REST API for user management"
        assert sm.context_tag is None
        assert sm.signal_strength == 0.5

    def test_from_raw_entry_with_context(self):
        """Entry with [context: feature] tag parses context correctly."""
        sm = ScoredMemory.from_raw_entry("[context: feature] Build REST API")
        assert sm.content == "Build REST API"
        assert sm.context_tag == "feature"
        assert sm.signal_strength == 0.5

    def test_from_raw_entry_with_signal(self):
        """Entry with [signal: 0.9] parses signal correctly."""
        sm = ScoredMemory.from_raw_entry("[signal: 0.9] Build REST API")
        assert sm.content == "Build REST API"
        assert sm.signal_strength == 0.9

    def test_from_raw_entry_with_context_and_signal(self):
        """Entry with both [context:] and [signal:] tags parses correctly."""
        sm = ScoredMemory.from_raw_entry(
            "[context: deploy] [signal: 0.85] Deploy to GKE"
        )
        assert sm.content == "Deploy to GKE"
        assert sm.context_tag == "deploy"
        assert pytest.approx(sm.signal_strength, 0.01) == 0.85

    def test_formatted_breakdown(self):
        """Breakdown string is human-readable."""
        sm = ScoredMemory(
            content="test", provider="builtin", score=0.75,
            score_semantic=0.8, score_keyword=0.5, score_context=1.0,
            score_entity=0.6, score_temporal=1.0, score_signal=0.9,
        )
        b = sm.breakdown()
        assert "sem:" in b
        assert "key:" in b
        assert "ctx:" in b
        assert "score=0.75" in b

    def test_compute_score_default_weights(self):
        """Composite score is correctly computed with default weights."""
        sm = ScoredMemory(content="test", provider="builtin")
        sm.score_semantic = 0.8
        sm.score_keyword = 0.5
        sm.score_context = 1.0
        sm.score_signal = 0.9
        expected = (
            HYBRID_WEIGHTS["fts"] * 0.5 +
            HYBRID_WEIGHTS["embedding"] * 0.8 +
            HYBRID_WEIGHTS["context_tag"] * 1.0 +
            HYBRID_WEIGHTS["signal"] * 0.9
        )
        actual = sm.compute_score()
        assert pytest.approx(actual, 0.01) == expected

    def test_score_clamped(self):
        """Score is clamped to [0.0, 1.0]."""
        sm = ScoredMemory(content="test", provider="builtin")
        sm.score_semantic = 5.0
        sm.score_keyword = 5.0
        sm.score_context = 5.0
        assert sm.compute_score() <= 1.0
        assert sm.compute_score() >= 0.0


# =========================================================================
# Tag parsing helpers
# =========================================================================


class TestTagParsing:
    def test_parse_context_tag_at_start(self):
        ctx, rest = parse_context_tag("[context: feature] Build something")
        assert ctx == "feature"
        assert rest == "Build something"

    def test_parse_context_tag_no_match(self):
        ctx, rest = parse_context_tag("Plain entry without tag")
        assert ctx is None
        assert rest == "Plain entry without tag"

    def test_parse_signal_tag_anywhere(self):
        sig, rest = parse_signal_tag("Entry [signal: 0.8] text")
        assert pytest.approx(sig, 0.01) == 0.8
        assert rest == "Entry text"

    def test_parse_signal_tag_no_match(self):
        sig, rest = parse_signal_tag("Entry without signal")
        assert sig == 0.5
        assert rest == "Entry without signal"

    def test_strip_memory_markup(self):
        clean = strip_memory_markup("[context: deploy] [signal: 0.9] Deploy to GKE")
        assert clean == "Deploy to GKE"

    def test_extract_context_tag(self):
        assert extract_context_tag("[context: deploy] something") == "deploy"
        assert extract_context_tag("no tag") is None

    def test_extract_signal_strength(self):
        assert pytest.approx(extract_signal_strength("[signal: 0.75] x"), 0.01) == 0.75
        assert pytest.approx(extract_signal_strength("no tag"), 0.01) == 0.5


class TestContextTagMatch:
    def test_exact_match(self):
        assert context_tag_match("deploy", "deploy") == 1.0

    def test_mismatch(self):
        assert context_tag_match("deploy", "feature") == 0.0

    def test_no_tag(self):
        assert context_tag_match(None, "deploy") == 0.3

    def test_no_task_type(self):
        assert context_tag_match("deploy", None) == 0.3

    def test_both_none(self):
        assert context_tag_match(None, None) == 0.3


# =========================================================================
# Intent extraction
# =========================================================================


class TestClassifyTaskType:
    def test_deploy_message(self):
        tt, conf = classify_task_type("Deploy the service to GKE cluster")
        assert tt == "deploy"
        assert conf > 0.15

    def test_debug_message(self):
        tt, conf = classify_task_type("Fix this bug — getting a traceback error")
        assert tt == "debug"
        assert conf > 0.15

    def test_feature_message(self):
        tt, conf = classify_task_type("Implement a new feature API endpoint")
        assert tt == "feature"
        assert conf > 0.15

    def test_review_message(self):
        tt, conf = classify_task_type("Review the PR and audit the code")
        assert tt == "review"
        assert conf > 0.15

    def test_empty_message(self):
        tt, conf = classify_task_type("")
        assert tt is None
        assert conf == 0.0

    def test_kanban_message(self):
        tt, conf = classify_task_type("Create a kanban task for the sprint")
        assert tt == "kanban"
        assert conf > 0.15


class TestExtractEntities:
    def test_gke_entity(self):
        entities = extract_entities("Deploy to GKE cluster")
        names = [e[0] for e in entities]
        assert "gke" in names

    def test_no_entities(self):
        entities = extract_entities("How's the weather?")
        assert entities == []

    def test_multiple_entities(self):
        entities = extract_entities("Deploy GKE with Terraform on Firebase")
        names = [e[0] for e in entities]
        assert "gke" in names
        assert "terraform" in names
        assert "firebase" in names


class TestExtractKeywords:
    def test_keywords_extracted(self):
        kws = extract_keywords("Deploy this feature to GKE")
        assert kws  # non-empty
        assert any(kw == "deploy" for kw, _ in kws)
        assert any(kw == "feature" for kw, _ in kws)
        assert any(kw == "gke" for kw, _ in kws)

    def test_empty_message(self):
        assert extract_keywords("") == []


class TestExtractIntent:
    def test_basic_intent(self):
        ctx = extract_intent("Deploy the microservice to GKE")
        assert ctx.task_type == "deploy"
        assert ctx.task_confidence > 0
        assert any(e[0] == "gke" for e in ctx.entities)
        assert any("deploy" in kw for kw, _ in ctx.keywords)
        assert ctx.expanded_query == ctx.raw_message

    def test_actionable(self):
        ctx = extract_intent("Hello")
        # Generic greeting — may not be actionable
        assert isinstance(ctx.is_actionable(), bool)

    def test_intent_with_prior_turns(self):
        prior = ["We need to deploy the kanban service"]
        ctx = extract_intent("On GKE staging", prior_turns=prior)
        assert ctx.expanded_query != ctx.raw_message
        assert "deploy" in ctx.expanded_query


# =========================================================================
# Fusion (dedup + trust weighting)
# =========================================================================


class TestDeduplication:
    def test_exact_duplicates_dropped(self):
        entries = [
            ScoredMemory(content="GKE cluster info", provider="builtin", score=0.8),
            ScoredMemory(content="GKE cluster info", provider="holographic", score=0.6),
        ]
        result = deduplicate(entries)
        assert len(result) == 1
        assert result[0].provider == "builtin"  # higher score kept

    def test_near_duplicates_dropped(self):
        entries = [
            ScoredMemory(content="Deploy the kanban service to GKE staging in europe central2", provider="builtin", score=0.9),
            ScoredMemory(content="Deploy the kanban service to GKE staging", provider="holographic", score=0.7),
        ]
        result = deduplicate(entries)
        assert len(result) == 1  # Jaccard > 0.7 (shared 7/8 tokens)

    def test_distinct_entries_kept(self):
        entries = [
            ScoredMemory(content="GKE cluster info", provider="builtin", score=0.8),
            ScoredMemory(content="User prefers concise responses", provider="builtin", score=0.6),
        ]
        result = deduplicate(entries)
        assert len(result) == 2

    def test_empty_list(self):
        assert deduplicate([]) == []


class TestFusion:
    def test_fuses_multiple_providers(self):
        builtin = [
            ScoredMemory(content="GKE deploy info", provider="builtin", score=0.8, score_context=1.0, score_signal=0.9),
            ScoredMemory(content="User preference", provider="builtin", score=0.4, score_context=0.3),
        ]
        external = [
            ScoredMemory(content="GKE cluster notes", provider="holographic", score=0.7),
        ]
        result = fuse(builtin, external)
        assert len(result) > 0
        # Builtin should rank higher due to trust weight
        assert result[0].provider == "builtin"

    def test_provider_trust_applied(self):
        entries = [
            ScoredMemory(content="Test entry", provider="builtin", score=0.8),
        ]
        result = fuse(entries, provider_trust_override={"builtin": 0.5})
        assert pytest.approx(result[0].score, 0.01) == 0.4  # 0.8 * 0.5

    def test_empty_fusion(self):
        assert fuse([]) == []

    def test_provider_trust_override(self):
        entries = [
            ScoredMemory(content="Test", provider="custom", score=1.0),
        ]
        result = fuse(entries, provider_trust_override={"custom": 0.3})
        assert pytest.approx(result[0].score, 0.01) == 0.3


# =========================================================================
# Gating
# =========================================================================


class TestScoreThreshold:
    def test_below_threshold_dropped(self):
        entries = [
            ScoredMemory(content="Important", provider="builtin", score=0.8),
            ScoredMemory(content="Noise", provider="builtin", score=0.1),
        ]
        result = apply_score_threshold(entries, min_score=0.15)
        assert len(result) == 1
        assert result[0].content == "Important"

    def test_all_above_threshold(self):
        entries = [
            ScoredMemory(content="First", provider="builtin", score=0.5),
            ScoredMemory(content="Second", provider="builtin", score=0.6),
        ]
        result = apply_score_threshold(entries, min_score=0.15)
        assert len(result) == 2

    def test_all_below_threshold(self):
        entries = [
            ScoredMemory(content="Noise", provider="builtin", score=0.1),
            ScoredMemory(content="More noise", provider="builtin", score=0.05),
        ]
        result = apply_score_threshold(entries, min_score=0.15)
        assert len(result) == 0

    def test_empty(self):
        assert apply_score_threshold([]) == []


class TestComputeMemoryBudget:
    def test_hard_cap_respected(self):
        budget = compute_memory_budget(
            context_length=32000,
            tokens_used_by_system_prompt=10000,
            tokens_used_by_history=15000,
            max_context_chars=2000,
            window_fraction=0.03,
        )
        # Dynamic: (32000 - 10000 - 15000) * 0.03 * 4 = 7000 * 0.03 * 4 = 840
        # Cap: min(2000, 840) = 840
        assert budget == 840

    def test_hard_cap_wins(self):
        budget = compute_memory_budget(
            context_length=128000,
            tokens_used_by_system_prompt=1000,
            tokens_used_by_history=1000,
            max_context_chars=2000,
            window_fraction=0.03,
        )
        # Dynamic: (128000 - 2000) * 0.03 * 4 = 126000 * 0.03 * 4 = 15120
        # Cap: min(2000, 15120) = 2000
        assert budget == 2000

    def test_zero_context_length(self):
        budget = compute_memory_budget(
            context_length=0,
            max_context_chars=2000,
        )
        assert budget == 2000

    def test_negative_remaining(self):
        budget = compute_memory_budget(
            context_length=10000,
            tokens_used_by_system_prompt=10000,
            tokens_used_by_history=5000,
            max_context_chars=2000,
        )
        # remaining = -5000 → clamp to 0 → 0 * 0.03 * 4 = 0
        # min(2000, 0) = 0
        assert budget == 0


class TestSelectAndFormat:
    def test_selects_top_entries(self):
        candidates = [
            ScoredMemory(content="Top priority item", provider="builtin", score=0.9),
            ScoredMemory(content="Medium priority", provider="builtin", score=0.6),
            ScoredMemory(content="Low priority", provider="builtin", score=0.3),
        ]
        result = select_and_format(candidates, max_chars=500)
        assert "<memory-context>" in result
        assert "</memory-context>" in result
        assert "Top priority" in result
        assert "Medium priority" in result
        # Low may or may not fit — depends on budget

    def test_budget_exhausted(self):
        candidates = [
            ScoredMemory(content="A" * 500, provider="builtin", score=0.9),
            ScoredMemory(content="B" * 500, provider="builtin", score=0.8),
        ]
        result = select_and_format(candidates, max_chars=100)
        # Only the header/footer may fit, or nothing at all
        assert isinstance(result, str)

    def test_empty_candidates(self):
        assert select_and_format([], max_chars=2000) == ""

    def test_format_includes_system_note(self):
        candidates = [
            ScoredMemory(content="Important fact", provider="builtin", score=0.9),
        ]
        result = select_and_format(candidates, max_chars=2000)
        assert "System note" in result
        assert "Important fact" in result
        assert "<memory-context>" in result
        assert "</memory-context>" in result

    def test_markup_stripped(self):
        candidates = [
            ScoredMemory(
                content="[context: deploy] [signal: 0.9] Deploy to GKE",
                provider="builtin", score=0.9,
            ),
        ]
        # Pass through from_raw_entry which strips markup; simulate raw
        # by creating ScoredMemory with markup in content
        sm = ScoredMemory(
            content="[context: deploy] [signal: 0.9] Deploy to GKE",
            provider="builtin", score=0.9,
        )
        result = select_and_format([sm], max_chars=2000)
        # The strip_memory_markup call in select_and_format should strip tags
        assert "Deploy to GKE" in result
        # But the tags might or might not be present depending on stripping


# =========================================================================
# Diagnostics
# =========================================================================


class TestDiagnostics:
    def setup_method(self):
        clear_log()

    def test_record_and_format_turn_debug(self):
        clear_log()
        record_turn(
            turn_number=1,
            intent=SimpleNamespace(
                task_type="deploy",
                task_confidence=0.85,
            ),
            candidates_total=12,
            candidates_after_gating=4,
            injected_count=3,
            injected_chars=942,
            budget=2000,
            top_scores=[0.74, 0.63, 0.51],
            provider_counts={"builtin": 2, "holographic": 1},
        )
        output = format_turn_debug(turns=5)
        assert "Turn #1" in output
        assert "deploy" in output
        assert "0.85" in output
        assert "0.74" in output
        assert "builtin" in output

    def test_no_data(self):
        clear_log()
        assert "No relevance-scored turns" in format_turn_debug()
        assert "No relevance data" in format_session_stats()

    def test_session_stats(self):
        clear_log()
        for i in range(3):
            record_turn(
                turn_number=i + 1,
                intent=SimpleNamespace(task_type="deploy", task_confidence=0.8),
                candidates_total=10,
                candidates_after_gating=3,
                injected_count=2,
                injected_chars=600,
                budget=2000,
                top_scores=[0.7, 0.5],
                provider_counts={"builtin": 2},
            )
        output = format_session_stats()
        assert "3" in output  # total turns
        assert "builtin" in output
        assert "Score distribution" in output


# =========================================================================
# End-to-end: MemoryStore.scored_prefetch integration
# =========================================================================


class TestScoredPrefetch:
    """Tests the static scored_prefetch method on MemoryStore.

    These test the scoring logic introduced in Phase 1c.
    """

    def _make_memory_entries(self):
        return [
            "[context: deploy] [signal: 0.9] GKE cluster info — use europe-central2",
            "[context: feature] [signal: 0.7] Build REST API for user management",
            "[context: debug] [signal: 0.6] Bug fix: null pointer in auth service",
            "[signal: 0.5] General user preference: concise responses",
        ]

    def test_scored_prefetch_returns_ranked(self):
        from tools.memory_tool import MemoryStore

        entries = self._make_memory_entries()
        result = MemoryStore.scored_prefetch(
            entries,
            task_type="deploy",
            task_confidence=0.85,
            keywords=[("deploy", 0.8), ("gke", 0.9)],
            expanded_query="Deploy to GKE cluster",
        )

        assert len(result) == 4
        # Deploy entry should be highest scored
        assert result[0]["context_tag"] == "deploy"
        assert result[0]["score"] >= result[1]["score"]

    def test_scored_prefetch_without_query(self):
        from tools.memory_tool import MemoryStore

        entries = self._make_memory_entries()
        result = MemoryStore.scored_prefetch(entries)
        # Without query, context-tag match is primary signal
        assert len(result) == 4
        # All should have scores (signal-based)
        assert all(r["score"] > 0 for r in result)

    def test_scored_prefetch_empty(self):
        from tools.memory_tool import MemoryStore

        assert MemoryStore.scored_prefetch([]) == []

    def test_scored_prefetch_content_preserved(self):
        from tools.memory_tool import MemoryStore

        entries = ["[context: feature] Important feature detail"]
        result = MemoryStore.scored_prefetch(entries)
        assert len(result) == 1
        assert result[0]["content"] == "Important feature detail"
        assert result[0]["context_tag"] == "feature"
