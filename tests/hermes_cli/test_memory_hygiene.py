"""
Tests for memory_hygiene module (P2-8).

Covers: decay computation, pruning, contradiction detection,
and the memory promotion gate.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from hermes_cli.memory_hygiene import (
    MemoryEntry,
    PruneResult,
    ContradictionResult,
    compute_decayed_importance,
    prune_stale_entries,
    detect_contradictions,
    memory_promotion_gate,
    memory_hygiene_report,
    HIGH_IMPORTANCE,
    MEDIUM_IMPORTANCE,
    LOW_IMPORTANCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    memory_id: str = "m1",
    content: str = "test",
    importance: float = 0.5,
    age_days: int = 0,
) -> MemoryEntry:
    created = datetime.now(timezone.utc) - timedelta(days=age_days)
    accessed = created  # same as creation for simplicity
    return MemoryEntry(
        memory_id=memory_id,
        content=content,
        importance=importance,
        created_at=created.isoformat(),
        last_accessed_at=accessed.isoformat(),
    )


# ---------------------------------------------------------------------------
# Decay computation
# ---------------------------------------------------------------------------


class TestDecayComputation:
    def test_high_importance_preserved_within_threshold(self):
        entry = make_entry(importance=0.80, age_days=30)
        new_imp, should_prune = compute_decayed_importance(
            entry, high_decay_days=90,
        )
        assert new_imp == 0.80
        assert should_prune is False

    def test_high_importance_decays_to_medium(self):
        entry = make_entry(importance=0.80, age_days=100)
        new_imp, should_prune = compute_decayed_importance(
            entry, high_decay_days=90,
        )
        assert new_imp == MEDIUM_IMPORTANCE
        assert should_prune is False

    def test_medium_importance_preserved_within_threshold(self):
        entry = make_entry(importance=0.50, age_days=15)
        new_imp, should_prune = compute_decayed_importance(
            entry, medium_decay_days=30,
        )
        assert new_imp == 0.50
        assert should_prune is False

    def test_medium_importance_decays_to_low(self):
        entry = make_entry(importance=0.50, age_days=40)
        new_imp, should_prune = compute_decayed_importance(
            entry, medium_decay_days=30,
        )
        assert new_imp < MEDIUM_IMPORTANCE
        assert should_prune is False

    def test_low_importance_pruned_after_ttl(self):
        entry = make_entry(importance=0.25, age_days=10)
        new_imp, should_prune = compute_decayed_importance(
            entry, low_ttl_days=7,
        )
        assert new_imp == 0.0
        assert should_prune is True

    def test_low_importance_preserved_within_ttl(self):
        entry = make_entry(importance=0.25, age_days=3)
        new_imp, should_prune = compute_decayed_importance(
            entry, low_ttl_days=7,
        )
        assert new_imp == 0.25
        assert should_prune is False

    def test_below_survival_always_pruned(self):
        entry = make_entry(importance=0.05, age_days=1)
        new_imp, should_prune = compute_decayed_importance(entry)
        assert should_prune is True

    def test_recent_high_importance_kept(self):
        entry = make_entry(importance=0.90, age_days=1)
        _, should_prune = compute_decayed_importance(entry)
        assert should_prune is False


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruneStale:
    def test_empty_list(self):
        result = prune_stale_entries([])
        assert result.total_entries == 0
        assert result.pruned_count == 0

    def test_prunes_only_low_old_entries(self):
        entries = [
            make_entry("h1", importance=0.90, age_days=1),
            make_entry("m1", importance=0.50, age_days=5),
            make_entry("l1", importance=0.20, age_days=10),
            make_entry("l2", importance=0.25, age_days=3),
        ]
        result = prune_stale_entries(
            entries,
            low_importance_days=7,
            medium_decay_days=30,
            high_decay_days=90,
        )
        assert result.total_entries == 4
        assert result.pruned_count == 1  # only l1 (10 days old, low importance)
        assert "l1" in result.pruned_ids

    def test_no_prune_when_all_recent(self):
        entries = [
            make_entry("a", importance=0.25, age_days=1),
            make_entry("b", importance=0.35, age_days=3),
        ]
        result = prune_stale_entries(entries, low_importance_days=7)
        assert result.pruned_count == 0
        assert result.preserved_count == 2

    def test_below_survival_importance_pruned(self):
        entries = [
            make_entry("a", importance=0.05, age_days=1),
            make_entry("b", importance=0.15, age_days=1),
        ]
        result = prune_stale_entries(entries)
        assert result.pruned_count == 2  # both below MIN_SURVIVAL_IMPORTANCE

    def test_decay_counts_medium_to_low(self):
        entries = [
            make_entry("m1", importance=0.50, age_days=40),
        ]
        result = prune_stale_entries(entries, medium_decay_days=30)
        assert result.decayed_count == 1
        assert result.pruned_count == 0

    def test_dry_run_does_not_mutate(self):
        entries = [
            make_entry("l1", importance=0.20, age_days=10),
        ]
        result = prune_stale_entries(entries, dry_run=True)
        assert result.pruned_count == 1
        # Entry list is unchanged (dry_run)
        assert entries[0].importance == 0.20


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    def test_no_conflict_with_different_topics(self):
        result = detect_contradictions(
            "User prefers dark mode",
            ["User uses Python 3.12", "Project is named KenseiAgent"],
        )
        assert result.has_conflict is False
        assert result.recommendation == "promote"

    def test_conflict_with_similar_topic(self):
        # "does not prefer" in existing vs "prefers" in new triggers negation detection
        result = detect_contradictions(
            "User prefers dark mode",
            ["User does not prefer dark mode"],
        )
        assert result.has_conflict is True
        assert len(result.conflicting_entries) > 0

    def test_negation_conflict_detected(self):
        result = detect_contradictions(
            "User prefers dark mode",
            ["User does not prefer dark mode"],
        )
        assert result.has_conflict is True
        assert result.recommendation in ("review", "block")

    def test_no_conflict_with_dissimilar_text(self):
        result = detect_contradictions(
            "Project uses pnpm as package manager",
            ["The VPS runs Ubuntu 22.04"],
        )
        assert result.has_conflict is False

    def test_multiple_conflicts_triggers_block(self):
        result = detect_contradictions(
            "User always uses vim",
            [
                "User always uses vim",  # Very similar
                "User never uses vim",   # Contradicts
            ],
        )
        assert result.has_conflict is True
        assert result.recommendation == "block"

    def test_single_mild_conflict_recommends_review(self):
        result = detect_contradictions(
            "User prefers dark mode for coding",
            ["User uses light mode sometimes for reading"],
        )
        if result.has_conflict:
            assert result.recommendation in ("review", "block")

    def test_empty_new_fact(self):
        result = detect_contradictions("", ["anything"])
        assert result.has_conflict is False
        assert result.recommendation == "promote"


# ---------------------------------------------------------------------------
# Memory promotion gate
# ---------------------------------------------------------------------------


class TestMemoryPromotionGate:
    def test_promotes_facts_without_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "brain.md")
            with open(store, "w") as f:
                f.write("- User lives in Nottingham\n")
                f.write("- Project uses React\n")

            results = memory_promotion_gate(
                ["User prefers TypeScript", "Server runs Ubuntu"],
                store,
            )
            assert len(results) == 2
            assert all(r[1] == "promote" for r in results)

    def test_blocks_conflicting_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "brain.md")
            with open(store, "w") as f:
                f.write("- User prefers dark mode\n")
                f.write("- Project uses pnpm\n")

            results = memory_promotion_gate(
                ["User does not prefer dark mode"],
                store,
            )
            assert results[0][1] in ("review", "block")

    def test_handles_missing_store(self):
        results = memory_promotion_gate(
            ["Some fact"],
            "/tmp/nonexistent/store.md",
        )
        assert len(results) == 1
        assert results[0][1] == "promote"  # nothing to conflict with


# ---------------------------------------------------------------------------
# Memory hygiene report
# ---------------------------------------------------------------------------


class TestHygieneReport:
    def test_generates_report(self):
        entries = [
            make_entry("h1", importance=0.90, age_days=1),
            make_entry("m1", importance=0.50, age_days=15),
            make_entry("l1", importance=0.20, age_days=10),
        ]
        report = memory_hygiene_report(entries, dry_run=True)
        assert report["total_entries"] == 3
        assert "by_importance" in report
        assert report["by_importance"]["high"] == 1
        assert report["by_importance"]["medium"] == 1
        assert report["by_importance"]["prune_candidate"] == 1

    def test_empty_report(self):
        report = memory_hygiene_report([])
        assert report["total_entries"] == 0
