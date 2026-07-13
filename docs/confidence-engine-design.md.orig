# Confidence Scoring Engine — Full Design

> Background document for the generic engine submitted in this PR.

## Problem

Memory in long-running AI agents grows without bound. Entries accumulate — some are transient troubleshooting notes, others are permanent user preferences — but they're all treated equally. When the memory cap is hit, the agent evicts blindly, often sacrificing high-value recent information while keeping stale transient data.

At 162 entries across 4 months of daily use, this problem became concrete. The agent started dropping freshly written preferences to make room for months-old completed-task records.

## Solution: Four-Dimensional Scoring

The core formula is `score = source_weight × recency_decay + Σ(bonus_keywords × clamped_weight)`, bounded [0, 1].

### Dimension 1 — Source Trust

| Source | Weight | Rationale |
|:--|:--:|:--|
| User-stated | 1.0 | User explicitly said this. Permanent TTL candidate. |
| LLM-inferred | 0.4 | Agent extracted from context. Subject to hallucination. Lower starting score. |

### Dimension 2 — Recency Decay

| Age | Decay | Rationale |
|:--|:--:|:--|
| 0–30 days | 1.0 | Current |
| 30–90 days | 0.8 | Recent |
| 90–180 days | 0.6 | Aging |
| 180–365 days | 0.4 | Stale |
| >365 days | 0.2 | Archive territory |

**Decay bypass**: Entries matching a CONSTANT_SAFELIST (birthdays, allergies, permanent rules) skip decay entirely. Separately, `fact_type_constant` entries — those without transient language markers — also bypass decay.

### Dimension 3 — Fact Type Classification

Based on arxiv 2604.11364's three-tier fact classification:

| Type | Detection | Decay behavior |
|:--|:--|:--|
| **Constant** | No transient markers + source=user_stated + length>20, OR safelist match | Decay bypassed |
| **Mutable** | All other entries | Normal decay |
| **Transient** | Keywords: "temporary," "one-time," "this time" | Accelerated decay (×0.5) |

### Dimension 4 — Bonus Keywords

Content patterns detected via regex that add or subtract from the base score:

| Bonus | Weight | Example pattern |
|:--|:--:|:--|
| Permanence marker | +0.3 | "permanent", "always", "lifetime" |
| Lesson learned | +0.2 | "lesson", "methodology", "rule", "principle" |
| Personal preference | +0.2 | "prefers", "habit", "usually" |
| Work decision | +0.1 | "decided", "chose", "designed" |
| Completed task | -0.2 | "completed", "deployed", "fixed" |
| One-time info | -0.3 | "temporary", "one-time", "this time" |

All bonuses clamped to [-0.3, 0.3].

### Threshold Mapping

| Score | Level | TTL | Action |
|:--|:--|:--|:--|
| ≥0.80 | Permanent | Never expire | Keep |
| ≥0.50 | Long-term | 365 days | Keep |
| ≥0.30 | Medium-term | 180 days | Keep |
| ≥0.10 | Short-term | 90 days | Review |
| <0.10 | Volatile | 30 days | Evict |

## Production Validation

The full engine (with project-specific safelist, Chinese regex patterns, and source classification from a separate Phase 5b pipeline) has been scoring 162 memory entries daily for 4 months. Key observations:

- **0 false evictions**: No user-stated critical facts have been flagged for eviction
- **3:1 signal ratio**: ~75% of entries fall into long-term/permanent tiers, ~25% into review/evict tiers — manageable churn
- **Batch performance**: `assess_batch(162)` completes in <50ms on consumer hardware

## What's Omitted from the Generic Version

| Omitted | Reason |
|:--|:--|
| CONSTANT_SAFELIST patterns | Project-specific (personal health, family, hostnames) |
| Chinese regex bonus keywords | Language-specific |
| Source classification pipeline | Depends on a separate Phase 5b metadata system |
| Tuned decay thresholds | Calibrated per-project usage patterns |

These are documented here for context but excluded from the generic PR to keep it minimally viable and project-agnostic.

## References

- arxiv 2604.11364: Four-tier confidence model for agent memory systems
- Google SRE: Error budget and SLO concepts applied to memory eviction
- Dead Man's Snitch pattern: Silent-when-healthy watchdog for eviction monitoring
