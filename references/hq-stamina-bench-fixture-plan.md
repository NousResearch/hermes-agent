# HQ Stamina Bench Fixture Plan

## Overview
StaminaBench-style multi-turn stamina fixtures for the HQ harness validator. Designed to run as docs-only prep; no live enforcement, no cron/Gateway mutation.

## Why now
- Planner source: `cron_331ecf7312ab_20260621_150248`
- upstream finding: good harness lifts model performance up to 6x; test-feedback-retry loops improve pass turns by 12x (StaminaBench, arXiv:2606.19613)
- all tested models fail within 5–6 turns without harness — long autonomous runs (cron, tool chains, Build/Review loop) are unstable without stamina checks
- current HQ state: dry-run harness only; live enforcement absent

## Scope (docs-only)
- design document only
- no code insertion into `scripts/hq_harness_validator.py` while worktree is `substantive-tracked`

## Design

### Programmatic Change-Request Sampler
- input: fixture spec (max_turns, failure_rate, domain)
- output: deterministic sequence of state changes (file edits, test runs, phase transitions)
- constraints:
  - each turn must be reproducible from seed + turn index
  - failure injection must be explicit and isolated

### 10-turn Baseline
- deterministic scenario covering:
  1. initial harness state
  2. one passing test
  3. one failing test + retry
  4. edit + test + pass
  5. phase transition (impl/refactor)
  6. memory/skill boundary check
  7. tool contract violation + recovery
  8. context compression trigger
  9. approval gate hit
  10. final state diff summary

### Harness Comparison
- run same seed set through:
  - current HQ harness validator (`hq_harness_validator.py`)
  - hypothetical "no harness" baseline (same state machine, no gate checks)
- compare:
  - pass@turn count
  - failure recovery rate
  - phase-transition accuracy
  - time-to-first-failure

## Target Integration (future, approval-gated)
- location: `scripts/hq_harness_validator.py` `StaminaFixtureGenerator` class
- tests: `tests/scripts/test_hq_harness_validator.py` new `TestStaminaFixtures` class
- rollback: revert fixture addition without touching existing validators

## Verification (read-only)
- `py_compile` on any future script file
- deterministic seed reproducibility: same seed → same turn sequence + outcomes
- fixture count >= 10, failure injection coverage >= 40% of turns

## Safety
- no secret access
- no live cron/Gateway changes
- no paid-provider calls
- docs-only until worktree is clean and explicitly approved for code insertion
