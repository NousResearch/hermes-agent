# Plan: Ares Context Governor Candidate-Custody Repair

**Created**: 2026-08-14
**Status**: Researching

## Goal

Repair the existing Ares candidate-custody implementation against the prior Sol
hostile-audit findings, preserve the failed candidate unchanged, and build a
new independently recoverable candidate without touching live Context Governor
state.

## Research

- [x] Read applicable global and repository instructions.
- [x] Record dirty-worktree and repository HEAD baselines.
- [x] Verify cited failed-candidate archive identity.
- [ ] Recover and verify the prior hostile-audit finding set.
- [ ] Map each current implementation path and existing regression coverage.

## Analysis

The worktrees already contain uncommitted Ares custody modules, candidate
builder scripts, and tests. These are the implementation baseline, not
disposable changes. The failed candidate is under canonical custody and will
only be read and fingerprinted.

## Implementation Steps

1. [ ] Verify failed-candidate manifests, audit subject, handoff, and prior audit evidence.
2. [ ] Write locked regression tests for each verified custody/activation/lease/GC finding.
3. [ ] Repair CandidateStore inventory, descriptor continuity, audit lease, GC, fault matrix, and runtime packaging.
4. [ ] Run focused red/green tests and full repository gates.
5. [ ] Create, certify, seal, publish, and independently recover a new candidate.
6. [ ] Produce a precise Sol hostile-audit handoff prompt and report.

## Files to Read First

- `hermes_cli/ares_candidate_store.py` — custody owner.
- `hermes_cli/ares_candidate_lifecycle.py` — lifecycle sequence and locks.
- `hermes_cli/ares_candidate_recovery.py` / `ares_candidate_gc.py` — recovery and deletion authority.
- `scripts/ares_context_governor_candidate.py` — candidate construction and post-seal evidence.
- `tests/hermes_cli/test_ares_candidate_store.py` — existing custody contract coverage.

## Verification

- [ ] Focused CandidateStore/lifecycle/recovery/GC tests through `scripts/run_tests.sh`.
- [ ] Candidate-builder and installed-runtime isolation tests.
- [ ] Ruff 0.15.10 lint and format check over candidate payload files.
- [ ] Python compilation and `git diff --check`.
- [ ] Context Governor fmt, clippy, Rust tests, and Python tests.
- [ ] Fresh-process recovery from canonical persistent custody.

## Safety Constraints

- Never modify, promote, activate, GC, or re-sign the cited failed candidate.
- Never read the legacy Context Governor HMAC key.
- Never activate Context Governor live or modify live Hermes/Ares config, services, receipts, or key state.
- No commit, push, reset, stash, or clean.

## Progress Log

### 2026-08-14 — Planning

- Captured repository HEAD and dirty-worktree baseline.
- Archive SHA-256 of the failed candidate matches the cited value.

## Test Lock

Tests written: 2026-08-14 (locked)

- `tests/hermes_cli/test_ares_candidate_store.py` — exact artifact-tree
  inventory, full lifecycle history, counterfeit/missing audit lease, and
  malformed-tombstone recovery. These tests must be satisfied by custody
  implementation changes rather than weakened assertions.
