# PR 98470 Passive Worker Contract Replacement Design

## Objective

Replace the cumulative upstream PR `NousResearch/hermes-agent#98470` with a
reviewable contribution that contains only the passive, evidence-aware worker
contract originally introduced by commit
`e0b848fed2f0bb446237c066eaa991e454096a43`.

The replacement must preserve contributor authorship, start from exact upstream
`main` commit `1cf36398135f4848a1d04b2167ffb564b7881d35`, and avoid changing the existing
published PR until the replacement is independently verified.

## Included Scope

The production replacement is restricted to these two files:

- `agent/worker_contract.py`
- `tests/agent/test_worker_contract.py`

The intended behavior is passive contract construction and validation:

- bind task, owner, profile, repository, and authority identity;
- distinguish observations, inferences, unknowns, and evidence sources;
- validate safety, freshness, authorization, and reversible next actions;
- serialize contract payloads safely; and
- report partial, blocked, or failed work without inventing completion.

The contract must not grant runtime, dispatch, Kanban, deployment, profile
mutation, or external-write authority.

## Explicit Exclusions

The replacement must contain none of the cumulative branch's unrelated work:

- model catalog or provider-default changes;
- Qwen 3.8 Flash changes already present on upstream `main`;
- Claude-to-Codex retuning;
- model-performance routing or per-model concurrency controls;
- egress, Kanban, PR-feedback, desktop, cron, gateway, or packaging changes;
- lifecycle integration from commits `9af3ec00`, `1c4afcd1`, or `7120646c`;
- Python runtime pinning; or
- generated artifacts unrelated to the two-file contract.

## Provenance and Integration Method

Create the replacement by cherry-picking the original contract commit onto the
verified upstream base. Preserve the original author and commit message. Resolve
only conflicts within the two-file allowlist. If the cherry-pick requires any
other tracked file, abort the integration and reassess instead of widening scope.

The later workforce-contract commits are not part of this replacement. They
overlap and duplicate lifecycle behavior, so any future integration requires a
separate semantic design and PR after the passive contract is reviewed.

## Verification

Verification is exact-head and must include:

1. clean tracked worktree state before integration;
2. two-file allowlist audit against upstream `main`;
3. the complete `tests/agent/test_worker_contract.py` module;
4. focused Ruff checks for both changed files;
5. Python bytecode compilation for both changed files;
6. a relevant adjacent agent test slice if current-main imports require it;
7. `git diff --check`; and
8. commit-author and diff-size verification.

Focused checks are not a full-suite receipt. Any unavailable or failing broader
lane remains explicit in the handoff.

## Publication and Disposition Boundaries

Local branch preparation and verification do not authorize a destination push,
new upstream PR, force-push, closure of `#98470`, branch deletion, or merge.
Immediately before publication, re-read upstream `main`, the original PR head,
and duplicate PR state. Publication should create a new cross-repository PR from
the user's fork with a body that names the exact base/head and supersedes only
the passive-contract claim. The user retains final authority over publishing
the replacement and disposing of `#98470`.
