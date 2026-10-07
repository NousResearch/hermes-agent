---
project: Hermes
repo_id: hermes-agent
goal: Pilot deterministic cross-agent continuity in Hermes
state: ACTIVE
active_task: hermes-continuity-b6l
change_id: 001-global-continuity-pilot
folder: /Volumes/Storage Unit/Application-Data/Codex/worktrees/hermes-continuity-pilot-20260824
supersedes: none
---

# Deterministic cross-agent continuity pilot

**Feature Branch**: `pilot/global-continuity-hermes-20260824`

**Created**: 2026-08-24

**Status**: Active pilot

## User scenarios and testing

### Resume in a fresh agent session (P1)

A Claude, Codex, or Hermes session starting in the exact pilot folder receives one
bounded preflight summary that identifies the goal, active task, active change, exact
repository state, next action, and blockers without replaying transcripts.

**Independent test**: Run the dispatcher twice from clean processes with the same
fixtures and verify equivalent authority and exact-state fields, output under 8,000
characters, and no prompt asking for already-recorded scope.

### Refuse stale completion (P1)

An agent cannot promote the work to `TESTED` or `ENFORCED` when the card, Beads task,
Spec Kit change, git SHA, integration base, dirty state, external input digests, or
gate-executed command outcomes disagree.

**Independent test**: Change each authority and exact-state field independently and
verify the gate refuses promotion with a named reason.

### Recover without lock-in (P2)

A human can read the card, JSONL task record, Markdown specification, and JSON receipt
without any of the three tools, and can remove the project adapters without damaging
Hermes or the shared checkout.

**Independent test**: Disable each binary, inspect the source files directly, run the
documented rollback, and verify Hermes has no new core memory provider or live global
hook dependency.

## Functional requirements

- **FR-001**: Preflight MUST identify repository root, branch, commit, and dirty state.
- **FR-002**: Preflight MUST read the exact bounded Basic Memory card, active Beads
  task, and matching Spec Kit change and verify their project, goal, folder, task, and
  change identifiers agree.
- **FR-003**: A Beads timeout MAY use the authoritative external JSONL as a degraded
  readable fallback, but MUST set `completion_allowed` to false.
- **FR-004**: Missing drive, wrong folder, stale task, absent/corrupt card, incomplete
  spec, or authority conflict MUST forbid completion and identify the cause.
- **FR-005**: Tool-call envelopes MUST reject interrupted adjacency. A single tool use
  or contiguous tool-use batch MUST be followed immediately by its matching result or
  ordered result batch, with no unrelated event between the two batches.
- **FR-006**: Only the gate MAY promote the lifecycle to `TESTED` or `ENFORCED`; it
  MUST execute required commands itself, authenticate the receipt, derive full-suite
  identity from committed policy, serialize promotion, and preserve a durable
  recoverable journal across every interrupted multi-authority transition stage.
- **FR-007**: Host adapters MUST call one dispatcher and MUST NOT persist raw prompt,
  transcript, reasoning, tool input, tool output, secret, or customer-data content.
- **FR-008**: All storage-dependent state MUST remain on the mounted external volume;
  live global host configuration MUST remain untouched during the pilot.
- **FR-009**: Preflight output MUST be at most 8,000 characters (approximately 2,000
  tokens), deterministic, and readable without a model.
- **FR-010**: Rollback MUST remove only pilot adapters/state and leave the dirty shared
  checkout and pre-existing global hooks unchanged.
- **FR-011**: Authority-bearing children MUST run without inherited credential
  variables; receipts MUST NOT persist raw argv/output; and external executables and
  instructions MUST match committed SHA-256 pins.
- **FR-012**: Claude/Codex finalization and Hermes tool execution MUST use each host's
  native blocking contract. Hermes MUST persist a content-free per-session, per-turn
  adjacency decision at `pre_llm_call` and enforce it fail-closed at every
  `pre_tool_call` in that turn. Hermes
  ordinary prose remains outside shell-hook blocking; the gate MUST still prevent
  lifecycle promotion from advisory context alone.
- **FR-013**: Receipt command, runtime, rollback, timeout, surface, risk-tier, and
  interpreter identities MUST come from committed policy at a clean branch based on
  the configured integration SHA; caller-supplied manifests MUST match that policy.

## Success criteria

- Fresh Claude, Codex, and sandboxed Hermes sessions recover the same active scope.
- Every listed failure drill forbids completion with a specific diagnostic.
- A changed SHA or dirty state invalidates an earlier receipt.
- Healthy preflight completes within its configured bounds and emits <=8,000 chars.
- Human-readable card, task JSONL, spec, and receipt remain independently inspectable.
- Rollback is dry-run verified and requires no subscription or network service.

## Assumptions and boundaries

- This is a T3 pilot because it touches lifecycle hooks and external storage, even
  though the repository changes are adapters and scripts.
- Basic Memory live Hermes capture is out of scope; the pilot writes one explicit card.
- Beads remains the task authority, not the completion authority.
- Native Hermes skills/configuration are proven in an isolated `HERMES_HOME` only.
- Hermes shell hooks can block `pre_tool_call` but cannot veto ordinary final prose or
  `on_session_end`; this pilot does not claim otherwise.
