---
name: engineering-memory
description: "Review and search shared engineering evidence."
version: 1.0.0
author: Mike DeMott (@mrkillbob), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [engineering, memory, provenance, review, diagnostics]
    category: software-development
    related_skills: [codebase-inspection, systematic-debugging]
---

# Engineering Memory Skill

Use the local engineering-memory ledger to share reviewed engineering evidence between Hermes, Claude, and Codex. It stores diagnostic knowledge with repository, revision, evidence, and review metadata; it does not replace profile-local memory, authorize actions, or mutate the stable system prompt.

## When to Use

- Before repairing a known repository, component, runner, or environment failure.
- After verifying a diagnosis or repair with a test, receipt, or reproducible command.
- When another client needs bounded, provenance-preserving engineering context.

## Prerequisites

- An explicit shared vault and SQLite index path configured by the operator.
- Access to the native `terminal` tool for `hermes engineering-memory` commands.
- Use `read_file` to inspect a candidate record and `search_files` to locate its evidence.

## How to Run

The feature is disabled until both paths are explicit:

`hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 verify --json`

Create a structured Markdown candidate inside the configured vault, then stage it:

`hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 propose /shared/engineering-memory/candidate.md --json`

A human reviews the proposal:

`hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 review RECORD_ID approve --reviewer HUMAN --reason "Receipt and current head checked" --json`

Rebuild and search only approved records:

`hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 rebuild --json`

`hermes engineering-memory --vault /shared/engineering-memory --index /shared/engineering-memory/index.sqlite3 search "live runner timeout" --repository NousResearch/hermes-agent --json`

## Quick Reference

| Operation | Meaning |
| --- | --- |
| `propose` | Validate, scan, and stage a candidate |
| `review` | Apply explicit human approval, rejection, supersession, or withdrawal |
| `rebuild` | Materialize approved Markdown records into SQLite |
| `search` | Return bounded approved evidence with provenance |
| `verify` | Report ledger and index health |

The JSON contract is `hermes.engineering_memory.v1`. Results retain `record_id`, status, source label, verified head, related records, conflicts, and a reason code. Claude and Codex use this same CLI/JSON contract and do not write directly to the approved index.

## Procedure

1. Use `read_file` to inspect the source record and confirm it contains only cross-engineering knowledge.
2. Confirm repository, workspace, branch/ref, verified head, observed/verified timestamps, and evidence references.
3. Run the candidate through `propose`; quarantine or review diagnostics are expected safety states.
4. Have a human compare the evidence before `review approve`.
5. Run `rebuild`, then query with repository/component/task/head filters.
6. Feed retrieved records to engineering reasoning as diagnostic evidence. Keep execution authority in the current repository/runtime checks.
7. When a newer record disagrees, preserve the conflict set and document explicit supersession evidence; never resolve by recency alone.

## Pitfalls

- Never put secrets, credentials, private preferences, raw transcripts, or executable instructions in a record.
- Never treat a candidate, quarantined record, LAYA probability, or model-generated suggestion as approved truth.
- Never point the shared vault or index at profile-local `MEMORY.md`, `USER.md`, or session state.
- Never use this index to authorize trading, routing, broker, deployment, or submit actions.
- Hindsight and Neural Steering are not part of this workflow.
- LAYA, when installed, is an optional local advisory sidecar. Missing, uncalibrated, or low-confidence LAYA results fall back to deterministic curator rules.
- The NVIDIA AI-Q research skill may use recalled records as diagnostic evidence, but the index cannot control signals or runtime execution.

## Verification

Run `verify` after setup and after every rebuild. Inspect the JSON diagnostics, then use `read_file` for the cited evidence and `search_files` for the exact test or receipt. The canonical Hermes implementation checks run through `scripts/run_tests.sh`; a local test pass is not evidence that a repair is active in another checkout or live runner.
