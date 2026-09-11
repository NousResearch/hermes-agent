# Prior-Work-First

Use this reference when an implementation task should resume from prior
accepted work instead of rediscovering everything.

## Core Rule

Run startup in this order:

1. `GOAL`
2. `CAPABILITY_ID`
3. `PRIOR_WORK_LOCATE`
4. `ACCEPTED_BASELINE`
5. `FIRST_UNPROVEN_BOUNDARY`
6. `INVALIDATOR_CHECK`
7. load exact relevant Contract / SDD / source / test slices
8. `IMPLEMENT`
9. `VERIFY`
10. `DURABLE_CLOSEOUT`

## Authority Order

- RAG is locator/resume/historical context only.
- Contract is semantic authority.
- SDD is architectural authority only for impacted slices.
- Current source and tests are implementation truth.
- Fresh machine evidence is PASS/FAIL authority.

Old conversation and RAG never override current source, Contract, or fresh
machine evidence.

## Reuse vs Refresh

If accepted prior work exists, resume from `FIRST_UNPROVEN_BOUNDARY`.

If `INVALIDATOR_COUNT=0`, all of these must hold:

- `RESUME_FROM_FIRST_UNPROVEN_BOUNDARY=true`
- `FULL_DISCOVERY_FORBIDDEN=true`
- `FULL_PREFLIGHT_REEXECUTION_FORBIDDEN=true`
- `REPLAN_FORBIDDEN=true`

Refresh only volatile state by default:

- current HEAD/tree
- focused file, symbol, test, and dependency digests
- worktree/branch inventory
- fresh machine evidence

## Required Discovery

Before writing or replacing work, locate as applicable:

- capability or technical-debt ledgers
- project handoffs
- accepted receipts
- active preflight packets
- worktrees and branches
- draft or partial builds
- RAG resume / locate / impact artifacts
- relevant Contract slices
- impacted SDD slices
- current source and tests

This discovery is targeted. Do not blindly reload all prior history.

## Invalidators

Explicit invalidators should cover:

- relevant Contract change
- relevant symbol change
- dependency change
- accepted baseline invalidation
- test semantics change
- contradictory fresh machine evidence
- architecture owner change

Unrelated HEAD movement does not invalidate accepted work by itself.

## One Writer

One canonical primary writer only. Parallel scouts may locate evidence, but
they remain read-only.

## Executable Guard

Use the repository helper:

```bash
python scripts/prior_work_first.py locate --capability-id <CAPABILITY_ID>
python scripts/prior_work_first.py check --packet <packet.json>
python scripts/prior_work_first.py resume --packet <packet.json>
```
