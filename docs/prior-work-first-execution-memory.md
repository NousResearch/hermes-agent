# Prior-Work-First Execution Memory

This document is the repository-level protocol for durable task continuity.
It gives a fresh agent enough structure to resume accepted work from repo
artifacts instead of repeating broad discovery.

## Scope

Use this protocol for repository-authorized implementation tasks where an
agent may resume in a fresh session with no chat history.

Do not use old conversation logs as implementation authority. They are
historical locator signals only.

## Canonical Owners

- `SOUL.md` holds the concise law: `PRIOR-WORK-FIRST`.
- `AGENTS.md` holds the durable repository rules future agents must follow.
- This document holds the detailed execution protocol and packet contract.
- `scripts/prior_work_first.py` is the machine-checkable guard and resume tool.
- Current source and tests remain implementation truth.
- Fresh machine evidence remains pass/fail truth.

## Authority Ranking

Authority depends on the question being answered.

- Locator / resume / historical context:
  `preflight packet` -> `receipt` -> `handoff` -> `RAG` -> old conversation
- Semantic contract:
  `Contract` -> current source -> current tests -> impacted SDD slice -> RAG
- Implementation truth:
  current source -> current tests -> Contract -> impacted SDD slice -> RAG
- Verification truth:
  fresh machine evidence -> current tests -> current source -> Contract -> RAG

Old conversation and RAG never override current source, Contract, or fresh
machine evidence.

## Startup Flow

Every fresh execution should follow this order:

1. `GOAL`
2. `CAPABILITY_ID`
3. `PRIOR_WORK_LOCATE`
4. `ACCEPTED_BASELINE`
5. `FIRST_UNPROVEN_BOUNDARY`
6. `INVALIDATOR_CHECK`
7. load the exact relevant Contract / SDD / source / test slices
8. `IMPLEMENT`
9. `VERIFY`
10. `DURABLE_CLOSEOUT`

The first accepted boundary is the resume anchor. If prior work is accepted,
resume from `FIRST_UNPROVEN_BOUNDARY` unless an explicit invalidator proves a
targeted refresh is required.

## Canonical Artifact Layout

Store repository-task continuity artifacts under:

```text
.task-state/prior-work-first/<capability-slug>/
  active.packet.json
  handoffs/
  receipts/
  drafts/
  rag/
  ledgers/
```

The protocol is compact by design. Do not store transcripts in the active
packet.

## Compact Preflight Packet Contract

The durable packet must retain only compact state needed to resume:

```json
{
  "PROGRAM_ID": "HERMES-PRIOR-WORK-FIRST-EXECUTION-MEMORY-V1",
  "GOAL": "what this task is trying to finish",
  "CAPABILITY_ID": "stable capability identifier",
  "PREFLIGHT_STATUS": "accepted|accepted_reuse|verified|targeted_refresh_required|global_refresh_required",
  "ACCEPTED_HEAD": "git head accepted by prior work",
  "ACCEPTED_TREE": "git tree accepted by prior work",
  "FIRST_UNPROVEN_BOUNDARY": "the earliest boundary not yet proven",
  "KNOWN_FILES": [{"path": "relative/path", "digest": "sha256"}],
  "KNOWN_SYMBOLS": [{"path": "file.py", "symbol": "qual.name", "digest": "sha256"}],
  "KNOWN_TESTS": [{"path": "tests/test_x.py", "digest": "sha256"}],
  "KNOWN_DEPENDENCIES": [{"name": "pytest", "path": "pyproject.toml", "digest": "sha256"}],
  "KNOWN_HANDOFFS": [{"path": ".task-state/.../handoffs/h1.md", "digest": "sha256"}],
  "KNOWN_RECEIPTS": [{"path": ".task-state/.../receipts/r1.json", "digest": "sha256"}],
  "KNOWN_WORKTREES": [{"path": "...", "branch": "feat/x", "head": "sha"}],
  "KNOWN_BRANCHES": [{"branch": "feat/x", "head": "sha", "upstream": "origin/main"}],
  "KNOWN_DRAFT_BUILDS": [{"path": ".task-state/.../drafts/build.txt", "digest": "sha256"}],
  "KNOWN_BASELINE_EXCEPTIONS": ["narrow accepted exceptions only"],
  "RELEVANT_CONTRACT_SLICES": [{"path": "AGENTS.md", "digest": "sha256"}],
  "RELEVANT_SDD_SLICES": [{"path": "docs/design/example.md", "digest": "sha256"}],
  "PROVEN_INVARIANTS": ["exact invariant already proven"],
  "INVALIDATE_IF": [
    {"kind": "contract_change", "path": "AGENTS.md", "expected_digest": "sha256"},
    {"kind": "symbol_change", "path": "agent/example.py", "symbol": "foo", "expected_digest": "sha256"},
    {"kind": "dependency_change", "path": "pyproject.toml", "expected_digest": "sha256"},
    {"kind": "test_semantics_change", "path": "tests/test_x.py", "expected_digest": "sha256"},
    {"kind": "contradictory_machine_evidence", "test": "tests/test_x.py"}
  ],
  "LAST_MACHINE_VERIFIED_AT": "2026-08-30T00:00:00Z",
  "PRIMARY_WRITER": "canonical agent id",
  "PARALLEL_SCOUTS": [{"name": "scout-a", "mode": "read_only", "read_only": true}]
}
```

The packet is intentionally compact:

- allowed: program, capability, goal, accepted baseline, first unproven
  boundary, exact files/symbols/tests/dependencies, focused receipts/handoffs,
  explicit invalidators, precise worktree/branch references, proven invariants
- forbidden: full transcripts, broad conversation dumps, blind history copies,
  speculative TODO lists detached from a proven boundary

## Stale vs Refreshed Data

Default to stale reuse only for data already accepted and still uninvalidated:

- accepted baseline head/tree
- exact file, symbol, contract, SDD, dependency, and test digests
- accepted handoffs and receipts
- the first unproven boundary

Refresh volatile state first:

- current HEAD and tree
- worktree inventory
- branch inventory
- current digests for exact tracked files/symbols/tests
- fresh machine evidence

Do not broaden discovery when only volatile state moved and no invalidator
fired.

## Invalidator Policy

Explicit invalidators should cover:

- relevant Contract change
- relevant symbol change
- dependency change
- accepted baseline invalidation
- test semantics change
- contradictory fresh machine evidence
- architecture owner change

Unrelated HEAD changes do not invalidate accepted work by themselves.

## Full Rediscovery vs Targeted Refresh

If `INVALIDATOR_COUNT=0`, all of these must hold:

- `RESUME_FROM_FIRST_UNPROVEN_BOUNDARY=true`
- `FULL_DISCOVERY_FORBIDDEN=true`
- `FULL_PREFLIGHT_REEXECUTION_FORBIDDEN=true`
- `REPLAN_FORBIDDEN=true`
- `REDUNDANT_PREFLIGHT_ATTEMPT=true` if a broad rediscovery was attempted

If a targeted invalidator fires, refresh only the affected slice. Full
rediscovery becomes allowed only when a global invalidator proves the accepted
baseline itself is no longer trustworthy.

## Discovery Requirements Before New Work

Before creating a new implementation, plan, or worktree, locate as applicable:

- capability or technical-debt ledgers
- project-context rules and handoffs
- accepted receipts
- active preflight packets
- active and historical worktrees
- active and historical branches
- draft or partial builds
- RAG resume / locate / impact artifacts
- relevant Contract slices
- impacted SDD slices
- current source and focused tests
- fresh machine evidence

This discovery stays targeted. Do not blindly load all historical context.

## One Writer Invariant

There may be only one canonical primary writer for a capability at a time.
Parallel scouts are allowed only when they are read-only. A read-only scout may
locate or summarize evidence; it may not become a second writer.

## Durable Closeout

At the end of a task, refresh the compact packet and any accepted receipt with
fresh machine evidence from the current state. A prior PASS is not reopened
without an explicit invalidator.

## Executable Guard

Run the targeted locator:

```bash
python scripts/prior_work_first.py locate --capability-id <CAPABILITY_ID>
```

Check whether the accepted packet can be reused:

```bash
python scripts/prior_work_first.py check \
  --packet .task-state/prior-work-first/<capability-slug>/active.packet.json
```

Reconstruct the startup flow in machine-readable form:

```bash
python scripts/prior_work_first.py resume \
  --packet .task-state/prior-work-first/<capability-slug>/active.packet.json
```

Normalize a packet into the compact durable shape:

```bash
python scripts/prior_work_first.py compact-packet --packet <path>
```
