# Hermes-loop protocol (upstream)

Shared invariants for all roles. Skills point here; do not duplicate.

## Purpose

Kanban-backed software factory: human freeze, scoped build units, SHA-tied review, humans merge. Designed for multi-hour and multi-day trains as linked short units.

## Mode requirement

While packets use triage freeze:

```yaml
kanban:
  auto_decompose: false
```

Otherwise the default decomposer can rewrite and fan-out triage cards before a human approves them.

## Packet (root card)

Starts in **`triage`**. Body must include:

- Goal (one verifiable sentence)
- Repo (`org/name` or absolute path) and base branch if not default
- Acceptance Criteria as stable `AC-N` observable outcomes
- Non-goals as binding `NG-N`
- Packet version `vN` (integer, starts at 1)
- Risk / authority notes (what agents may not do)
- How to verify (covers every AC)
- Factory role on worker cards: `spec-orchestrator` | `builder` | `reviewer`

Template: `templates/packet.md`

## Freeze

Human moves the **unchanged** packet from `triage` → `ready`.

- Do not run Specify or Decompose on the approved packet
- Spec-orchestrator never self-freezes
- Editing AC/NG/constraints/repo/risk after freeze: bump packet version and return to human freeze path

## Task graph (v1)

Do **not** walk a single card through the kernel `review` status. That status forces `sdlc-review` and keeps the current assignee.

Correct graph:

1. Root packet (orchestrator) owns the immutable contract
2. Ordinary **build** task (builder profile, prefer `worktree`)
3. Build completes with structured handoff + full git SHA
4. Ordinary **review** task (reviewer profile), separate card
5. Changes requested → new **build-fix** task → new **review** task

## Evidence contract

Canonical evidence is **kanban handoff text + git**, not forge labels.

Build handoff must include:

- Packet version
- Full commit SHA
- Changed files summary
- Commands run + results
- PR URL or branch name if any
- `Other behaviour changes: None` (or stop and amend packet)

Review verdict must include:

- Packet version reviewed
- Full commit SHA reviewed
- CI: passed | failed | pending | not configured
- Findings (must-fix / should-fix)
- Verdict: approve-evidence | changes-requested | needs-human-review
- Residual risk

Rules:

- New commit after a green verdict **invalidates** that verdict
- Human merge only when latest verdict SHA equals current head SHA
- Required CI missing or not configured → `needs-human-review`, never treat as approved evidence
- Optional GitHub labels (`loop-approved`, `loop-changes-requested`, `needs-human-review`) may project state; they never grant merge authority

## Agent hard bans

Agents never:

- merge or enable auto-merge
- deploy
- change credentials/permissions/security settings
- expand scope outside AC-N / against NG-N
- use kernel `review` status for this loop

## Recovery

- Heartbeat on long builder/reviewer runs
- Reclaim reuses the same worktree and prior attempt history when possible
- Gateway down pauses dispatch; board state remains
- Idempotent re-entry: do not open a duplicate PR for the same packet unit without cause

## UI (v1)

Hermes Agent CLI + **web dashboard** Kanban tab. Electron Desktop is out of scope for v1.
