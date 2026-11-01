---
name: hermes-loop
description: "Runs human-gated software work on Hermes kanban."
version: 1.0.0
author: Joel Brilliant (@joelbrilliant), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, workflow, factory, review, build, long-running, software-factory]
    related_skills: [hermes-agent]
---

# Hermes-loop Skill

Turn Hermes kanban into a software factory for multi-hour and multi-day
work trains with human freeze and SHA-tied review. It does not merge code,
replace human approval, or use the kernel review column.

## When to Use

- You want a queue that survives crashes and restarts (kanban + gateway dispatcher)
- Work is larger than one chat turn but can be sliced into day-or-less units
- You need human freeze before agents spend build tokens
- You need SHA-tied review evidence before a human merges

## Prerequisites

1. Gateway running if you want automatic dispatch of `ready` tasks
2. **`kanban.auto_decompose: false`** while using triage freeze (default true will fan-out packets before human approval)
3. Profiles for builder and reviewer exist (names are yours; pin skills on each)
4. Optional: a dedicated board (`hermes kanban boards create ...`)

```bash
hermes config set kanban.auto_decompose false
hermes gateway status   # or start
```

## How to Run

1. Read `references/protocol.md` for the shared invariants.
2. Load the reference for your current role.
3. Use the matching template for the packet, build handoff, or review verdict.
4. Keep every worker unit to one day or less and link it to the root packet.

## Quick Reference

| Role | File |
|------|------|
| Invariants | `references/protocol.md` |
| Spec-orchestrator | `references/spec-orchestrate.md` |
| Builder | `references/build.md` |
| Reviewer | `references/review.md` |

| Artifact | Template |
|----------|----------|
| Frozen packet | `templates/packet.md` |
| Build evidence | `templates/build-handoff.md` |
| Review verdict | `templates/review-verdict.md` |

Roles are nouns, not people:

- **spec-orchestrator** researches, files the packet, owns the root card, and creates build and review tasks
- **builder** implements one unit inside the packet
- **reviewer** adversarially reviews one handoff at one git SHA
- **human** freezes packets into `ready` and merges code

## Procedure

1. Spec-orchestrator files the root packet in **`triage`** with full AC,
   NG, packet version, and repository using `templates/packet.md`.
2. **Human freezes** the unchanged packet with a direct status move from
   `triage` to `ready`. Do **not** run Specify or Decompose on the approved
   packet.
3. Spec-orchestrator or dispatcher creates an ordinary **build** child task
   in a worktree, not a kernel status shortcut.
4. Builder implements only AC-N, preserves NG-N, and returns
   `templates/build-handoff.md` with the **full git SHA**.
5. Spec-orchestrator creates a **separate ordinary review task** with the
   reviewer profile. **Never use the kernel `review` column** for
   Hermes-loop v1.
6. Reviewer returns `templates/review-verdict.md` tied to that full SHA.
   Fixer mode stays off, with no push and no merge.
7. If changes are requested, create a new build-fix task followed by a new
   review task.
8. **Human merges** only when the latest verdict SHA equals the PR head SHA
   and CI policy is satisfied. Agents never merge.

## Pitfalls

- No agent merge, auto-merge, deploy, credential changes, or destructive ops outside the packet
- No kernel kanban `review` status for this loop
- No Specify/Decompose on a frozen/approved packet
- Missing required CI → human review path, never auto-approve
- Forge labels (if used) are optional projections, never authority
- Pre-freeze packets stay in **`triage`** (parent-free `todo` auto-readies)
- Do not copy shared invariants into each role file. Keep them in
  `references/protocol.md`.
- A multi-day effort is a graph of one-day-or-less worker units, not one
  immortal chat session. Heartbeats and reclaim keep workers honest, while
  the board remains the memory.

## Verification

- Confirm `kanban.auto_decompose` is `false` before moving a packet to
  `ready`.
- Confirm the frozen packet version is unchanged before creating build
  tasks.
- Confirm every build handoff and review verdict names the full commit SHA.
- Confirm the latest verdict SHA equals the PR head SHA.
- Confirm required CI passes before the human merge path.
