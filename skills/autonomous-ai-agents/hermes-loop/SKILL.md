---
name: hermes-loop
description: "Kanban software factory loop: human freeze, scoped build, SHA-tied review, humans merge. Multi-hour/day trains as linked one-day units. Use for factory-spec, factory-build, factory-review workflows on Hermes kanban."
version: 1.0.0
author: Hermes Agent contributors
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, workflow, factory, review, build, long-running, software-factory]
    related_skills: [hermes-agent]
---

# Hermes-loop

Turn Hermes kanban into a software factory for multi-hour and multi-day work trains.

**Roles (nouns, not people):**
- **spec-orchestrator** — research, interview, file the packet, own the root card, spawn build/review tasks
- **builder** — implement one unit inside the packet
- **reviewer** — adversarial review of one handoff at one git SHA
- **human** — freeze packets into `ready`, merge code

Load the matching procedure under `references/` for the role you are playing. Shared invariants live in `references/protocol.md` only — do not fork them into three copies.

## When to use

- You want a queue that survives crashes and restarts (kanban + gateway dispatcher)
- Work is larger than one chat turn but can be sliced into day-or-less units
- You need human freeze before agents spend build tokens
- You need SHA-tied review evidence before a human merges

## Prerequisites (checked)

1. Gateway running if you want automatic dispatch of `ready` tasks
2. **`kanban.auto_decompose: false`** while using triage freeze (default true will fan-out packets before human approval)
3. Profiles for builder and reviewer exist (names are yours; pin skills on each)
4. Optional: a dedicated board (`hermes kanban boards create ...`)

```bash
hermes config set kanban.auto_decompose false
hermes gateway status   # or start
```

## Quick cycle

1. Spec-orchestrator files root packet in **`triage`** with full AC/NG/packet version/repo (template: `templates/packet.md`).
2. **Human freezes** the unchanged packet: direct status move `triage` → `ready` (dashboard drag, or API status write). Do **not** run Specify or Decompose on the approved packet.
3. Spec-orchestrator (or dispatcher) creates an ordinary **build** child task (worktree), not kernel status tricks.
4. Builder implements only AC-N, preserves NG-N, returns `templates/build-handoff.md` with **full git SHA**.
5. Spec-orchestrator creates a **separate ordinary review task** (reviewer profile). **Never use the kernel `review` column** for Hermes-loop v1.
6. Reviewer returns `templates/review-verdict.md` tied to that full SHA. Fixer mode off — no push, no merge.
7. Changes requested → new build-fix task → new review task.
8. **Human merges** only when latest verdict full SHA equals PR head SHA (and CI policy is satisfied). Agents never merge.

## Procedures

| Role | File |
|------|------|
| Invariants | `references/protocol.md` |
| Spec-orchestrator | `references/spec-orchestrate.md` |
| Builder | `references/build.md` |
| Reviewer | `references/review.md` |

## Hard limits

- No agent merge, auto-merge, deploy, credential changes, or destructive ops outside the packet
- No kernel kanban `review` status for this loop
- No Specify/Decompose on a frozen/approved packet
- Missing required CI → human review path, never auto-approve
- Forge labels (if used) are optional projections, never authority
- Pre-freeze packets stay in **`triage`** (parent-free `todo` auto-readies)

## Long trains

A multi-day effort is a **graph of one-day-or-less worker units**, not one immortal chat session. Heartbeats and reclaim keep workers honest; the board is the memory.
