---
name: mind-palace
description: >-
  Dev-il's Advocate prestige prompt structure system for project navigation.
  Multi-stage directions: prestige → entry → state → plan → testing → project → index.
  4-phase dev-il's advocate review loop: Claim → Verify → Risk → Mitigate.
  Slate markdown format for consistent project documentation.
  Create this in a .hermes/ directory for any project needing structured oversight.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
tags: [workflow, project-management, dev-ils-advocate, quality-assurance, documentation]
aliases: [mp]
---

# Mind Palace — Dev-il's Advocate Project Navigation System

## Overview

A **mind palace** is a `.hermes/` directory at a project's root that contains all the context an agent needs to work on that project. It uses a **walkway** of files read in sequence, a **4-phase dev-il's advocate loop** to catch bad claims, and a **slate** markdown format for consistent documentation.

The design goal: a new agent should be able to read the walkway and understand the full project state, past decisions, pitfalls, and next actions without reading git history or conversation transcripts.

## Walkway (Read Order)

Seven files read in sequence, each building on the last:

```
prestige.md → entry.md → state.md → plan.md → testing.md → project.md → index.md
```

| File | Purpose | Questions It Answers |
|------|---------|---------------------|
| `prestige.md` | Full context resume, one-liner mission, architecture stack, priority queue | "What's the project? Where are we? What's next?" |
| `entry.md` | HW spec, build/launch/kill commands, architecture diagrams | "How do I run this?" |
| `state.md` | LIVE status dashboard, benchmarks, verification signals | "Is it working? What's the evidence?" |
| `plan.md` | Phase roadmap, milestone targets, dependency graph | "Where are we going and how do we get there?" |
| `testing.md` | THE protocol, exact steps, pass/fail criteria | "How do I know if it's working?" |
| `project.md` | Mission statement, done/pending list, constraints | "What's the big picture?" |
| `index.md` | Full navigation hub of all .hermes files | "Where is everything?" |

### Additional Convention: Overnight Map

For autonomous sessions that run without supervision, create `overnight-map.md` in `.hermes/mind-palace/`. Structure:

```
# Project — Overnight Navigation Map

## Quick Trunk Reference
[table of key paths]

## Where We Are
[phase status with VERIFIED vs CLAIMED-UNVERIFIED sections]

## Workstreams (pick one)
A — [highest impact]
B — [next]

## Data You Should Not Re-Derive
[brief fact list; don't over-explain]

## Fallback
If blocked, write a plan.md
```

## 4-Phase Dev-il's Advocate Loop

For every claimed result, benchmark, or milestone, run this loop BEFORE accepting it:

### Phase 1: Claim
What is being claimed? Record it verbatim.

```
Claim: [exact statement]
Source: [session log, test binary, calculation]
Trust: [high/medium/low]
```

### Phase 2: Verify
Test the claim against reality. Never skip this step.

- **Code exists?** `ls -la` the file, `wc -l` for content depth
- **Binary exists?** `file <binary>` to confirm executable
- **Was it actually run?** Check for stdout/stderr artifacts
- **Is the math right?** Trace formulas with actual values
- **Grep the source** before trusting docs — `grep -rn "feature" src/`

### Phase 3: Risk Assessment

| Risk Type | Question |
|-----------|----------|
| **Confirmation bias** | "Are we looking for evidence that supports our approach?" |
| **Survivorship bias** | "Did we only test on favorable cases?" |
| **Measurement error** | "Is the benchmark measuring what we think?" |
| **Scope creep** | "Does this claim depend on other unfinished work?" |
| **State staleness** | "When was this last verified? What changed since?" |

### Phase 4: Mitigate

```
Mitigation: [what to do if this risk materializes]
Monitoring: [what signal tells us this is happening]
Revert threshold: [at what point do we undo this change]
```

## Slate Markdown Format

All walkway files follow a consistent structure:

```
# [File Title]

## Purpose
[30-word max. One sentence.]

## Key Decision
[One decision that shaped everything.]

## [Tables / Priority Queue]
[Structured data in | key | value | format.]

## [Content Section]
[Keep it terse. Bullets > paragraphs for walkway.]
```

### File Naming Convention

- All lowercase with hyphens
- Date-stamped plans: `plan-2026-05-12.md`
- `overnight-map.md` for autonomous session nav
- `fresh_start_prompt.md` for boot sequence

### Directory Structure

```
.hermes/
├── mind-palace/           ← Walkway files live here
│   ├── plans/              ← Implementation plans (date-stamped)
│   ├── references/         ← Paper summaries, deep dives, HOWTOs
│   └── presentation/       ← Curated nav panel (public-facing)
├── research/               ← Paper downloads and notes
├── references/             ← Project-specific reference docs
└── vault/                  ← Index of all research/code artifacts
```

## 1000-Step Dev-il's Advocate Review Process

For autonomous long-running sessions:

| Steps | Phase | Action |
|-------|-------|--------|
| 1-100 | Survey | Read all walkway files. Map the claim landscape. |
| 101-300 | Execute | Pick highest-priority undone step. Execute it. |
| 301-400 | Verify | Run the test. Check output. Record the claim. |
| 401-500 | Dev-il's Advocate | Run 4-phase loop on the result. |
| 501-700 | Adjust | Fix issues found. Re-test. |
| 701-900 | Document | Update walkway files. Note new risks. |
| 901-1000 | Plan Next | Write extension/next-phase roadmap. |

## Extension Phase Plane

After a phase completes, document the extension to the next:

1. **What changed** — new capabilities, new metrics
2. **What's stale** — walkway files needing update, moot risks
3. **What's next** — next phase walkway (can reuse structure)
4. **What to watch** — new risks from this phase's work

## Implementation: Creating a Mind Palace

```bash
# Create the structure in any project root:
mkdir -p .hermes/mind-palace/{plans,references,presentation} \
         .hermes/research .hermes/vault
touch .hermes/mind-palace/{prestige,entry,state,plan,testing,project,index}.md
```

Fill each file following the slate format above. Then create an `overnight-map.md` for autonomous sessions and a `fresh_start_prompt.md` with the boot sequence.

## Pitfalls

1. **Don't let the walkway become stale.** Every session that changes something should update at least one walkway file. If the last-updated timestamp is more than 3 sessions old, the walkway is probably wrong.
2. **State files rot faster than plan files.** A plan's vision changes slowly; a state's benchmark numbers change every session. Update state.md aggressively.
3. **The devil's advocate step is the most skipped.** When things are working, it's tempting to skip verification. This is exactly when verification is most important — working code often means a coincidental pass on limited test data.
4. **Don't put task progress in memory.** Session-to-do items belong in plan.md and the walkway, not in Hermes Agent memory. Memory is for durable facts that survive project completion.
5. **Overnight maps need a fallback.** If a long session hits blockers on all workstreams, the overnight map must include "write a plan.md" so the session isn't wasted.
6. **Alias `mp` is registered for this skill.** Load with `skill_view(name='mind-palace')` or via the `mp` alias.
