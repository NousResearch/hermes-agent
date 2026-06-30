---
name: investigator
description: >
  Use when a task is underspecified and you want an agent to autonomously RESOLVE the unknowns before
  answering — not just list them. Calls the information-gain ranker for the next-best questions, then
  researches the top ones with a full Hermes agent (full agency by default — all tools), folds each
  distilled fact into one continuously-growing context, re-ranks, and repeats until it converges, then
  produces the final response. Records answered facts and known gaps as tombstones. Capability is full
  (act) by default; `--capability experiment|read` down-scopes for caution. Best where a clarification
  SHAPES the work (build/spec) or the answer is researchable. Triggers: "figure out what I'm missing
  and just do it", "investigate and answer", "resolve the unknowns then respond".
version: 0.1.0
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [investigator, clarifying-questions, value-of-information, grounded-research, autonomous, ollama]
    related_skills: [information-gain, ask]
    config:
    - key: investigator.k
      description: Top-K questions to research per round (by rank)
      default: 6
      prompt: How many questions should the investigator research per round?
    - key: investigator.max_rounds
      description: Max investigate-then-re-rank rounds before responding
      default: 3
      prompt: Max investigation rounds?
    - key: investigator.capability
      description: Default capability — act (full agency) | experiment (reversible) | read (read-only)
      default: act
      prompt: Default capability level for the investigator?
---

# Investigator — resolve the unknowns, then respond

## Overview

This is the **orchestrator** layer that sits on top of the report-only `information-gain` ranker. The
ranker decides *what is worth clarifying*; the Investigator goes and **answers it**, then responds.

The loop is **one continuously-growing, append-only context**:

```
tombstones = []                              # answered facts + known gaps
for round in range(max_rounds):
    evidence = facts(tombstones)             # the shared growing context
    ranked   = information-gain.run(problem, evidence)   # next-best questions, given everything known
    top      = [q for q in ranked if value >= floor and not answered][:K]    # top-K BY RANK
    if not top: stop "converged"
    for q in top: tombstones += grounded_answer(q)       # full Hermes agent, distilled fact back
final = respond(problem, evidence)           # best response over the enriched context
```

Each round conditions on the *entire* accumulated context, so the model's implicit posterior sharpens
as facts accrue — which is why we **always append** and keep tombstones clean, high-signal facts.

## How to run

```bash
# Inside the hermes container, FROM the user's project dir (so the answerer researches the real repo):
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>"
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --k 6 --max-rounds 3 --capability read
python3 ${HERMES_SKILL_DIR}/scripts/iterate.py --problem "<task>" --dry-run   # loop logic, no model calls
```

## Capability ladder

Default is **full agency** — it answers questions by any means (read, experiment, real action),
unattended. `--capability` only **down-scopes**:

| `--capability` | answerer tools | meaning |
|---|---|---|
| **act** (default) | file, web, terminal | full agency, unattended — current behavior |
| **experiment** | file, web, terminal | restricted by directive to **reversible** experiments (scratch/worktree) |
| **read** | file, web | **read-only** — inspect/search only; action-needing questions return NOT_FOUND |

Capability maps to the answerer's toolsets + a prompt directive (`CAPABILITIES` in `scripts/iterate.py`)
— it does not build a separate permission system. See `references/investigator.md`.

## Per-question outcomes (tombstones)

- **ANSWERED** (`Q → A`) — a discovered fact enters the context.
- **NOT_FOUND** (`Q → gap`) — recorded as a known gap; the final response proceeds with a stated
  assumption. (No revival machinery yet — v1.)
- **user-only** — a genuine preference no investigation can resolve → surfaced as a clarifying question.

## Status (v1, validated with caveats)

End-to-end value is **task-dependent** (de-confounded A/B: helps where a clarification shapes the work,
redundant where a capable agent self-investigates). The ranking it relies on is validated in the
agentic domain (realized-change ρ≈0.66). See `information-gain/references/evsi-validation-findings.md`.

## Dependency

Depends on the **information-gain** ranker (imported in-process, resolved via `HERMES_HOME` or
`INFOGAIN_SCRIPTS_DIR`) and the **ask** skill's `model_utils` (the grounded answerer/responder run a
full Hermes agent via `dispatch_single`; resolved via `HERMES_HOME` or `ASK_SCRIPTS_DIR`).

## Verification

- Loop logic (no network): `python3 tests/test_iterate.py` (13 tests).
- Live (in container): `python3 scripts/iterate.py --problem "<task>"` produces a final response;
  `--capability read` confirms down-scoping.
- End-to-end A/B harness: `evals/validate_wrapper.py` (baseline vs wrapper, blind-judged).
