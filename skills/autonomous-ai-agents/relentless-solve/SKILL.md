---
name: relentless-solve
description: >
  Use when a prompt should be driven to a solution without letting go: it clarifies the prompt
  first (information-gain ranks the next-best questions by EVSI; the investigator researches
  them), executes with the resilient-planner (AND/OR backtracking to a terminal state), and when
  execution fails it harvests the failure conditions — dead branches, exhaustion, budget guards —
  as evidence folded into the next clarify round, looping until success or the search is provably
  information-dry. Deterministic outer loop (no LLM decides control flow), durable and resumable
  (resumable-script flow). Triggers: "relentlessly solve this", "keep going until it works",
  "clarify then execute and learn from failures".
version: 0.1.0
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [orchestrator, clarify-execute-loop, value-of-information, backtracking, durable, autonomous]
    related_skills: [investigator, information-gain, resilient-planner, resumable-script]
---

# Relentless Solve — clarify → execute → harvest failures → repeat

## Overview

One deterministic loop over three existing layers. Each cycle:

1. **CLARIFY** — `investigator/scripts/iterate.py` (in-process): information-gain ranks the
   next-best questions given *everything known so far*; the top-K are researched by a full
   Hermes agent; answers and gaps come back as tombstones.
2. **EXECUTE** — `resilient-planner` via `drive.py` (subprocess): the prompt, re-rendered with
   the evidence ledger, runs as a fresh planner slug `<slug>-c<N>` to a terminal STATE.
3. **HARVEST** — `scripts/harvest.py` (pure parser): ✝ dead-set reasons, exhaustion, and
   guard-halt notes become ledger records seeded into the next clarify round, where the EVSI
   gate naturally retires resolved questions and surfaces the next-best unknowns.

The **prompt is immutable** (intent); the **ledger only grows** (facts / gaps / dead-ends).
Stop conditions, in order of honor: planner `SUCCESS` · **information-dry** (a full cycle
yields zero fresh facts — clarify converged AND every harvested failure was already known;
the anti-flap guard that separates relentless from flailing) · `max_cycles` · outer wallclock.

GUARD-HALT forks default to **assume-and-note**: the open fork is recorded as a gap and the
next clarify round ranks it. `--gate` opts into suspending instead (engine exit 10; answer
with `resume --answer`).

## How to run (inside the hermes container)

```bash
docker exec -u 10000 hermes python3 \
  /opt/data/hermes-agent/skills/autonomous-ai-agents/relentless-solve/scripts/relentless.py run \
  --slug my-task --answer-cwd /path/to/target/project \
  --prompt 'the intent, stated once' \
  [--max-cycles 5] [--wallclock 14400] [--k 6] [--inv-rounds 3] [--capability act] [--gate]
# after a --gate suspension:
... relentless.py resume --slug my-task --answer 'prefer source D'
```

`--answer-cwd` pins where the clarify answerer researches — always set it to the target
project (the known failure mode is researching the install dir). Host-side runs are
tests-only (`tests/`, fakes; iterate.py needs the container's model_utils live).

## Building-block resolution (env overrides)

| Dependency | Default | Override |
|---|---|---|
| information-gain ranker | sibling `../information-gain/scripts` (pinned before importing iterate) | `INFOGAIN_SCRIPTS_DIR` |
| ask model dispatch | `${HERMES_HOME}/skills/productivity/ask/scripts` | `ASK_SCRIPTS_DIR` |
| resilient-planner driver | `${HERMES_HOME}/skills/resilient-planner/scripts/drive.py` | `RESILIENT_DRIVE` |
| resumable-script engine | `${HERMES_HOME}/skills/resumable-script/scripts` | `RESUMABLE_ENGINE_DIR` |

The engine must be deployed at `${HERMES_HOME}/skills/resumable-script/` (sync it there if
only present in a staging tree).

## State layout

```
${HERMES_HOME}/relentless/<slug>/
  flow/           # resumable-script engine state (journal.jsonl, state.json, blobs/, lock)
  prompt-c<N>.md  # the rendered prompt each cycle actually executed
  ledger.jsonl    # human-readable evidence ledger snapshot (flow journal is the durable truth)
  report.md       # final outcome + ledger by kind
${HERMES_HOME}/plans/<slug>-c<N>/   # one resilient-planner tree per cycle (read once by harvest)
```

## Exit codes / result

Exit codes are the resumable-script engine's: `0` completed — read `result.outcome` from the
final stdout JSON (`success` | `information-dry` | `max-cycles` | `wallclock`; exit 0 covers
all four) · `10` suspended on a `--gate` fork · `1/2/3` failed/usage/skew. A crash or kill is
resumable: re-run the same `run` command and completed steps replay from the journal (after
editing relentless.py mid-run, add `--accept-flow-change`).

## Design notes

- Dead-end fingerprints key on the **method label**, not the reason — a method dying twice
  with fresh wording is the flap the information-dry guard exists to catch. An early dry stop
  therefore means "no new methods and no new facts", not "no new error text".
- No `--bump-guard` re-drives: a guard bump re-runs the same tree with the same knowledge.
  Recycling converts the halt into evidence and lets EVSI decide what the frontier is worth.
- Full design + locked decisions: `src/hermes/skills/relentless-solve/DESIGN.md` (staging).
