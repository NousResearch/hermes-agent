---
name: darwinian-evolver
description: >-
  Evolutionary optimizer for prompts, regexes, SQL, and small code snippets.
  Uses LLM-driven mutation and crossover over a MAP-Elites / (μ+λ)-ES /
  NSGA-II core, with a principled fitness contract, budget enforcement, and
  DSPy-compatible trace export that bridges into
  NousResearch/hermes-agent-self-evolution.
version: 0.1.0
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags:
      - evolutionary
      - optimization
      - prompt-engineering
      - self-improvement
      - research
      - map-elites
      - nsga2
      - genetic-algorithm
      - dspy
      - gepa
    category: research
    related_skills:
      - research-paper-writing
      - jupyter-live-kernel
    homepage: https://github.com/NousResearch/hermes-agent/tree/main/optional-skills/research/darwinian-evolver
prerequisites:
  commands:
    - python3
  env_vars: []
---

# Darwinian Evolver

## Overview

`darwinian-evolver` evolves **text artifacts** — prompts, regexes, SQL
queries, or small code snippets — toward a user-supplied fitness function.
It maintains a population of candidates, repeatedly applies LLM-driven
mutation and crossover operators, and preserves both the fittest and the
most behaviorally diverse solutions across generations.

Three tiers of evolution are supported:

| Tier | Engine | License | When |
|---|---|---|---|
| **1 — core** | In-process (μ+λ)-ES · MAP-Elites · NSGA-II | MIT (this skill) | Default. Works for prompts, regex, SQL, small Python. |
| **2 — heavy** | External `darwinian-evolver` CLI (Imbue) or `openevolve` CLI | AGPL v3 / Apache 2.0 | Opt-in. Long-horizon code evolution with verification. |
| **3 — bridge** | Emit DSPy-compatible traces → `NousResearch/hermes-agent-self-evolution` | MIT | Hand off to the reflective GEPA pipeline for skill/tool/prompt optimization at repo scale. |

The tiers are independent; Tier 1 has zero non-stdlib runtime deps beyond
`httpx` (already in Hermes core). Tiers 2 and 3 are thin adapters that
fail gracefully when their externals are absent.

## When to use

Invoke this skill when the user asks for any of:

- "Evolve / optimize / improve this prompt (regex, SQL, function) ..."
- "Find the best prompt for ..."
- "Generate variations of ... and pick the best"
- "Run a prompt optimization loop / genetic algorithm on ..."
- "Tune this prompt against these I/O examples"
- "Bridge this experiment to DSPy / GEPA / hermes-agent-self-evolution"

Do **not** use for:

- Single-shot generation (just call the model)
- Open-ended creative writing (no fitness signal available)
- Tasks without a clear evaluation — evolution requires a score

## Prerequisites

Tier 1 (default) needs nothing beyond Python 3.11+ and a reachable LLM
endpoint configured in `~/.hermes/config.yaml`. Any OpenAI-compatible
server works: local llama.cpp / Ollama / vLLM / LM Studio, or hosted
OpenRouter / Anthropic / OpenAI.

Tier 2 optional installs (only when user opts in):

```bash
# Apache 2.0, safer default for Tier 2
pip install openevolve

# AGPL v3 — license-check before use. The skill invokes it only as a
# subprocess (mere aggregation); no Python imports into Hermes.
pip install darwinian-evolver
```

## Quick reference

| Intent | Command |
|---|---|
| Create a new experiment | `evolver init <name> --task prompt` |
| Run the evolutionary loop | `evolver run <dir> --generations 30 --budget 1.00` |
| Check live progress | `evolver status <dir>` |
| Get the top-K candidates | `evolver best <dir> --k 5` |
| Inspect a candidate's ancestry | `evolver lineage <dir> --id <cid> --format mermaid` |
| Hand off to DSPy / GEPA | `evolver export <dir> --format dspy-jsonl` |
| Reproduce a run bit-for-bit | `evolver replay <dir> --seed N` |

All subcommands print structured JSON to stdout.

## Workflow

```
  1.  hermes-terminal>  evolver init email_regex --task regex
                        → scaffolds ~/.hermes/skills/research/darwinian-evolver/data/email_regex/
                           with fitness.py, seed/initial.txt, experiment.yaml

  2.  user edits fitness.py — defines what "good" means
      user edits seed/initial.txt — drops in a starting regex

  3.  hermes-terminal>  evolver run email_regex --generations 40 --budget 0.50
                        → prints one JSON line per generation:
                          {"gen": 7, "best": {"id": "...", "fitness": 0.82},
                           "pareto_size": 4, "budget_used": 0.17}

  4.  hermes-terminal>  evolver best email_regex --k 3
                        → returns top-3 regexes with scores + lineage IDs

  5.  hermes-terminal>  evolver lineage email_regex --id abc123 --format mermaid
                        → Mermaid DAG showing which operators produced the winner
```

## Theoretical foundations

The skill implements a small library of well-studied primitives and
composes them. Citations indicate the source the implementation follows.

- **(μ+λ)-Evolution Strategy** (Bäck & Schwefel, 1993) — the default
  population-replacement scheme. μ elite parents produce λ offspring;
  survival selection operates on the combined pool.
- **MAP-Elites** (Mouret & Clune, 2015) — quality-diversity archive. The
  fitness landscape is partitioned by *behavioral descriptors* (e.g.
  prompt length, CoT presence, instruction style). Each cell retains its
  best occupant; illumination over the feature space is a first-class
  goal, not a side effect.
- **NSGA-II** (Deb et al., 2002) — fast non-dominated sort for
  multi-objective runs where `fitness()` returns a dict (e.g. accuracy
  vs. cost vs. latency). Crowding-distance breaks ties.
- **Language-Model Crossover** (Meyerson et al., 2023) — two-parent
  semantic blend via a structured LLM prompt. Complements segment-splice
  for prompts and AST-splice for code.
- **PromptBreeder-style meta-mutation** (Fernando et al., 2023) — the
  mutation prompt itself is part of the genome and co-evolves.
- **GEPA-lite critique-then-edit** (Agrawal et al., 2025) — two-phase
  mutation: the LLM first critiques the candidate against observed
  failures, then edits to the critique. Cheaper approximation of the
  full reflective-GEPA loop used by `hermes-agent-self-evolution`.
- **Novelty search** (Lehman & Stanley, 2011) — available as a mutation
  bias when the user opts to escape plateaus by pursuing behavioral
  distance rather than score improvement.
- **Successive halving / Hyperband-lite** (Li et al., 2017) — evaluates
  offspring on a cheap probe first; only survivors receive the full
  evaluation budget.
- **Exp3 bandit** (Auer et al., 2002) over mutation operators — cheap
  operators dominate early generations; expensive ones only fire once
  cheap ones plateau.

## Fitness contract

The user owns the fitness function. Drop it at `fitness.py` in the
experiment directory:

```python
from evolver_sdk import fitness_spec

@fitness_spec(
    held_out_frac=0.2,            # reserved eval split vs. reward hacking
    timeout_s=30,                  # hard per-eval wall clock
    objectives=["accuracy", "cost"],  # if scalar, omit → single-objective
)
def fitness(candidate: str, context: dict) -> float | dict[str, float]:
    """Score a candidate. Return higher = better for each objective."""
    ...
```

The evaluator guarantees:

- `context["seed"]` is propagated for reproducibility.
- Code candidates run in a subprocess with `resource.setrlimit` caps.
- A 20 % held-out split re-scores the top-5 each generation; candidates
  with more than a 15 % train-to-held-out generalization gap get a
  reward-hacking penalty.
- Global `--budget` (USD or tokens) hard-kills the run.

## Architecture

```
evolver.py           CLI entry (argparse subcommands → JSON stdout)
algorithms.py        (μ+λ)-ES, MAP-Elites, NSGA-II
operators.py         LLM mutation + crossover, Exp3 bandit
evaluator.py         Async batch fitness, successive halving, held-out guard
sandbox.py           Subprocess + rlimit + timeout for code eval
storage.py           SQLite lineage graph + content-addressed genomes
llm.py               OpenAI-compat client, seed propagation, prompt caching
adapters.py          Tier 2 (subprocess) and Tier 3 (DSPy JSONL) wrappers
```

All scripts are stdlib-only except for `httpx` (Hermes core dep). No
cross-imports into the Hermes agent code — this skill is self-contained.

## Storage layout

```
~/.hermes/skills/research/darwinian-evolver/data/<experiment>/
├── experiment.yaml   # static config
├── fitness.py        # user-owned
├── seed/             # initial genomes
├── lineage.db        # SQLite: candidates, fitness, lineage, budget_ledger
└── logs/<run-id>/    # per-run JSONL stream
```

Genomes are content-addressed (`blake2b(genome)[:16]`), so duplicate
mutations collapse into a single node and crossover reuses identical
parents without re-evaluation.

## Pitfalls

- **No fitness signal → no evolution.** Creative writing tasks without a
  scoring function degenerate into expensive random walks.
- **Reward hacking.** Candidates optimize what you measure. Use the
  `held_out_frac` guard and inspect the top candidates manually.
- **LLM-as-judge drift.** If you use an LLM as your judge, pin the judge
  model and seed; otherwise rankings are not comparable across days.
- **Budget blowup.** Always set `--budget`. Prompt caching helps but
  Tier 1 with 16 population × 30 generations × 6 LLM calls per candidate
  is ~3000 requests.
- **Determinism with local servers.** vLLM and llama.cpp honor `seed`;
  Ollama-backed llama.cpp also does. Anthropic and OpenAI honor `seed`
  as a best-effort hint only.

## Verification

```bash
# Help text sanity check
python3 ~/.hermes/skills/research/darwinian-evolver/scripts/evolver.py --help

# Run the packaged example end-to-end (requires local LLM configured)
cd ~/.hermes/skills/research/darwinian-evolver/examples/summarize_10_words
python3 ../../scripts/evolver.py init summarize_10w --task prompt
python3 ../../scripts/evolver.py run summarize_10w --generations 5 --budget 0.10

# Reproduce the same run bit-for-bit
python3 ../../scripts/evolver.py replay summarize_10w --seed 42
```

## Scope (v0.1)

Shipped:

- Tier 1 core: (μ+λ)-ES, MAP-Elites, NSGA-II, six LLM operators,
  held-out + budget guards, SQLite lineage, Mermaid lineage export.
- Tier 2 adapters: `openevolve` (Apache 2.0), `darwinian-evolver`
  (AGPL, subprocess only).
- Tier 3 adapter: DSPy-compatible JSONL export.
- Templates for prompt, regex, SQL, code fitness.
- End-to-end `summarize_10_words` example.

Explicitly **not** in scope for v0.1:

- RL fine-tuning (Phase 4 of #337).
- UI dashboard.
- Distributed / multi-node execution.
- Automatic hyperparameter search.
- Importing Imbue's AGPL Python into this skill — CLI only, always.

## Related work in this repo

- **Issue #336** — Darwinian Evolver Skill (this closes it, scoped).
- **Issue #337** — Unified self-improvement plan (this is Phase 3).
- **Issue #483** — Missing-affordance detection feeds new targets here.
- **Issue #1935** — Skill Factory generates skills; this evolves them.
- **`NousResearch/hermes-agent-self-evolution`** — Tier 3 export is the
  data handoff to that pipeline.
