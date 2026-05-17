---
name: darwinian-evolver
description: Evolve prompts/regex/SQL/code with Imbue's evolution loop.
version: 0.1.0
author: Bihruze (Asahi0x), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [evolution, optimization, prompt-engineering, research]
    related_skills: [arxiv, jupyter-live-kernel]
---

# Darwinian Evolver

Run Imbue's [darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) — an
LLM-driven evolutionary search loop — to optimize a **prompt, regex, SQL query,
or small code snippet** against a fitness function.

Status: thin wrapper around the upstream tool. The skill installs it, walks the
agent through writing a `Problem` definition (organism + evaluator + mutator),
and drives the loop via the upstream CLI or a small custom Python driver.

**License:** the upstream tool is **AGPL-3.0**. The skill ONLY ever invokes it
via the upstream CLI or a `subprocess`/`uv run` call (mere aggregation). Do NOT
import upstream classes into Hermes itself.

## When to Use

- User says "optimize this prompt", "evolve a regex for X", "auto-improve this
  code/SQL", "search for a better instruction".
- You have a scorer (exact match, regex pass-rate, unit test, LLM-judge, runtime
  metric) AND a starting candidate (organism). If you don't have a scorer, stop
  and define one first — that's the hard part.
- Cost is OK: a typical run is 50–500 LLM calls. On gpt-4o-mini that's pennies;
  on Claude Sonnet it can be a few dollars.

Do **not** use this when:
- The optimization target is differentiable (use gradient descent / DSPy).
- You only need to try 2–3 variants — just write them by hand.
- The fitness signal is purely subjective with no measurable criterion.

## Prerequisites

- Python ≥3.11
- `git`, `uv` (or `pip`)
- One of: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`

The skill ships a small `parrot_openrouter.py` driver that uses `OPENROUTER_API_KEY`
via the OpenAI SDK, so any model on OpenRouter works. The upstream CLI itself
hardcodes Anthropic and needs `ANTHROPIC_API_KEY`.

## Install (One-Time)

Run via the `terminal` tool:

```bash
mkdir -p ~/.hermes/cache/darwinian-evolver && cd ~/.hermes/cache/darwinian-evolver
[ -d darwinian_evolver ] || git clone --depth 1 https://github.com/imbue-ai/darwinian_evolver.git
cd darwinian_evolver && uv sync
```

Verify:

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver \
  && uv run darwinian_evolver --help | head -5
```

## Quick Start — The Built-In Parrot Example

Tiny smoke test (requires `ANTHROPIC_API_KEY`):

```bash
cd ~/.hermes/cache/darwinian-evolver/darwinian_evolver
uv run darwinian_evolver parrot \
  --num_iterations 2 \
  --num_parents_per_iteration 2 \
  --mutator_concurrency 2 --evaluator_concurrency 2 \
  --output_dir /tmp/parrot_demo
```

Outputs:
- `/tmp/parrot_demo/snapshots/iteration_N.pkl` — pickled population per iteration
- `/tmp/parrot_demo/<jsonl>` — per-iteration JSON log (path printed at end)

Open `~/.hermes/cache/darwinian-evolver/darwinian_evolver/darwinian_evolver/lineage_visualizer.html`
in a browser and load the JSON log to see the evolutionary tree.

## Quick Start — OpenRouter Driver (No Anthropic Key)

The skill ships `scripts/parrot_openrouter.py` — same parrot problem, but the
LLM call goes through OpenRouter so any provider works.

```bash
# From wherever the skill is installed:
SKILL_DIR=~/.hermes/skills/research/darwinian-evolver
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver

cd "$DE_DIR" && \
  EVOLVER_MODEL='openai/gpt-4o-mini' \
  uv run --with openai python "$SKILL_DIR/scripts/parrot_openrouter.py" \
    --num_iterations 3 --num_parents_per_iteration 2 \
    --output_dir /tmp/parrot_or
```

Inspect the result with `scripts/show_snapshot.py`:

```bash
uv run --with openai python "$SKILL_DIR/scripts/show_snapshot.py" \
  /tmp/parrot_or/snapshots/iteration_3.pkl
```

Expected output: 7 evolved prompt templates ranked by score, with the best
landing around 0.6–0.8 (the seed `Say {{ phrase }}` scored 0.000).

## Defining a Custom Problem

The skill ships `templates/custom_problem_template.py` — copy, edit, run.
Three things you must define:

1. **`Organism`** — a Pydantic `BaseModel` subclass holding the artifact being
   evolved (`prompt_template: str`, `regex_pattern: str`, `sql_query: str`,
   `code_block: str`, etc.). Add a `run(*args)` method that exercises it.

2. **`Evaluator`** — `.evaluate(organism) -> EvaluationResult(score=..., trainable_failure_cases=[...], holdout_failure_cases=[...], is_viable=True)`.
   - **`score`** is in `[0, 1]`. Higher is better.
   - **`trainable_failure_cases`** — what the mutator sees. Include enough
     context (input, expected, actual) for the LLM to diagnose.
   - **`holdout_failure_cases`** — kept out of the mutator's view. Use these
     to detect overfitting.
   - **`is_viable=True`** unless the organism is completely broken (raises,
     returns None, etc.). A 0-score viable organism is fine — it just gets
     down-weighted in parent selection.

3. **`Mutator`** — `.mutate(organism, failure_cases, learning_log_entries) -> list[Organism]`.
   Typically: build an LLM prompt that includes the current organism + a
   failure case + an ask to propose a fix; parse the LLM's response; return
   a new `Organism`. Return `[]` on parse failure — the loop handles it.

Then write a driver script that wires `Problem(initial_organism, evaluator, [mutators])`
into `EvolveProblemLoop` and iterates over `loop.run(num_iterations=N)` — the
shipped `scripts/parrot_openrouter.py` is the reference.

## Hyperparameters That Actually Matter

| flag | default | when to change |
|---|---|---|
| `--num_iterations` | 5 | bump to 10–20 once you trust the evaluator |
| `--num_parents_per_iteration` | 4 | drop to 2 for cheap exploration |
| `--mutator_concurrency` | 10 | drop to 2–4 to avoid rate limits |
| `--evaluator_concurrency` | 10 | same; evaluator hits the LLM too |
| `--batch_size` | 1 | raise to 3–5 once your mutator handles multiple failures |
| `--verify_mutations` | off | turn on once mutator is wasteful (>10× cost saving on later runs per Imbue) |
| `--midpoint_score` | `p75` | leave alone unless scores cluster |
| `--sharpness` | 10 | leave alone |

## Pitfalls

1. **`Initial organism must be viable`** — set `is_viable=True` in your
   `EvaluationResult` even on a 0-score seed. The loop refuses non-viable
   organisms because they imply the loop has nothing to evolve from.
2. **Provider content filters kill runs.** Azure-backed OpenRouter models
   reject phrases like "ignore previous instructions" with HTTP 400. Wrap
   the LLM call in `try/except` and return `f"<LLM_ERROR: {e}>"` — the
   evolver will just score that organism 0 and move on.
3. **`loop.run()` is a generator** — calling it doesn't run anything until
   you iterate. Use `for snap in loop.run(num_iterations=N):`.
4. **Snapshots are nested pickles.** `iteration_N.pkl` contains a dict with
   `population_snapshot` (more pickled bytes). To unpickle you must have the
   `Organism` class importable under the same dotted path it was pickled at.
5. **Concurrency defaults are aggressive.** 10/10 will hit rate limits on
   most providers. Start with 2/2.
6. **CLI is hardcoded to Anthropic.** `uv run darwinian_evolver <problem>`
   reaches for `ANTHROPIC_API_KEY` and uses Claude Sonnet. To use any other
   provider, write a driver like `parrot_openrouter.py`.
7. **AGPL.** Never `from darwinian_evolver import ...` inside Hermes core.
   Custom driver scripts under `~/.hermes/skills/...` are user-side and fine.
8. **No PyPI package.** `pip install darwinian-evolver` will pull the wrong
   thing. Always install from the GitHub repo.

## Verification

After install + a parrot run, exit code 0 from this is sufficient:

```bash
DE_DIR=~/.hermes/cache/darwinian-evolver/darwinian_evolver
ls "$DE_DIR/darwinian_evolver/lineage_visualizer.html" >/dev/null && \
cd "$DE_DIR" && uv run darwinian_evolver --help >/dev/null && \
echo "darwinian-evolver: OK"
```

## References

- [Imbue research post](https://imbue.com/research/2026-02-27-darwinian-evolver/)
- [ARC-AGI-2 results](https://imbue.com/research/2026-02-27-arc-agi-2-evolution/)
- [imbue-ai/darwinian_evolver](https://github.com/imbue-ai/darwinian_evolver) (AGPL-3.0)
- [Darwin Gödel Machines](https://arxiv.org/abs/2505.22954)
- [PromptBreeder](https://arxiv.org/abs/2309.16797)
