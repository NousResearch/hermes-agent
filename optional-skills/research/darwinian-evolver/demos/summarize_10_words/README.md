# summarize_10_words — packaged end-to-end demo

Evolves a prompt that instructs a 10-word summary, starting from the
trivial seed `"Summarize this."`.

## Fitness

`fitness.py` scores a candidate on three deterministic signals:

| component | weight | what it rewards |
|---|---|---|
| word-count proximity | 0.7 | being close to 10 words |
| brevity keyword      | 0.2 | explicitly mentioning "one" / "single" / "exactly" |
| character budget     | 0.1 | staying under 140 characters |

The fitness never calls the LLM — only the mutation operators do. That
keeps the demo cheap to re-run and makes the improvement curve
immediately visible even on a small local model.

## Run

```bash
# One-time scaffold into ~/.hermes/skills/research/darwinian-evolver/data/summarize_10w
python3 ../../scripts/evolver.py init summarize_10w --task prompt

# Copy the demo fitness + seed in (idempotent cp)
EXP=~/.hermes/skills/research/darwinian-evolver/data/summarize_10w
cp fitness.py          "$EXP/fitness.py"
cp seed/initial.txt    "$EXP/seed/initial.txt"

# Evolve (uses whichever model Hermes is configured for)
python3 ../../scripts/evolver.py run summarize_10w \
    --generations 15 --pop 6 --budget 0.25 --algorithm map-elites

# Best-so-far
python3 ../../scripts/evolver.py best summarize_10w --k 3

# Ancestry graph for the winner
WINNER=$(python3 ../../scripts/evolver.py best summarize_10w --k 1 \
         | python3 -c "import json,sys;print(json.load(sys.stdin)['candidates'][0]['id'])")
python3 ../../scripts/evolver.py lineage summarize_10w --id "$WINNER" --format mermaid
```

## Expected behaviour

Seed score is ~0.07 (`Summarize this.` has 2 words, no brevity keyword).
After 10–15 generations with MAP-Elites on a competent 7B-or-larger
local model, the best candidate typically approaches 0.95+ and reads
along the lines of:

> *"Summarize the passage in exactly one concise sentence of ten
> well-chosen words."*

…with a lineage DAG showing `paraphrase` and `structural_edit` as the
dominant operators, plus occasional `cot_inject` spurs that the archive
retains in a separate bin thanks to the CoT-presence behavioural axis.

## What to look at

* `lineage.db` — SQLite; `sqlite3 $EXP/lineage.db '.tables'` reveals
  the `candidates`, `fitness`, `lineage`, and `budget_ledger` tables.
* `evolver status summarize_10w` — one-shot JSON snapshot suitable for
  agent-facing polling.
* `evolver export summarize_10w --format dspy-jsonl` — hands the run
  off to `NousResearch/hermes-agent-self-evolution` for reflective
  GEPA-style refinement.
