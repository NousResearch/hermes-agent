# Compaction Eval Harness

Measures what context compaction actually costs in *recall*, not just tokens.

## What it does

1. Takes a real long transcript (JSON: `{"messages": [...]}`, chat format).
2. Generates a bank of factual recall questions from the region that
   compaction will summarize away (cached per transcript for reproducibility).
3. Runs the transcript through `ContextCompressor.compress()` under each
   policy in the matrix (current default, aggressive tail, codex-style, ...).
4. For each policy, asks a fresh LLM the recall questions with ONLY the
   post-compaction context, and judges answers against gold.
5. Emits a scorecard: recall accuracy vs tokens retained, per policy.

## Usage

```bash
# from repo root, venv active
python evals/compaction/runner.py \
    --transcript /path/to/lineage.json \
    --policies current,aggressive,floor10k \
    --questions 15 \
    --out evals/compaction/results/run1
python evals/compaction/report.py evals/compaction/results/run1
```

Transcripts are NOT committed (they contain real session data). Point
`--transcript` at a local file. See `fixtures.py` for the expected shape and
a synthetic-transcript generator used by CI smoke tests.

## Policies

Defined in `policies.py`. Each policy maps to `ContextCompressor` constructor
kwargs plus optional attribute overrides applied post-construction (e.g.
`tail_token_budget`). Add new policies there — the runner picks them up by
name.

## Notes

- Question generation and judging use `agent.auxiliary_client.call_llm`
  (same transport the compressor uses), so the harness needs a configured
  provider. Costs real tokens: ~(policies x questions) answer calls plus
  one generation and one judge pass.
- Accuracy is judged 2/1/0 (correct / partial / wrong); the scorecard
  reports normalized percent. The judge sees gold answers, the answerer
  does not.
- `--also-uncompacted` adds a control arm that answers from the full
  original transcript — the recall ceiling.
