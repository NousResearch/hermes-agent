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

## Building transcripts from real sessions (`scripts/`)

Compaction rotations mean a single active session rarely exceeds ~300K
tokens, but the *lineage* (parent→children chain) carries the full
uncompacted history. The scripts reconstruct those into eval transcripts:

```bash
# 1. ALWAYS copy the DB first — never point at the live state.db
cp ~/.hermes/state.db /tmp/state_copy.db

# 2. Find big lineages (sessions with parent_session_id form chains), then:
python evals/compaction/scripts/reconstruct_lineage.py \
    /tmp/state_copy.db <root_session_id> /tmp/lineage.json

# 3. (optional) Replay a 500K prefix through one checkout's compressor and
#    dump before/after for the HTML viewer:
python evals/compaction/scripts/replay_lineage.py <checkout> /tmp/lineage.json out.json 500000
python evals/compaction/scripts/build_html_report.py <runs_dir> report.html
```

`reconstruct_lineage.py` walks the whole descendant tree chronologically,
dedupes rotation-copied rows by content hash, strips synthetic compaction
artifacts (summaries, todo snapshots), and resolves the system prompt through
the `system_prompts` dedup table (sessions only carry a hash). The HTML
report renders before/after transcripts side by side with compaction
artifacts color-coded.

## Region-scoping tripwire

`test_region_scoping.py` plants sentinels in head/middle/tail and asserts the
summarizer's serialized-turns input carries ONLY the middle (compacted)
region in both legacy and lean modes. Run it directly or via pytest.

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
