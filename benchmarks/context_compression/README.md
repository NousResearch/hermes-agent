# Context-compression lexical-retention benchmark

This directory contains privacy-safe, provider-neutral fixtures for checking
which expected strings and known integrity constraints survive compaction. The
compact synthetic cases model failure shapes that matter in long Hermes
workflows; they are not a context-size or throughput load test.

- exact Windows paths and Unicode glyphs;
- late corrections that supersede stale state;
- reverse signals such as “stop” and “do not publish”;
- secrets that must not appear in a summary; and
- unrelated, tool-heavy material between the original fact and the active task.

The benchmark does **not** call an LLM. Any compressor or memory provider can
write a JSON object mapping case IDs to candidate summaries, then use the same
deterministic scorer:

```bash
python scripts/context_compression_benchmark.py \
  --fixture benchmarks/context_compression/long_session_v1.json \
  --summaries benchmarks/context_compression/example_summaries.json
```

Use `--output result.json` to save the machine-readable report. The process exits
with status 1 if any case fails.

## Fixture schema

Each case contains synthetic source `turns` and weighted `checks`:

- `contains`: the value must appear. Matching is exact by default; opt into
  case-folding or whitespace normalization with `normalize`.
- `contains_any`: at least one accepted deterministic wording must appear. Use
  this for facts with a small, auditable set of equivalent phrasings.
- `excludes`: the value must not appear. Exclusion checks are safety/integrity
  gates and do not inflate the retention score.
- `excludes_any`: none of the listed values may appear. Use this to cover known
  contradiction, stale-action, secret, and reverse-signal failure variants.

The scorer reports weighted **lexical** retention separately from exclusion
safety. A case passes only when both are perfect.

## Important limitation

This deterministic scorer does not prove semantic entailment. Arbitrary prose
can mention an expected string while negating or quoting it. Each repository
case therefore includes explicit exclusions for known inversions, and
`adversarial_summaries.json` verifies those attacks fail. This is a repeatable
lexical/integrity smoke test—not a replacement for a semantic judge or manual
review. Reports must not be described as semantic-accuracy scores.

## Adding cases

1. Use entirely synthetic names, paths, account data, and secrets.
2. Keep exact checks for paths, command flags, code points, and active requests.
3. Use `contains_any` only for genuinely equivalent restatements; do not add a
   variant merely to make a weak summary pass.
4. Add `excludes`/`excludes_any` checks and an adversarial summary for every
   correction or reverse signal that could cause the agent to take a wrong
   action.
5. Run the focused tests:

```bash
scripts/run_tests.sh tests/scripts/test_context_compression_benchmark.py
```

A future online runner can invoke Hermes, Hindsight, Supermemory, or another
provider to generate the candidate-summary JSON. Keeping generation separate
from scoring makes CI deterministic and allows apples-to-apples evaluation.
