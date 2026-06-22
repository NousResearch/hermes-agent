# Memory v2 evals

Memory v2 evals are deterministic, local regression checks for Hermes's Memory v2 provider. They are meant to catch retrieval regressions before dogfood or release work: source-grounded recall, stale/irrelevant memory suppression, and bounded memory packet behavior.

The default packaged fixture is:

```text
plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml
```

## Purpose

Use these evals to answer practical questions:

- Does Memory v2 retrieve the right facts for preference and project-continuity queries?
- Does it preserve source references for recalled memory?
- Does it suppress irrelevant queries instead of injecting unrelated memory?
- Does it stay deterministic and cheap enough to run in local regression tests?

The eval harness is not a full agent benchmark. It exercises the memory ingestion, consolidation, routing, retrieval, and scoring layers without making model or network calls.

## Baselines

Local deterministic baselines are implemented under `plugins/memory/memory_v2/evals/`:

- `no_memory` — returns no memory. This is the floor for recall and a useful control for suppression.
- `raw_fts` — indexes raw event text with SQLite FTS and retrieves lexical matches. This tests whether Memory v2 beats simple log search.
- `memory_v2` — runs the Memory v2 fixture path through the real local write gate, consolidation, index, router, and packet composer.

External adapter status helpers live in `plugins/memory/memory_v2/evals/adapters.py`, but local regression commands do not require external providers, API keys, or network access.

## Metrics

Each query produces a score row with:

- `source_recall` — fraction of expected source refs found in retrieved refs.
- `text_contains` — fraction of expected answer substrings present in the answer or memory packet.
- `suppression` — 1.0 when a query that should not retrieve memory retrieves nothing; 0.0 when it leaks memory into irrelevant queries.
- `retrieved_count` — number of records returned.
- `token_estimate` — rough packet-size estimate for budget checks.
- `latency_ms` — local retrieval time for the query.

Reports summarize metric averages per baseline.

## Command examples

Run the packaged local fixture with all deterministic local baselines:

```bash
python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --baseline no_memory \
  --baseline raw_fts \
  --baseline memory_v2
```

Write a JSON report to a file:

```bash
python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --output memory-v2-eval-report.json
```

Run eval unit tests:

```bash
python -m pytest tests/plugins/memory/evals -q
```

Run dogfood scenarios and embed the local eval summary in the dogfood report:

```bash
python -m plugins.memory.memory_v2.dogfood \
  --target-home ~/.hermes/profiles/memory-v2-dogfood \
  --source-home ~/.hermes \
  --default-home ~/.hermes \
  --fresh \
  --run-local-eval
```

For release/regression work, run:

```bash
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
python -m pytest tests/plugins/memory/test_memory_v2_dogfood.py tests/plugins/memory/evals -q
python -m ruff check plugins/memory/memory_v2/dogfood.py tests/plugins/memory/test_memory_v2_dogfood.py -q
```

## Adding fixtures

Add new YAML fixtures under:

```text
plugins/memory/memory_v2/evals/fixtures/
```

A fixture should include:

- `version`, `name`, and `description` metadata.
- `events`: deterministic user/assistant events with stable `id`, `session_id`, `role`, and `text` fields.
- `queries`: query cases with stable `id`, expected route label, query `text`, `expected_answer_contains`, `expected_source_refs`, and optional `should_retrieve: false` for suppression tests.

Keep fixtures small, synthetic, and privacy-safe. Do not copy real private conversations or machine-specific paths into fixtures.

After adding a fixture, add or update tests under `tests/plugins/memory/evals/` and run the CLI against the packaged fixture path.

## Adding an external adapter

External adapters should be opt-in and separate from deterministic local regression tests.

Recommended process:

1. Add a readiness/status entry in `plugins/memory/memory_v2/evals/adapters.py` that checks import and required environment variables without making network calls.
2. Implement the adapter behind an explicit CLI or test flag so local tests remain offline by default.
3. Normalize adapter results into the same report shape as local baselines.
4. Document required packages, environment variables, and limitations without including real keys or account identifiers.
5. Add skipped-by-default tests that verify adapter availability checks without requiring credentials.

## Limitations

- The harness is deterministic and local; it does not measure LLM answer quality or full multi-turn agent behavior.
- Substring scoring can miss semantically correct paraphrases and can over-reward copied text.
- The fixtures are intentionally small, so passing them is a regression signal, not proof of broad memory quality.
- Latency numbers are local-machine dependent and should be interpreted as rough smoke signals.
- External provider comparisons are not part of default local regression because they may require credentials, network access, provider-specific setup, and non-deterministic behavior.
