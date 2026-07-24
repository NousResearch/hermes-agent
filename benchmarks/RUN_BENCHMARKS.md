# Benchmark Run Commands

Run these from the hermes-agent repo root on your benchmark branch.

By default, results write to `benchmarks/results/<backend>.json`, but for ad-hoc runs it is often cleaner to use a temporary output directory.

## 0. Verify code and tests

```bash
pytest tests/benchmarks -q
python -m benchmarks --backend baseline-flat --suite d --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

Expected in this framework branch:
- benchmark tests pass
- Suite D smoke test completes and writes a local result JSON file

## 1. Recommended local multi-seed runs

Use this for backends you can run locally and reset cleanly:

```bash
python -m benchmarks --backend baseline-flat --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir /tmp/bench-out
python -m benchmarks --backend holographic --suite all --runs 5 --seeds 42 43 44 45 46 --output-dir /tmp/bench-out
```

## 2. Service / API backends

Examples below use placeholders only. Never commit real credentials.

### mem0

```bash
MEM0_API_KEY="***" \
  python -m benchmarks --backend mem0 --suite all --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

### hindsight

```bash
HINDSIGHT_BASE_URL="http://localhost:8888" \
  python -m benchmarks --backend hindsight --suite all --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

### honcho

```bash
HONCHO_BASE_URL="http://localhost:8000" \
  python -m benchmarks --backend honcho --suite all --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

### retaindb

```bash
RETAINDB_API_KEY="***" \
  python -m benchmarks --backend retaindb --suite all --runs 1 --seeds 42 --output-dir /tmp/bench-out
```

## 3. Generate reports from local results

```bash
python -m benchmarks --report --result-file /tmp/bench-out/baseline-flat.json
python -m benchmarks --dashboard --result-file /tmp/bench-out/baseline-flat.json
python -m benchmarks --compare /tmp/bench-out/before.json /tmp/bench-out/after.json
```

## 4. Recommended truth-in-advertising workflow

For development:
- use 1 seed
- use heuristic judge
- run small suites first (`d`, then `a`, then `all`)

For serious comparison:
- use 5 seeds where feasible
- prefer shared-suite comparisons across backends
- say how many runs/seeds were used
- use `--judge llm` for semantic-sensitive conclusions

## 5. Troubleshooting

Common causes of bad runs:
- missing Python deps (`pytest`, `httpx`, `requests`)
- missing backend credentials
- local service not running
- stale or incomparable old result files in your output directory

If stdout JSON is noisy, prefer writing to an output directory and reading the result file instead of parsing stdout directly.
