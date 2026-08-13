# Core-Toolset A/B Eval Harness

The hard A/B evaluation used for the August 2026 core-toolset performance
batch (tracker: [#77056](https://github.com/NousResearch/hermes-agent/issues/77056)).
It measures whether a set of tool-layer changes actually reduces model waste —
LLM turns, tool calls, tool errors, retries, result bytes, wall clock — on a
battery of **error-inducing tasks**, each derived from a waste class measured
in real production traffic.

## Design

- **Two arms, one variable.** `baseline` and `fixes` runs differ ONLY by
  `PYTHONPATH` (a checkout of `origin/main` vs your integration branch). Same
  Hermes home, same model, same tasks, same reps.
- **Tasks are traps.** Each of the 9 tasks is constructed so a specific
  failure class fires: `python` vs `python3`/venv confusion, an
  already-applied patch, an ambiguous multi-match edit, wrong-casing search,
  hidden-dir search, giant truncated output, cd-heavy multi-dir work, a
  blocklist-tripping inline script, and a paginated big-file read. A change
  that claims to fix a waste class must move the needle on its trap.
- **Scoring is from traces, not self-report.** Metrics come from NeMo Relay
  ATOF traces emitted by the run itself (`llm`/`tool` scope events), plus wall
  clock and a per-task programmatic success check (marker strings + on-disk
  verification).
- **Resume-safe.** Completed `run_id`s in `meta.jsonl` are skipped, so a
  killed battery continues where it left off. Startup crashes (nonzero exit
  with empty output) are NOT recorded — they retry on resume instead of
  polluting cells (this bit the first pass of the Aug 2026 run).

## Setup

1. Create a dedicated Hermes home with credentials for the models under test:

   ```bash
   export ABEVAL_HOME=/tmp/abeval-home
   mkdir -p "$ABEVAL_HOME"
   # minimal config.yaml + provider key, e.g. OpenRouter:
   cat > "$ABEVAL_HOME/config.yaml" <<'YAML'
   model:
     provider: openrouter
   YAML
   printf 'OPENROUTER_API_KEY=%s\n' "$KEY" > "$ABEVAL_HOME/.env"
   HERMES_HOME=$ABEVAL_HOME hermes plugins enable observability/nemo_relay
   ```

2. Prepare the two trees:

   ```bash
   git worktree add /tmp/abeval-baseline origin/main
   # fixes tree = your integration branch checkout
   ```

## Run

```bash
cd scripts/toolperf_abeval
export ABEVAL_ROOT=/tmp/abeval-workspace   # results + sandboxes land here
export ABEVAL_HOME=/tmp/abeval-home
./run_all.sh /tmp/abeval-baseline /path/to/fixes-tree 3 \
  "anthropic/claude-sonnet-4.5" "qwen/qwen3-coder-30b-a3b-instruct"
```

108 runs (2 models x 2 arms x 9 tasks x 3 reps) took ~2.5h on the original
battery. Re-print tables any time:

```bash
python3 ab_eval.py report --models "anthropic/claude-sonnet-4.5,qwen/qwen3-coder-30b-a3b-instruct"
```

## Reading the results

- Weak models are the signal. Strong models recover from most induced errors
  in one turn, so expect parity there; the fixes' win shows up as fewer
  turns/tool calls/errors on the weak model. The Aug 2026 batch measured
  −21% turns, −29% tool calls, errors→0, −23% wall on
  qwen3-coder-30b, with sonnet-4.5 at parity.
- Success-rate deltas at n=3 are noise. Audit any sub-100% cell run-by-run
  (read `meta.jsonl` `tail`) before calling it a regression.
- The eval can catch product gaps on BOTH arms — e.g. the original run found
  the hidden-file search probe only fired on total-zero-match searches
  (fixed on main since).

## Extending

Add a task by appending to `TASKS` (the prompt), `make_sandbox` (the trap),
and `SUCCESS` (the programmatic check). Keep checks strict and mechanical —
marker strings and on-disk state, never judge-by-vibes.
