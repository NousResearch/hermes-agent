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
   ```

   The runner writes a per-run Relay `plugins.toml` and points the native SDK
   integration at it; no Hermes observability plugin needs to be enabled.

2. Prepare the two trees:

   ```bash
   git worktree add /tmp/abeval-baseline origin/main
   # fixes tree = your integration branch checkout
   ```

## Run

```bash
cd evals/toolperf_abeval
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

## Credit verdicts (`credit`)

`report` is a human-readable table: it pairs nothing, accounts for nothing and publishes no
uncertainty. `credit` is the machine verdict over the same results tree:

```bash
export ABEVAL_ROOT=/tmp/abeval-workspace
export ABEVAL_HOME=/tmp/abeval-home
python3 ab_eval.py credit --models "anthropic/claude-sonnet-4.5" --reps 3 --seed 7 \
  --metric ok --guardrails llm,tools,errs --margin 0.10
```

It is offline and deterministic — no model call, no network, no credentials — and it appends
one row per candidate to `$ABEVAL_ROOT/results/<model>/verdicts.jsonl`, credited, denied and
withheld alike, so rejected candidates survive as negative data.

Declare before you look; every declared value is echoed into the artifact, so a reader sees
what was asked before what was found:

| Flag | Default | Meaning |
|---|---|---|
| `--metric` | `ok` | the single primary (promotion) metric |
| `--alpha` | `0.05` | one-sided significance |
| `--comparisons` | `1` | candidates judged on this battery; alpha is Bonferroni-corrected |
| `--min-pairs` / `--min-tasks` | `5` / `5` | minimum paired runs / distinct tasks |
| `--bootstrap` | `10000` | resamples |
| `--seed` | required | determinism: same inputs + same seed = identical bounds |
| `--guardrails` / `--margin` | `llm,tools,errs` / `0.10` | metrics that must be non-inferior within a relative margin |
| `--reps` | derived | the scheduled rep count; declare it, a battery whose last rep vanished cannot be reconstructed from its survivors |

Outcomes:

- **`withheld`** — a precondition failed, and `reason` names it: `incomplete` (a scheduled
  cell has no usable evidence: crashed, lost its trace, lost its oracle, or was never
  recorded), `battery_changed` / `battery_unverified` (the fingerprint recorded by the
  baseline runs does not match this file's battery, or predates fingerprints), `insufficient`
  (fewer pairs or tasks than declared).
- **`denies`** — the evidence exists but does not clear zero (`no_effect`), or a guardrail
  fell through its floor (`guardrail_breach`). `NO UPDATE` is a legitimate result.
- **`credits`** — complete, attributable, paired and above the declared bar.

Why it is strict, one line each:

- Every scheduled cell is accounted for, so "the gates behaved on the completed cases" can
  never be confused with "the experiment completed all the scheduled cases".
- A run whose trace is missing is excluded from the means and its twin is counted as
  unpaired — it never enters the table as a zero-waste run (which is how a *lost* trace used
  to read as a *faster* run).
- The bootstrap resamples TASKS, not runs, so raising `--reps` buys precision inside a task
  instead of significance across tasks.
- Deltas are **relative** improvements, so one `--margin` covers turns and KB alike.
- The artifact carries no trace text — no `tail`, no sandbox paths — so a verdict can be
  pasted into a public issue while the raw traces stay on the machine.

`run` additionally records `tree` (the git revision of the `--pythonpath` tree, read from
`.git` without spawning a process), `pythonpath` and the battery fingerprint in each
`meta.jsonl` row: that is what makes a verdict attributable to a revision pair and a battery
version, and what makes a silent battery edit visible.

Known limits: the runner writes no row for a startup crash, so a dropped cell is
`unaccounted` (and withholds the verdict) rather than assumed harmless; `TASKS` is 9 traps
with no healthy control task, so guardrails protect metrics, not task families; and no noise
floor is recorded for the battery yet (#111119), so a small real effect will simply be
`withheld`/`denies`.

## Extending

Add a task by appending to `TASKS` (the prompt), `make_sandbox` (the trap),
and `SUCCESS` (the programmatic check). Keep checks strict and mechanical —
marker strings and on-disk state, never judge-by-vibes.
