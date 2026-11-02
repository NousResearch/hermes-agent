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
  dedicated Hermes home, same model, same tasks, same reps. Each `ab_eval run`
  starts one foreground gateway from that arm's checkout for the whole batch
  of tasks and repetitions. Switching arms starts a new gateway. The home's
  configuration and persistent state are retained across batches.
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
- **Verified evidence.** Relay captures exporter configuration when the gateway
  starts. Changing the chat client's environment does not change an existing
  gateway's trace destination. Each batch keeps one configuration and shared
  trace stream. The batch shares one 60-second budget for outstanding trace
  completion before graceful shutdown. After the gateway shuts down,
  the runner matches each trial's working directory to exactly one session,
  selects its events by Relay root UUID, and validates the complete session.
  These events are saved in the existing per-trial files. The Relay
  `session_uuid` (distinct from the CLI session ID) is saved in `meta.jsonl`
  and checked again when reporting.
- **Resume-safe.** Trials enter `meta.jsonl` only with verified traces. A model
  task that fails with output, such as exhausting its turn budget, remains a
  measurement with its original exit code and the existing task-success checks.
  Resume skips that measured attempt instead of retrying until success. As
  before, nonzero exits with no output are treated as startup failures and left
  unrecorded. After clean gateway shutdown, each candidate is validated
  independently: an incomplete trial does not discard other valid sessions.
  Missing, malformed, ambiguous, and incomplete traces are errors, never
  zero-cost results. Results are committed after batch shutdown; a hard
  interruption can require rerunning unrecorded trials. Recorded entries are
  revalidated before being skipped.

## Setup

The runner handles Linux, macOS, WSL, and native Windows process layouts. A
Windows virtual-environment launcher can have a different PID from the gateway;
the runner verifies the gateway is its child or a descendant of that child,
and targets the actual gateway worker for graceful shutdown. Native platform
validation runs in the corresponding OS test lanes.

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

   Use an idle, dedicated home, without messaging platforms or scheduled jobs,
   and run one battery at a time. If another gateway already owns the home,
   the runner refuses to attach; it only shuts down the child it started.
   Do not point `ABEVAL_HOME` at your everyday Hermes home.

   `ABEVAL_HOME` may also select a dedicated named profile such as
   `/tmp/abeval-home/profiles/eval`. Both gateway and chat commands stay on that
   exact home, regardless of the saved active profile. The runner does not
   change the saved profile selection. On Hermes versions that normally serve
   named profiles through a shared host gateway, the dedicated evaluation
   profile must opt into its own gateway with `gateway.standalone: true` in
   its `config.yaml`. The runner preserves Hermes's existing ownership guards.

   The runner writes a per-batch Relay `plugins.toml` using the observability
   schema supported by the pinned Relay 0.10 runtime (version 4). No Hermes
   observability plugin needs to be enabled. Both checkouts must support the
   foreground gateway and its planned-stop API; use their supported Python
   runtime with NeMo Relay installed.

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

An `ab_eval.py run` exit of **0** means its measurements are complete, even if
some model tasks failed. Exit **3** means measurements are missing but the
gateway shut down cleanly; `run_all.sh` continues the other arms/models, prints
the available results, and exits 3 overall. Missing trials can be retried on
resume. Timeouts, interruptions, and gateway lifecycle failures stop the
battery instead. Valid prior trials are retained after clean shutdown when
possible; a failed shutdown does not publish the batch.

## Reading the results

`wall_s` measures each chat client invocation. Starting the batch gateway,
waiting for trace completion, and shutting down are excluded. Background
gateway warm-up can still affect the first chat; this is not a controlled
cold-start benchmark. Compare both arms with this runner: older batches reused
gateways inconsistently, so historical wall times are not directly comparable.
Trial diagnostics are in `runs/<model>/<arm>/<run_id>/chat.stdout` and
`chat.stderr`. Each record's `batch_id` points to the gateway log, fixed Relay
configuration, and shared trace in `results/<model>/<arm>/batches/<batch_id>/`.
Traces and logs can contain task data; keep the evaluation workspace private.

Older metadata lacks the verified session UUID. Use a fresh `ABEVAL_ROOT` to
rerun those trials; adding IDs to old records would not repair missing or
mixed traces. The batch stream contains multiple sessions; each extracted trial
file must contain exactly one. Tasks that delegate to subagents or rotate
sessions need a separate scoring contract before being added to this battery.

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
