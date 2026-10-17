# Core-Toolset A/B Eval Harness

The hard A/B evaluation used for the August 2026 core-toolset performance
batch (tracker: [#77056](https://github.com/NousResearch/hermes-agent/issues/77056)).
It measures whether a set of tool-layer changes actually reduces model waste —
LLM turns, tool calls, tool errors, retries, result bytes, wall clock — on a
battery of **error-inducing tasks**, each derived from a waste class measured
in real production traffic.

## Design

- **One variable per comparison.** Code arms differ only by `PYTHONPATH`
  (a checkout of `origin/main` vs your integration branch). Guidance-ablation
  arms instead copy the same configured Hermes home and turn off exactly one
  existing `agent.*` switch: `tool_use_off`, `execution_off`, or `all_off`.
  The copied home keeps credentials but prevents session/config leakage.
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

The default guidance matrix runs `all_on`, `tool_use_off`, `execution_off`,
and `all_off` for each code arm. To run only a comparison, set (for example)
`ABEVAL_GUIDANCE_ARMS=all_on,execution_off`. Each guidance arm includes
tool-action and local write/read-back instruction-following tasks plus a
trivial edit (where excess verification is measurable cost). The first requires
`read_file`; the second explicitly requires `write_file` followed by `read_file`.
Success requires matching completed ATOF tool spans for the exact target, a
successful result, the expected read content, and (for read-back) a read started
after the write completed. Missing traces and failed runs cannot pass. These
local-file tasks measure compliance with an explicit user requirement; they do
not establish that execution guidance supplies external-state verification.
Arm configuration uses the normal comment-preserving YAML writer and keeps
unrelated settings, including inline mappings and non-default indentation.

Re-print tables any time (pass the arms that were run):

```bash
python3 ab_eval.py report --models "anthropic/claude-sonnet-4.5,qwen/qwen3-coder-30b-a3b-instruct" --arms "baseline-all_on,baseline-execution_off"
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
