# Browser Use Mode Benchmark

The A/B battery behind PR [#81958](https://github.com/NousResearch/hermes-agent/pull/81958)
(Browser Use CLI 3.0 mode, salvage of #66476 by @laithrw): built-in
`browser_*` toolset vs the single `browser_exec` driver, measured as total
task tokens / tool calls / wall clock at accuracy parity on live multi-step
web tasks.

## Design

- **Arms differ only by tree + config.** `base` runs the built-in twelve
  `browser_*` tools from a merge-base checkout; `pr` runs `browser_exec`
  (`browser.backend: browser-use`) from the branch checkout; `prns` is `pr`
  with the schema's helpers digest stripped to the header (isolates the
  digest's value). Each cell gets a throwaway `HERMES_HOME`; web-fetch
  credentials are stripped so every arm must actually drive the browser.
- **Tasks are oracle-checked.** toscrape-family sites (stable content, no
  anti-bot), regex oracles over the final answer. `tasks/easy.json` (5 tasks:
  price lookup, category extract, count/aggregate, login, pagination) and
  `tasks/hard.json` (6 tasks: full-category multi-page crawls, five-star
  rating aggregation, JS/delayed render, login chain, cross-category
  compare).
- **Resume-safe.** Completed cells in `results/*.jsonl` are skipped on rerun
  (same pattern as `scripts/toolperf_abeval`).
- **Backend matrix.** `orchestrate.py` drives a local headless-Chrome CDP;
  `orchestrate_cloud.py --backend nous-cloud|browserbase` provisions a real
  cloud browser per cell through the same provider plumbing the product uses.

## Run

```bash
# arms are pinned checkouts — e.g. merge-base worktree vs your branch
export BUBENCH_BASE_TREE=/path/to/merge-base-tree
export BUBENCH_PR_TREE=/path/to/branch-tree
```

Note: since #81958 merged (and #85170 made Browser Use the default driver),
a current-main checkout resolves to `browser_exec` in BOTH arms. The `base`
arm only measures the built-in `browser_*` toolset when `BUBENCH_BASE_TREE`
is pinned to a pre-#81958 tree (the original run used the PR's merge-base
worktree). For future A/Bs of new browser changes, pin `base` to the
merge-base of the change under test — the arms are generic.

```bash
google-chrome --headless=new --remote-debugging-port=9333 \
  --user-data-dir=/tmp/bubench-chrome --no-first-run --disable-gpu about:blank &

python3 orchestrate.py --tasks tasks/hard.json --reps 3     # 108 cells @ 2 models x 3 arms
python3 report.py results/results.jsonl
```

## Baseline scorecard (Aug 8-10 2026, the #81958 run — 204 cells total)

**Hard-task battery, local Chrome CDP** (6 tasks x 3 reps per cell; final
corrected-oracle readout, nothing excluded):

```
model      arm       ok  tok_mean  tok_med  calls  wall_s  vs base tok
opus4.8    base   18/18     64594    63776    4.1    25.2            —
opus4.8    pr     18/18     25934    25030    2.0    17.5         -60%
opus4.8    prns   18/18     25578    27934    3.2    23.7         -60%
kimi-k3    base   18/18     56464    53276    5.3    50.0            —
kimi-k3    pr     18/18     19230    16710    2.4    33.3         -66%
kimi-k3    prns   18/18     23099    21160    4.1    50.5         -59%
```

Digest ablation: pr (with helpers digest) 36/36 ok, mean 22,582 tok; prns
(header-only) 36/36 ok, mean 24,339 tok — the pinned 3.4KB digest costs
nothing and saves a little; the full 11KB live skill dump adds nothing.

**Backend matrix** (pr arm, same tasks):

```
model      backend          ok  tok_mean  calls   wall
opus4.8    local-cdp     17/18     25934    2.0   17.5
opus4.8    nous-cloud    12/12     33330    2.8   33.8
opus4.8    browserbase    6/6      26712    2.2   23.2
kimi-k3    local-cdp     18/18     19230    2.4   33.3
kimi-k3    nous-cloud    12/12     22050    2.9   41.4
kimi-k3    browserbase    6/6      22121    2.8   35.2
```

**Easy battery, round 1** (5 tasks x 3 reps, sonnet-5 + qwen3-coder-30b;
after excluding provider-noise runs — raw chat-template XML, 0 tool calls):

```
model                     arm    ok     prompt  compl   total  calls  wall_s
claude-sonnet-5           base  15/15    39771    324   40095   2.7    16.5
claude-sonnet-5           pr    15/15    27482    509   27991   2.4    14.3
qwen3-coder-30b           base  13/14    59509    559   60068   5.7    21.5
qwen3-coder-30b           pr    10/11    57146   1616   58763   6.8    26.3
```

sonnet-5: −30% tokens at parity. qwen3-30b: a wash — weak coders burn the
savings retrying exec code. The token win concentrates on multi-step tasks
and grows with task hardness; strong models also finish in fewer tool calls.

Compatibility probes from the same run: Firecrawl cloud browsers attach fine
(CDP websocket); Camofox has no CDP surface — structurally incompatible,
hence the automatic fallback to the built-in toolset in #81958.

Caveats: toscrape-family sites (no anti-bot, no heavy SPA); n<=3 per cell;
success-rate deltas at this n are noise — audit sub-100% cells run-by-run
before calling a regression.

## Stagehand single-tool benchmark spike

The `feat/stagehand-single-tool-facade` branch adds a `stagehand` arm that
keeps the exact `browser_exec` tool name and swaps Browser Use's Python CLI
for a JavaScript Playwright-shaped facade executed by Stagehand V4
`experimentalBatch`. It is benchmark wiring, not a production installer:
the arm points at a separately built Stagehand checkout.

Prepare Stagehand V4:

```bash
git clone https://github.com/browserbase/stagehand.git /path/to/stagehand
git -C /path/to/stagehand checkout c6d9baacd5fb668a71a4300436a45ae319660d5c
pnpm --dir /path/to/stagehand install --frozen-lockfile
pnpm --dir /path/to/stagehand --filter @browserbasehq/stagehand build
pnpm --dir /path/to/stagehand --filter @browserbasehq/stagehand-integrations build

# Match the Browser Use CLI used by the reference run. browser-use 0.13.7
# installs browser-harness 0.1.8, whose executable is named browser-use.
uv tool install --force browser-use==0.13.7
```

Run Browser Use 3.0 and Stagehand on the official six-task hard battery using
fresh Browserbase browsers. This example is one model × six tasks × two arms
× three repetitions = 36 trajectories, with no automatic trajectory retry:

```bash
export BUBENCH_BROWSER_USE_TREE=/path/to/hermes-agent-current-main
export BUBENCH_PR_TREE=/path/to/hermes-agent-stagehand-branch
export BUBENCH_STAGEHAND_ROOT=/path/to/stagehand
export BROWSERBASE_API_KEY=...
export BROWSERBASE_PROJECT_ID=...
export BUBENCH_MODEL_PROVIDER=ai-gateway
export AI_GATEWAY_API_KEY=...

# Optional non-billable plan check; must print scheduled_cells: 36.
python3 evals/browser_use/orchestrate_cloud.py \
  --backend browserbase \
  --tasks evals/browser_use/tasks/hard.json \
  --arms pr,stagehand \
  --models anthropic/claude-opus-4.8 \
  --reps 3 \
  --dry-run

python3 evals/browser_use/orchestrate_cloud.py \
  --backend browserbase \
  --tasks evals/browser_use/tasks/hard.json \
  --arms pr,stagehand \
  --models anthropic/claude-opus-4.8 \
  --reps 3 \
  --results evals/browser_use/results/stagehand-vs-browser-use.jsonl

python3 evals/browser_use/report.py \
  evals/browser_use/results/stagehand-vs-browser-use.jsonl
```

The Browser Use arm imports Hermes from `BUBENCH_BROWSER_USE_TREE` and attaches
its CLI to the session provisioned by the orchestrator. The Stagehand arm
imports Hermes from `BUBENCH_PR_TREE` and launches and closes its own
Browserbase session inside the cell subprocess. Both therefore get a fresh
Browserbase browser for every trajectory while preserving their native
execution paths; the baseline is not silently affected by the Stagehand diff.

The commit above is the Stagehand checkout used for the facade smoke test. Pin
both Hermes trees as well (`git rev-parse HEAD`) when publishing results. Live
model serving and Browserbase sessions are nondeterministic, so exact token and
latency values will vary; the task file, arm code, model, repetitions, and
dependency versions remain fixed.

Set `BUBENCH_MODEL_PROVIDER=openrouter` and `OPENROUTER_API_KEY` instead to run
the same arm/task matrix through OpenRouter. Do not mix providers within a
comparison.

## Provenance

The original per-run `results*.jsonl` files lived in `/tmp/bu-bench/` (tmpfs)
and were lost in a host reboot on Aug 12 2026. The harness, task definitions,
and aggregate readouts in this directory were recovered verbatim from the
session transcripts of the benchmark run (session `20260808_050008_5f615e`
tool-call history); `single_run.py`/`orchestrate*.py` are the recovered
scripts with the hardcoded `/tmp/bu-bench` paths parameterized. Rerunning the
battery reproduces fresh per-run data.
