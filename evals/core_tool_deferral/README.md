# core_tool_deferral — live A/B harness for tool-visibility changes

Built for the PR #97979 maintainer battery (core-tool deferral behind the
tool_search bridge). Runs REAL in-process `AIAgent`s from two pinned
checkouts and grades task outcomes programmatically — accuracy, api turns,
tokens, wall, bridge-call counts — across any set of models.

Original verdict + full numbers: `results/SUMMARY.md` and the PR #97979 body
(288 runs; gpt-5.6-terra / glm-5.3-flash / qwen3.8-27b).

## Layout

- `tasks.py` — 14-task battery: one task per deferred tool, multistep
  (todo discipline, GUI chains), long-range (session_search → backup →
  cron → todo), a destructive-ambiguity clarify trap, an eager-only
  control, and a false-discovery distractor. Each task carries fixtures,
  a programmatic grader (0–1 partial credit), and scripted user replies.
- `worker.py` — one (arm, model, task, rep) cell in an isolated
  subprocess: temp HERMES_HOME + workspace, hermetic env (only
  OPENROUTER_API_KEY survives), seeded session DB (targets + decoys),
  deterministic desktop-surface stubs (desktop_ui emitter + agent
  callbacks), computer_use/image_generate stubbed at the registry
  handler. Terminal/files/cron/process/session-DB are REAL.
  Exit 3 = infra/config error (never scored).
- `orchestrator.py` — battery runner: resume-safe, per-task wall
  timeouts, parallel cells, errored-record retry, 3-infra-abort fuse.
- `report.py` — per-task table both arms (score spread, turns, tok, wall,
  bridge calls), mean-of-task-means, noise/error accounting.

## Running

```bash
# 1. Two plain checkouts pinned to the SHAs under test (never pip install -e)
git worktree add /tmp/abdefer-base <baseline-sha>
git worktree add /tmp/abdefer-pr   <pr-sha>

export ABDEFER_BASE_TREE=/tmp/abdefer-base
export ABDEFER_PR_TREE=/tmp/abdefer-pr
export OPENROUTER_API_KEY=...   # the only key the worker keeps

# 2. Smoke one cheap cell first
python3 worker.py base openai/gpt-5.6-terra config_grep_distractor 1 /tmp/smoke.json

# 3. Battery (per model; start with the STRONGEST model to validate variance)
python3 orchestrator.py openai/gpt-5.6-terra 3 --parallel=5
python3 orchestrator.py z-ai/glm-5.3-flash 3 --parallel=5
python3 orchestrator.py qwen/qwen3.8-27b   3 --parallel=5

# 4. Readout
python3 report.py
```

`ABDEFER_PYTHON` overrides the worker interpreter (defaults to the
orchestrator's own); `ABDEFER_RESULTS` overrides the results root.

## Discipline (from the readtool/session_search harness lineage)

- Verify model slugs against the live OpenRouter list before launching.
- Interactive fairness: if the agent ends its turn with a plain-text
  question, the worker sends the scripted reply (max 2, counted as
  `user_roundtrips`) — without this, every clarify-shaped task scores 0
  unfairly and the battery is poisoned (the first terra run was discarded
  for exactly this).
- Same-denominator rule: errored runs score 0 and STAY in the accuracy
  denominator; they are excluded from efficiency means.
- Extend contested cells (score spread at n=3) to n=6 before concluding.
- For discovery-rate regressions, always check base-arm usage on the same
  tasks first — a tool models skip even when visible is not a deferral
  regression.
- Audit anomalous cells from `*.transcript.json` before publishing.

`results/` is gitignored except SUMMARY.md — rep JSONs are rebuildable,
verdicts are the artifact.
