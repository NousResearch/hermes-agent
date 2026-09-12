---
name: hillclimb
description: "Optimize one metric via frozen harness and decision log."
version: 1.0.0
author: "Emmanuel Ketcha (@ketchalegend) + Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [optimization, metrics, experiments, decision-log, benchmarking, iteration]
    category: software-development
    related_skills: [systematic-debugging, spike, test-driven-development, requesting-code-review, subagent-driven-development]
---

# Unattended Mode Reference

Running the loop without a human watching. Two patterns: live fan-out across worktrees, and scheduled runs through the `cronjob_manage` tool. Both rely on the harness staying byte-identical - the whole point of the loop is that every number came from the same measurement method.

## Live Fan-Out (parallel hypotheses)

Give each independent hypothesis its own git worktree so parallel attempts cannot collide, and its own `.hillclimb/` directory so the logs stay separate.

```bash
# One worktree per hypothesis
git worktree add ../repo-hyp-1 -b hyp-1
git worktree add ../repo-hyp-2 -b hyp-2

# In each worktree, freeze the SAME baseline, then make that worktree's one change
cd ../repo-hyp-1 && python3 <skill_dir>/scripts/sample_metric.py baseline --harness "<command>" --samples 5
```

Rules that make the comparison valid:

- The harness command must be identical across worktrees. Same command means the same `harness_id`, which is what makes the medians comparable.
- Dispatch the hypotheses with `delegate_task` (parallel `tasks`), giving each child its own worktree path and its own `.hillclimb/` directory. Children cannot ask questions mid-flight, so put the hypothesis, the exact harness command, and the required reporting format in each child's context.
- Supervise the diff instead of accepting a summary: read each child's `decision.tsv` with `read_file` and compare `delta` values. A child's report about its own numbers is a claim, not evidence.
- Apply only the winning change to the main tree and re-measure there. A win in a scratch worktree is not yet a win in the real checkout.

## Scheduled Runs (`cronjob_manage`)

An agent schedules with the `cronjob_manage` tool (`action='create'`). A human uses the CLI: `hermes cron create '<schedule>' '<prompt>' [flags]`.

| Field | Flag | Meaning |
|---|---|---|
| `script` | `--script` | Path to a script under the Hermes home's `scripts/` directory. Default mode: its stdout is injected into the agent's prompt each run. With `no_agent=True` the script is the whole job. Relative paths resolve under `<hermes_home>/scripts/`. |
| `monitor` | `--monitor-script`, `--monitor-url` | Cheap change-detector run each tick before the agent. Output byte-identical to the previous tick skips the agent run entirely; changed output wakes the agent with a diff injected into the prompt. Must be deterministic - no timestamps - or every tick looks changed. Incompatible with `no_agent`. |
| `no_agent` | `--no-agent` | Skip the LLM: run the script on schedule and deliver its stdout verbatim. Empty stdout sends nothing. |
| `continuity` | `--continuity` | Each run wakes with the job's own previous output injected, so it continues where the last run stopped. |
| `context_from` | tool field | Injects the most recent output of *other* jobs. This chains a collector job into a processor job; for a job's own history use `continuity` instead. |
| `workdir` | `--workdir` | Absolute existing path to run the job from. Injects that directory's `AGENTS.md` and anchors the file and terminal tools there. This is how the job reaches the target repo. |
| `skills` | `--skill` | Attach a skill; repeatable. Attach this one. |
| `enabled_toolsets` | tool field | Restrict the job's agent to a small toolset to cut per-run token overhead. |
| `schedule` | positional | `'30m'`, `'every 2h'`, `'0 9 * * *'`, or an ISO one-shot. `--repeat` sets a count. |

**The thing readers get wrong:** `--script` cannot point into the target repo. Copy the sampler into the Hermes home's `scripts/` directory (for example as `hillclimb_sample.py`) and pass the repo through `--workdir`. A repo-relative script path does not resolve.

**The monitor gate is the point of the design.** A frozen harness already emits a deterministic, cheap number - exactly the signal that should decide whether to spend a turn at all. Point `monitor` at the sampler: while the metric is unchanged, no turn is spent.

**Timing, and what follows from it.** The script's own timeout defaults to 3600s. The agent turn runs under an inactivity watchdog rather than a wall-clock cap: 600 seconds of no activity by default, overridable with the `HERMES_CRON_TIMEOUT` environment variable, `0` for unlimited. So put the expensive measurement in the script - deterministic, no LLM - and keep the agent turn to reading the log, interpreting the result, and appending one row.

**Scope.** A job runs in a fresh session with no chat context. A one-shot fires once; if the loop is unfinished the job must reschedule itself (`'in 10m'`) or be created recurring. Background `delegate_task` work is process-local and does not survive a restart - anything that must outlive a restart is a cron job or a `terminal(background=True, notify_on_complete=True)` process.

### Shape of the job

Two jobs, because sampling and deciding fail differently:

1. **Sampler** - `no_agent=True`, runs the harness and prints the metric. Silent when nothing changed.
2. **Decider** - `monitor` pointed at the sampler's deterministic output (so it wakes only when the number moves), `continuity=True` to carry the trail forward, this skill attached via `skills`, and `workdir` set to the target repo. It reads `decision.tsv`, measures, decides, and appends exactly one row.

## Safety rails for unattended runs

- **Verify before appending.** Run `decision_log.py verify` first. If it reports problems, stop and report rather than appending to a trail that is already corrupt.
- **Revert on a failed test** even when the metric improved: `tests: fail` means revert. A faster suite that no longer passes is not a win.
- **Stop on a proven dead end.** When `stats` reports a plateau and the pivot categories are exhausted, record the dead end as a `tests: none` row and stop. Churning attempts past a provable floor burns budget and produces nothing.
- **Never trust a summary.** A job or child reporting its own improvement is a claim; the evidence is the row in `decision.tsv` with a tool-computed delta.
