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

# Hillclimb

Improves one quantitative metric of a codebase by making a single change, measuring it, and keeping or reverting it on the evidence. The loop is deliberately narrow: one metric, one change per attempt, one measurement, one log row. It does not profile code, diagnose correctness bugs, or weigh competing objectives - see When to Use for what to reach for instead.

## When to Use

Reach for this skill when the target is a single number and the user wants it moved:

- "Make the test suite faster", "get bundle size down", "cut token cost per turn", "reduce the flaky-test rate", "raise coverage".
- A change is about to be justified by intuition or by reading the code, and you would rather the data decide.
- A tuning effort will run many attempts, possibly with nobody watching.

Do NOT use it when:

- You cannot build a harness that reliably prints one number. Without a repeatable measurement this skill has nothing to offer - say so and stop, rather than improvising a metric that flatters the change.
- The target is a correctness bug rather than a measurement: use `systematic-debugging`.
- The work is code shape with no metric attached: use `simplify-code`.
- The goal is a tradeoff across several metrics with no dominant one. Choose the metric first, then come back.

## Prerequisites

- A target repo under git, and one command that prints the metric (the harness).
- `<skill_dir>` below means this skill's own directory; the helper scripts are `<skill_dir>/scripts/decision_log.py` and `<skill_dir>/scripts/sample_metric.py`.
- Python 3, standard library only. The scripts create `.hillclimb/` in the working directory on their first write; `--dir`, or the `HILLCLIMB_DIR` environment variable, relocates it.
- Add `.hillclimb/` to the target repo's `.gitignore`. The log is a working trail, not a deliverable; it is quoted into the PR body rather than committed.

## How to Run

1. Build the harness, then freeze a baseline with `sample_metric.py baseline`.
2. Make one change, then measure it with `sample_metric.py compare`.
3. Record the attempt with `decision_log.py append`.
4. When progress stalls, run the plateau protocol in `references/plateau-playbook.md`.
5. Land the surviving change and quote the kept rows in the PR body.

## Quick Reference

| Command | Purpose |
|---|---|
| `sample_metric.py baseline --harness CMD --samples N` | Sample N times, freeze the median as the baseline |
| `sample_metric.py run --harness CMD --samples N` | Sample without touching the baseline |
| `sample_metric.py compare --harness CMD` | Exit 0 = improvement, 1 = not, 2 = no baseline or bad value, 3 = harness changed since the freeze |
| `decision_log.py append ...` | Add one attempt row; the tool computes `delta` |
| `decision_log.py list [--limit N] [--verdict kept\|reverted]` | Read the trail, oldest first |
| `decision_log.py stats [--window N] [--threshold F]` | Direction-aware best/worst plus a plateau verdict |
| `decision_log.py verify` | Validate the trail: exit 0 clean, 1 problems, 2 bad invocation |

`--extract` selects how a number is read out of the harness's stdout: `auto` (default - the last number printed), `regex:PATTERN` (first capture group), `json:dotted.path`, or `line:PREFIX` (the first number on the first line beginning with PREFIX).

## Procedure

### 1. Build the harness

Write one command that prints exactly one number, then confirm it is stable across runs. Use enough samples to clear noise: the median of N runs, never a single run.

**Done when:** repeated runs of the harness land within a spread you are willing to call noise, and `sample_metric.py run` prints a median for that command.

### 2. Freeze the baseline

```bash
python3 <skill_dir>/scripts/sample_metric.py baseline --harness "<command>" --samples 5 --name "test suite seconds" --direction minimize
```

This writes `.hillclimb/baseline.json` with the recorded median, `frozen: true`, and a `harness_id` fingerprint of the command. Add `.hillclimb/` to `.gitignore` now. The harness is immutable from this point: any edit changes the `harness_id`, and every earlier number stops being comparable.

**Done when:** `.hillclimb/baseline.json` exists, is `frozen: true`, and a repeat run of the same harness reproduces its median within the sampling spread.

### 3. The loop (one change, one measurement)

Make exactly one change to the codebase - never to the harness - then:

```bash
python3 <skill_dir>/scripts/sample_metric.py compare --harness "<command>" --samples 5
```

Record the attempt either way:

```bash
python3 <skill_dir>/scripts/decision_log.py append \
  --hypothesis "Parallel test execution cuts wall clock" \
  --change "pytest-xdist, 4 workers" \
  --before 5.8 --after 5.2 --tests pass --verdict kept
```

`delta` is always raw `after - before`, so here it is -0.6. For a `minimize` metric a negative delta is an improvement; the direction is recorded once in `baseline.json`. When the metric moves the wrong way, record the same shape with the real numbers and `--verdict reverted` - for example `--before 5.8 --after 6.1 --verdict reverted`. Never stack untested changes, and never record a win you did not measure.

If an attempt measured nothing - a category pivot, a harness fix, a proven dead end - record it with `--before na --after na --tests none` so the trail still explains the search.

**Done when:** exactly one new row exists with a tool-computed `delta`, a `tests` value, and a `verdict`, and the change was kept or reverted to match.

### 4. Plateau

Run `decision_log.py stats`. It reports a plateau when **each** of the last `--window` attempts (default 3) improved by less than `--threshold` (default 0.02, i.e. 2%), where improvement is relative to that attempt's own `before` value and interpreted by the recorded direction. `plateau_reasons` names the attempts that failed to clear the bar.

A plateau is a signal to change hypothesis *category*, not a signal to stop. Follow `references/plateau-playbook.md`: pivot category, combine the near-misses, re-read the source instead of guessing, and try something more radical before concluding the hill is climbed.

**Done when:** the plateau is either broken by a recorded category pivot, or written down as a provable dead end in a `tests: none` row.

### 5. Land it

```bash
python3 <skill_dir>/scripts/decision_log.py verify
```

`verify` must exit 0. Then re-run the harness and the target repo's own test suite against the surviving tree, commit the change, and quote the kept rows (before/after/delta) into the PR body. Never commit `.hillclimb/`.

**Done when:** `verify` exits 0, the repo's tests pass on the surviving tree, and the PR body carries the kept rows.

## Two Modes

- **Interactive** - one agent, one change at a time, measuring and keeping or reverting as it goes.
- **Unattended or parallel** - fan hypotheses into separate git worktrees, or run the loop on a schedule. See `references/unattended-mode.md` for the real `cronjob_manage` fields, the monitor gate, and the worktree recipe.

## Pitfalls

1. Editing the harness after the freeze invalidates every earlier number. `compare` will exit 3 rather than compare across two measurement methods - re-baseline deliberately and record why.
2. Re-baselining to escape a result you did not like is the failure this skill exists to prevent. The `harness_id` guard makes it impossible to do silently, which is the point: the escape has to be written down.
3. A failed or unparseable sample aborts the whole measurement (exit 2). There is deliberately no way to average over failed runs; a metric built from failed samples is not a metric.
4. Reading the code and concluding a change helped is not evidence. The data decides, or the attempt is recorded as `tests: none` with no measurement.
5. A plateau is not the end of the hill. Pivoting category, combining near-misses, and re-reading the source are all still available.
6. Skipping `verify` lets a malformed trail - a hand-edited delta, a row from a different harness - reach the PR body.
7. Forgetting `.gitignore` puts the working trail into the commit.

## Verification

- [ ] `.hillclimb/baseline.json` exists with `frozen: true` and a `harness_id`
- [ ] Every attempt has exactly one row, with a tool-computed `delta` and an explicit `tests` value
- [ ] Kept rows have real before/after numbers taken from `compare`, not from inspection
- [ ] `decision_log.py verify` exits 0
- [ ] `verify` reports no `stale-harness` rows, so every number came from the same measurement method
- [ ] `.hillclimb/` is not in the commit, and the kept rows are in the PR body
