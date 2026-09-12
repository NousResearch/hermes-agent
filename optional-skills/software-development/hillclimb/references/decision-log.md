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

# Decision Log Reference

`.hillclimb/decision.tsv` is the trail: one row per attempt, appended by `<skill_dir>/scripts/decision_log.py append`. It is tab-separated with a required header row and exactly 11 columns, in this order.

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `id` | integer from 1 | Assigned by the tool. Never reused or reordered. |
| 2 | `timestamp` | ISO-8601 UTC | `YYYY-MM-DDTHH:MM:SSZ`, written by the tool. |
| 3 | `hypothesis` | free text | What you expected to happen, and why. |
| 4 | `change` | free text | The single change made, as files or a one-line description. |
| 5 | `before` | float or `na` | The frozen baseline median, or the previous attempt's `after`. |
| 6 | `after` | float or `na` | The median measured for this attempt. |
| 7 | `delta` | float or `na` | Always `after - before`. Computed by the tool. |
| 8 | `tests` | `pass` / `fail` / `none` | Whether the repo's tests still pass. |
| 9 | `verdict` | `kept` / `reverted` | What happened to the change. |
| 10 | `harness_id` | 12 hex chars | Fingerprint of the harness that produced the numbers. |
| 11 | `note` | free text | What the data showed. This is what you read before the next attempt. |

`append` replaces tab and newline characters inside the free-text fields with spaces, so a note containing a tab cannot corrupt the row. `list` skips a malformed row rather than crashing on it, and says so on stderr - `verify` is what names the problem precisely.

## Why `delta` is raw arithmetic

The tool computes `delta` and never accepts it from a caller. Direction is not a property of a row - it is recorded once, in `baseline.json`, as `direction: minimize` or `maximize`, and applied by `stats` and `compare`. For a `minimize` metric (test-suite seconds, bundle KB, tokens per turn) a negative delta is an improvement; for a `maximize` metric (coverage %) a positive delta is. Keeping direction out of the TSV means a row cannot assert a direction the baseline disagrees with.

## Why `tests` has three states

- `pass` - the metric moved and the repo's tests still pass.
- `fail` - the metric moved but at least one test now fails. The change must be reverted even if the number is better. This is the state a boolean would hide.
- `none` - nothing was measured. Use it for category pivots, harness edits, and dead ends.

## Why `harness_id` is on every row

`harness_id` is the first 12 hexadecimal characters of the SHA-256 of the whitespace-normalized harness command. Two rows with different `harness_id` values were produced by different measurement methods and are not comparable, however similar the numbers look. `verify` reports any row whose `harness_id` differs from the frozen baseline as `stale-harness`, which is what makes a silent re-baseline impossible.

## Recording an attempt that measured nothing

Category pivots, harness repairs, and proven dead ends still get a row, with `--before na --after na --tests none`, so the trail explains why the search changed direction rather than appearing to circle. `delta` is recorded as `na` in those rows.

## Worked example rows

Columns are tab-separated; the rows below are aligned for reading. Direction is `minimize` (test-suite seconds), so a negative delta is an improvement.

```
id timestamp             hypothesis                        change            before after delta tests verdict  harness_id   note
1  2026-09-10T14:23:00Z  xdist parallelizes the suite      pytest-xdist x4   5.8    5.2   -0.6  pass  kept     a5b9c2d3e4f7  wall clock down 10%
2  2026-09-10T14:45:00Z  Reworking fixtures removes setup   fixture rework    5.2    6.1   0.9   pass  reverted a5b9c2d3e4f7  slower: fixtures now rebuilt per module
3  2026-09-10T15:02:00Z  Skipping teardown saves time       skip teardown     5.2    5.1   -0.1  fail  reverted a5b9c2d3e4f7  broke test_session_resume
4  2026-09-10T15:30:00Z  Per-test tuning is exhausted; try  na                na     na    na    none  reverted a5b9c2d3e4f7  category pivot: concurrency, not micro-optimization
```

Row 3 is the case the three-state `tests` column exists for: a real 0.1s win, reverted because it broke a test. Row 4 is a pivot, recorded with no measurement so the trail shows why the search moved on.

## `verify` problem types

`verify` prints a JSON list of problems and exits 1 if there are any, 0 if the trail is clean.

| Type | Meaning |
|---|---|
| `no-log` | No `decision.tsv` exists. Nothing to validate; exits 0. |
| `non-numeric-id` | The `id` column is not an integer. |
| `non-monotonic-ids` | Ids are not in increasing order. |
| `duplicate-ids` | The same id appears more than once. |
| `row-wrong-column-count` | The row does not have exactly 11 columns. |
| `delta-mismatch` | `delta` is not equal to `after - before`. |
| `invalid-metric` | The `before` or `after` column is neither a number nor `na`. |
| `invalid-tests` | `tests` is not one of `pass`, `fail`, `none`. |
| `invalid-verdict` | `verdict` is not one of `kept`, `reverted`. |
| `stale-harness` | The row's `harness_id` differs from the frozen baseline's. |

## Committing the trail

`.hillclimb/` is gitignored: the log is a working trail, and most of it is noise to a reviewer. What the reviewer needs is the evidence for the claim, so quote the kept rows - before, after, delta - into the PR body. If a run's reasoning matters more than its numbers, attach the relevant rows to the PR as part of the description; do not commit the directory.
