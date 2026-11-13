---
name: x-analytics-import
description: Use when X Analytics CSV exports must be inspected, validated, normalized, imported, or compared through a repeatable private-by-default workflow.
version: 1.0.0
author: Tony Simons
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: data-analysis
    tags: [x, analytics, csv, validation, statistics, privacy]
    related_skills: []
---

# X Analytics Import

## Overview

Use this skill to turn X Analytics CSV exports into validated, normalized, private local snapshots. The deterministic engine is `scripts/x_analytics_import.py`; it uses only the Python standard library and has no network, browser, Git, or publishing side effects.

The operating rule is simple:

```text
first run: inspect -> validate-only -> dry-run -> full-import
later runs: inspect -> validate-only -> dry-run -> incremental-import -> compare
```

Read `references/operations.md` only when command detail or recovery guidance is needed.

## When to Use

Use this skill when:

- Overview, content, or video analytics CSVs are available locally.
- A first normalized baseline must be established.
- New exports must be added without duplicating prior imports.
- Two snapshots must be compared using matched coverage.
- Lane rules changed and historical posts must be reclassified.

Do not use this skill when:

- The exports still need to be downloaded from X.
- The task is to post, reply, follow, message, or modify an X account.
- The data comes from GA4, Search Console, or another analytics product.
- The user wants raw exports copied into Git or published.

## Safety Contract

1. Treat raw exports, normalized rows, post text, post links, source hashes, revenue fields, and lane results as sensitive.
2. Select exact files by path, date coverage, and modification time. Never trust alphabetical glob order.
3. Run `inspect`, `validate-only`, and `dry-run` before every write mode.
4. Store outputs outside repositories, normally under `~/.hermes/data/x-analytics-import`.
5. Never copy, rename, overwrite, stage, commit, or publish raw CSVs.
6. Never use `--force` unless the user explicitly requests an identical re-import.
7. A successful import does not authorize updates to notes, dashboards, websites, or Git.

## Modes

| Mode | Purpose | Writes |
|---|---|---:|
| `inspect` | Detect export type, schema, hash, rows, and coverage | No |
| `validate-only` | Run structural and semantic checks | No |
| `dry-run` | Execute the complete pipeline without artifacts | No |
| `full-import` | Create the first baseline or deliberate rebuild | Yes |
| `incremental-import` | Add a new source set only when hashes differ | Yes |
| `compare-snapshots` | Compare matched dates and post IDs | No |
| `rebuild-classification` | Reapply lane rules without reparsing CSVs | Yes |

## Workflow

### 1. Identify the exact export set

Expected file families:

```text
account_overview_analytics*.csv
account_analytics_content_<start>_<end>.csv
video_overview_analytics*.csv
```

Video is optional. Confirm the files belong to the intended export run. Chrome suffixes such as `(1)` are not chronology.

Completion criterion: every selected file has an exact path, plausible coverage, and no stale duplicate has been substituted.

### 2. Choose the run type

- No completed manifest exists: follow **First baseline**.
- A completed manifest exists and the exports are newer: follow **Recurring refresh**.
- The same source hashes already exist: stop at `already_imported`.

Completion criterion: the agent can name the prior import ID or state that no baseline exists.

### 3. First baseline

Run, in order:

```text
inspect
validate-only
dry-run
full-import
```

Use the same exact file paths for all four stages. Stop before `full-import` when validation fails, reconciliation fails, or file pairing is doubtful.

After `full-import`, verify:

- A completed manifest and normalized snapshot exist.
- Validation, reconciliation, classification, analysis, and public-safe reports exist.
- Manifest hashes match the selected inputs.
- Date coverage and partial-day status are recorded.
- Re-running the same files with `incremental-import` returns `already_imported`.

Completion criterion: one verified baseline exists and duplicate protection is proven.

### 4. Recurring refresh

Run, in order:

```text
inspect
validate-only
dry-run
incremental-import
compare-snapshots
```

Use `incremental-import` for ordinary future runs. If it returns `already_imported`, do not force a duplicate and do not create a comparison.

Compare the new snapshot against the previous completed snapshot. Use the original baseline only for a separate long-range view.

Comparison rules:

- Use overlapping overview dates.
- Use matched post IDs for post-level deltas.
- Report new and removed post IDs separately.
- Do not subtract whole rolling-window totals when coverage differs.

Completion criterion: a new import is either safely rejected as duplicate or written once and compared on matched coverage.

### 5. Interpret the result

Use medians, p25, p75, p90, sample sizes, and IQR outliers. Separate originals, replies, quotes, reposts, and unknown types. Treat CSV-only post type as heuristic. Preserve `Uncertain` lane results rather than forcing a winner.

Exclude detected partial dates from default statistics. State confidence and source coverage. Do not claim causation from correlation.

Completion criterion: every recommendation includes evidence, sample size, confidence, and outlier context.

### 6. Report without leaking data

Return operational status first:

- selected filenames
- import status and import ID
- validation and reconciliation status
- partial-day status
- artifact paths
- idempotency result
- warnings requiring attention

Exact metrics, post content, source hashes, revenue, and strategy findings remain local unless the user explicitly requests them.

Completion criterion: the response is useful without exposing sensitive values by default.

## Command Pattern

```bash
python "<skill-dir>/scripts/x_analytics_import.py" <mode>   --overview "<overview.csv>"   --content "<content.csv>"   --video "<video.csv>"   --config "<local-config.json>"   --output-dir "<private-output-dir>"   --json
```

Omit `--video` or `--config` when unused. Copy `references/config.example.json` to a private local path before customizing lane rules.

Snapshot comparison:

```bash
python "<skill-dir>/scripts/x_analytics_import.py" compare-snapshots   --snapshot-a "<older.json>"   --snapshot-b "<newer.json>"   --json
```

## Failure Gates

Stop before a write mode when:

- Required columns are missing.
- Dates or numeric values are malformed.
- Duplicate post IDs are present.
- Content extends beyond overview coverage.
- Files appear to come from different export runs.
- The output path is inside a repository.

Warnings such as an X filename window starting before the first actual content row may be acceptable when actual row coverage and end dates align. Explain the warning rather than silently discarding it.

## Common Pitfalls

1. **Using `full-import` every time.** Use it once for the baseline; use `incremental-import` afterward.
2. **Selecting the alphabetically last download.** Inspect exact paths, timestamps, hashes, and coverage.
3. **Assuming the newest day is complete.** Use recorded partial-day detection.
4. **Comparing unlike windows.** Compare overlapping dates and matched post IDs.
5. **Treating content metrics as daily account totals.** Content rows are keyed by publish date and are not directly comparable to daily overview totals.
6. **Forcing lane or post-type certainty.** Preserve uncertainty and confidence labels.
7. **Publishing the local analysis report.** Use the public-safe report as the disclosure starting point.
8. **Using real exports as fixtures.** Tests must remain synthetic.

## Verification Checklist

- [ ] The exact overview and content paths are recorded
- [ ] Video is included only when compatible and current
- [ ] `inspect`, `validate-only`, and `dry-run` completed first
- [ ] Baseline uses `full-import`; later runs use `incremental-import`
- [ ] Validation has no errors
- [ ] Reconciliation has no unresolved failure
- [ ] Output is outside every repository
- [ ] Manifest hashes match the selected sources
- [ ] Partial-day status is recorded
- [ ] Duplicate re-import returns `already_imported`
- [ ] Comparisons use matched coverage
- [ ] Public-safe output excludes sensitive values
- [ ] Raw CSVs were not copied, staged, committed, or published
