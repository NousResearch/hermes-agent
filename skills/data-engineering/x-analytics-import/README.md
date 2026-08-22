# X Analytics Import

A field-tested Hermes skill for validating, normalizing, importing, and comparing X Analytics CSV exports.

## Why it exists

X exports contain practical traps: duplicate browser filenames, rolling content windows, partial current days, multi-section video files, changing headers, heuristic post types, and metrics with different scopes. This skill turns that work into a deterministic local pipeline instead of a one-off spreadsheet ritual.

The public bundle was derived from a repeatedly used private workflow. It contains no real exports, account history, personal lane rules, private paths, source hashes, credentials, or unpublished strategy.

## What it does

- Detects overview, content, and video schemas
- Validates columns, dates, numeric cells, duplicates, and coverage
- Normalizes rows into versioned snapshots
- Detects likely partial current days
- Classifies post type with method and confidence
- Applies optional deterministic lane rules
- Calculates robust statistics and IQR outliers
- Prevents duplicate imports with SHA-256 manifests
- Compares snapshots on matched dates and post IDs
- Generates local analysis and public-safe methodology reports

## Requirements

- Python 3.11 or newer
- No third-party Python packages
- Local X Analytics CSV exports

## Install

Install through Hermes Field Kit as a tap using the command supported by your Hermes version, or copy `skills/x-analytics-import` into your local Hermes skills directory. Start a new Hermes session after installation because skill discovery may be cached.

Manual install from a clone:

```bash
cp -R skills/x-analytics-import ~/.hermes/skills/
```

PowerShell:

```powershell
$destination = Join-Path $env:LOCALAPPDATA "hermes\skills"
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item -Recurse "skills\x-analytics-import" $destination
```

## First run

The first completed import establishes the baseline:

```text
inspect -> validate-only -> dry-run -> full-import
```

Then run the same source set once with `incremental-import`. The expected result is `already_imported`, proving idempotency.

See [First baseline](examples/first-baseline.md).

## Later runs

Every new export set follows:

```text
inspect -> validate-only -> dry-run -> incremental-import -> compare-snapshots
```

Compare against the immediately previous completed snapshot using overlapping dates and matched post IDs. See [Recurring refresh](examples/recurring-refresh.md).

## Configuration

The importer works without a config file. With no lane rules, posts fall into `Unclassified`.

To customize behavior, copy [config.example.json](references/config.example.json) to a private local path and edit that copy. Do not put personal rules or private thresholds into a public repository.

## Inputs

- Account overview CSV
- Content analytics CSV
- Optional video overview CSV

Current X video exports may contain a daily overview table followed by a separate `Your videos` table. The importer reads the daily overview section and stops at the next section boundary.

## Outputs

Write modes create:

```text
<output-dir>/
├── manifests/
├── snapshots/
└── reports/
```

Reports include validation, reconciliation, classification, local analysis, and a public-safe methodology summary. Raw CSVs are never copied.

## Privacy

Raw rows, normalized rows, post text, post links, source paths, hashes, lane findings, and revenue fields are sensitive by default. The public-safe report excludes exact analytics and content.

The tool never uploads data, calls a network service, updates Git, or publishes results.

## Limitations

- Post type is heuristic unless the export gains an authoritative type column.
- Lane quality depends on user-defined rules.
- Content and overview metrics have different scopes and are not forced into false equality.
- A public-safe report is a disclosure aid, not automatic publication permission.
- The importer does not download exports from X.

## Testing

```bash
python -B -m unittest discover -s skills/x-analytics-import/tests -v
python scripts/validate.py
python -m unittest discover -s tests -v
```

All importer fixtures are synthetic.

## Version history

### 1.0.0

- Initial public release
- Baseline and recurring import workflows
- Idempotent manifests and snapshots
- Schema validation, partial-day detection, reconciliation, classification, robust statistics, comparison, and privacy guards
