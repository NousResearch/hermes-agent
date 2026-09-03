# Operations Reference

Use this file when exact commands or recovery rules are needed. The short operating contract remains in `SKILL.md`.

## Variables

```text
SKILL_DIR     installed x-analytics-import directory
OVERVIEW      exact account overview CSV path
CONTENT       exact content CSV path
VIDEO         exact compatible video CSV path, optional
CONFIG        private local config path, optional
OUTPUT        private output directory outside repositories
```

Default output when `--output-dir` is omitted:

```text
~/.hermes/data/x-analytics-import
```

## First baseline

### Inspect

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" inspect   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --json
```

Confirm export kinds, row counts, date coverage, unknown columns, and source hashes. Omit optional arguments when absent.

### Validate

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" validate-only   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --json
```

Exit code `2` means validation failed. Do not continue.

### Dry-run

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" dry-run   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --output-dir "$OUTPUT" --json
```

Verify `writes` is empty and reconciliation is not failed.

### Create baseline

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" full-import   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --output-dir "$OUTPUT" --json
```

Verify the manifest, snapshot, and five reports. Record the import ID.

### Prove idempotency

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" incremental-import   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --output-dir "$OUTPUT" --json
```

Expected status: `already_imported`.

## Recurring refresh

Repeat inspect, validate-only, and dry-run with the new exact files. Then:

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" incremental-import   --overview "$OVERVIEW" --content "$CONTENT" --video "$VIDEO"   --config "$CONFIG" --output-dir "$OUTPUT" --json
```

When status is `complete`, compare the previous and new snapshots:

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" compare-snapshots   --snapshot-a "$OUTPUT/snapshots/<previous-id>.json"   --snapshot-b "$OUTPUT/snapshots/<new-id>.json"   --json
```

Use the previous completed snapshot for routine change detection. Use the original baseline only as a second long-range comparison.

## Classification rebuild

Copy the example config to a private path, change the lane rules, then run:

```bash
python "$SKILL_DIR/scripts/x_analytics_import.py" rebuild-classification   --snapshot "$OUTPUT/snapshots/<import-id>.json"   --config "$CONFIG" --output-dir "$OUTPUT" --json
```

This must preserve imported metrics and source attribution.

## Recovery

| Condition | Action |
|---|---|
| Validation failed | Correct file selection or schema issue; rerun from inspect |
| Reconciliation failed | Verify the overview/content pair and date coverage |
| `already_imported` | Stop; the source set is already represented |
| Interrupted write | Rerun the same import; no completed manifest should exist |
| Lane error | Change private config and rebuild classification |
| Public-safe leakage guard failed | Stop; do not copy or publish the report |
| Video missing | Continue without `--video` |

Never recover by deleting manifests, editing snapshots manually, or using `--force` without explicit user authorization.
