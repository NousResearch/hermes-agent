# Hermes Architect CLI Specification

## Command

```bash
hermes architect review <project>
```

## Purpose

Review a project before implementation and report whether it has enough architecture to proceed.

## Options

- `--json`: emit machine-readable review output.
- `--scope <categories>`: limit review categories.
- `--block-on-critical`: return a blocked exit code when critical gaps exist.
- `--write-report`: store the architecture report as a Hermes OS artifact.

## Exit Codes

- `0`: pass.
- `2`: warning.
- `3`: blocked.
- `64`: invalid request.

## Output

The command emits an architecture score, critical gaps, missing documents, missing schemas, missing dashboards, missing approvals, automation opportunities, recommendations, and priority roadmap.
