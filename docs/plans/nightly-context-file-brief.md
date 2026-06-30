# Nightly context file brief helper

## Goal

Add a small local-first helper that scans agent context files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `SOUL.md`, `.cursorrules`) for prompt-budget bloat and produces a deterministic morning brief.

## Why

Joe's Hermes workflow depends on durable context, but oversized rule files silently increase prompt cost and make instruction drift harder to review. A read-only helper lets cron or manual runs flag files that need trimming without touching private content or secrets.

## Requirements

- Read-only: never modifies or deletes files.
- Local-first: no network calls or credentials.
- Deterministic output for cron usage.
- Scan a configurable root recursively while skipping common generated/vendor directories.
- Report file status by byte size:
  - `ok`: below warn threshold
  - `warn`: at/above warn threshold and below over threshold
  - `over`: at/above over threshold
- Support Markdown output for humans and JSON output for automation.
- Support exact `[SILENT]` when all discovered files are `ok`.

## Test plan

- Unit tests for classification thresholds and scan exclusions.
- CLI tests for Markdown, JSON, and silent behavior.
- Focused wrapper run: `scripts/run_tests.sh tests/scripts/test_context_file_brief.py`.
