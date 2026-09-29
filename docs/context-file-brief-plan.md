# Context file brief helper plan

## Problem

Hermes loads repo and profile context files such as `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, and `SOUL.md`. These files can silently grow until they consume or truncate useful prompt budget. Joe's nightly workflow needs a small, local-only check that turns context bloat into a reviewable signal without printing private context contents.

## Scope

Build a stdlib-only script at `scripts/context_file_brief.py` that:

1. Deterministically discovers known context files at the workspace root and `.hermes/` subdirectory.
2. Reports character count, line count, budget ratio, and status (`ok`, `warn`, `over`).
3. Emits actionable cleanup guidance for `warn` / `over` files.
4. Supports JSON output for automation.
5. Supports exact `[SILENT]` when `--silent-ok` is set and every discovered file is OK.

## Non-goals

- Do not print file contents.
- Do not rewrite, split, or delete context files.
- Do not inspect secrets or expand data access.

## Acceptance tests

- Discovery includes root + `.hermes/` context files and ignores unrelated files.
- Oversized files produce `over` and a specific split-to-skills/references suggestion.
- `--silent-ok` emits exactly `[SILENT]` only when all discovered files are OK.
- JSON output uses stable relative paths.
