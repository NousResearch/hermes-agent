# Local Workspace — Agent Instructions

## If you see clutter at repo root

1. Check `.gitignore` — the file is probably intentional local scratch.
2. Read [`README.md`](README.md) category table before moving or deleting.
3. **Never** `git add` logs, media, secrets, or `_docs/` unless the operator explicitly requests it.

## Moving files

Prefer moving scratch into existing ignored dirs:

- `output/` for generated media
- `tmp/` for short-lived experiments

Use these stable subdirectories when classifying root scratch:

- `output/media/` for audio, video, and image renders
- `output/reports/` for JSON, HTML, text, and run summaries
- `output/logs/` for generated logs that are not implementation records
- `tmp/probes/` for one-off scripts and diagnostics
- `tmp/snapshots/` for generated source or configuration snapshots

Create the destination before moving an existing file and verify that the
source and destination are inside this repository. Move, do not delete, local
scratch. Preserve symlinks as symlinks; do not dereference a link while
classifying it, and never move a link target outside the repository.

The root `AGENTS.md` remains the required entrypoint. This file supplies the
safe local-workspace rules; it must not be copied into runtime directories or
used to make ignored output look like source code.

Update any hardcoded paths in `scripts/daily_*.py` only when the operator requests relocation.

## Probes and tests

- Do not leave `test_*.py` at repo root — use `tests/` with `scripts/run_tests.sh`.
- Delete or gitignore one-off probes after use.

## Harness safety

Root-level scratch must **not** be referenced from `scripts/merge_tools/` policy or overlay paths. Overlays target stable paths under `plugins/`, `tools/`, `scripts/`.

## Identity files

`SOUL.md`, `brain/*`, and sovereign identity files may exist locally; `brain/*` is `preserve_custom` in merge policy. Do not publish private identity content upstream.
