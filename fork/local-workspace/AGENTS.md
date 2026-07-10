# Local Workspace — Agent Instructions

## If you see clutter at repo root

1. Check `.gitignore` — the file is probably intentional local scratch.
2. Read [`README.md`](README.md) category table before moving or deleting.
3. **Never** `git add` logs, media, secrets, or `_docs/` unless the operator explicitly requests it.

## Moving files

Prefer moving scratch into existing ignored dirs:

- `output/` for generated media
- `tmp/` for short-lived experiments

Update any hardcoded paths in `scripts/daily_*.py` only when the operator requests relocation.

## Probes and tests

- Do not leave `test_*.py` at repo root — use `tests/` with `scripts/run_tests.sh`.
- Delete or gitignore one-off probes after use.

## Harness safety

Root-level scratch must **not** be referenced from `scripts/merge_tools/` policy or overlay paths. Overlays target stable paths under `plugins/`, `tools/`, `scripts/`.

## Identity files

`SOUL.md`, `brain/*`, and sovereign identity files may exist locally; `brain/*` is `preserve_custom` in merge policy. Do not publish private identity content upstream.
