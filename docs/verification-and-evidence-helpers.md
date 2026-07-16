# Verification and evidence helper scripts

This repo keeps LazyCodex/GJC-style workflow improvements at the edge of Hermes:
scripted helpers, tests, and documentation instead of new core model tools.

## Changed-path verification bundle

Use `scripts/suggest_verification_bundle.py` to turn changed paths into a
copyable list of recommended local checks.

Examples:

```bash
python scripts/suggest_verification_bundle.py --paths gateway/run.py apps/desktop/src/store/layout.ts
git diff --name-only | python scripts/suggest_verification_bundle.py --stdin
python scripts/suggest_verification_bundle.py --stdin < changed_paths.txt
python scripts/suggest_verification_bundle.py --from-git --base origin/main --format json
```

The helper is read-only. It does not execute checks, read file contents, inspect
secrets, or mutate runtime state. It disables import bytecode writes and writes
rendered output only to stdout; use shell redirection if a report file is needed.
Changed paths are canonicalized before classification, so traversal-like spellings
cannot downgrade focused checks. Unknown or empty path inputs fail open by
recommending broader checks instead of skipping work. The suggested added-line
security command scans staged, unstaged, and untracked files with redacted-only
findings. It fails closed on binary or undecodable additions and refuses to read
untracked files through symlink, reparse/junction, hardlink, or out-of-repository
paths. Tracked scans disable external diff, textconv, color, and customized diff
indicators before parsing Git output. The generated scanner binds sanitized Git
discovery to the current repository root, while `--from-git` explicitly decodes
Git path output as UTF-8 and ignores repository-selection environment overrides.
Active Git filter attributes fail closed before any clean filter can execute, and
every generated Git call has a bounded timeout. NUL-bearing path output and raw
Git failure text are rejected behind stable sanitized errors. Assume-unchanged,
skip-worktree, sparse, and other nonstandard tracked-index tags fail closed;
fsmonitor and replace-object behavior are disabled for helper Git calls. Active
clean filters, `ident`, working-tree encodings, and gitlink/submodule entries fail
closed before transformed content can hide a worktree change. Run generated
commands from the repository root. Executing the suggested scan is a separate
explicit verification step.

## ULW evidence ledger scaffold

Use `scripts/scaffold_ulw_ledger.py` to create a durable evidence directory for
long-running ULW/Fleet work.

```bash
python scripts/scaffold_ulw_ledger.py \
  --run-id hq-example-20260629 \
  --goal "Implement and verify a bounded Hermes improvement"
```

Default root:

```text
$HERMES_HOME/reports/ulw-loop/<run-id>/
```

If `HERMES_HOME` is unset, the helper still avoids Hermes runtime/profile
inspection and falls back directly to platform-local paths:

```text
Windows: %LOCALAPPDATA%/hermes/reports/ulw-loop/<run-id>/
Other:   ~/.hermes/reports/ulw-loop/<run-id>/
```

Generated files:

```text
brief.md
goals.json
ledger.jsonl
evidence/README.md
```

Safety properties:

- `--dry-run --format json` reports intended paths without writing files.
- Existing run IDs are refused unless `--force` is passed.
- Path separators and traversal-like run IDs are rejected.
- `--force` refuses symlinks, Windows junction/reparse points, and hardlinked
  scaffold files; normal replacements use same-directory atomic writes instead
  of truncating an existing target in place.
- The helper does not read raw memory, credential stores, `.env`, OAuth files,
  or Hermes config.

## Review rule

These helpers produce suggestions and scaffolds only. A reviewer must still
read real command output, file readbacks, and evidence artifacts before marking
work complete.
