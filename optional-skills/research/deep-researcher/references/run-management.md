# Run state, storage, and recovery

Use a new dedicated directory per run, outside the skill package. Replace SKILL and RUN
with actual absolute paths; do not assume shell variables survive across tool calls.

```sh
python3 SKILL/scripts/research_run.py init RUN --question "Research question" --scope "Audience, timeframe, exclusions" --mode standard --storage-mb 64
```

The manifest records assumptions, effective search/storage budgets, completed work, gaps,
next action, and an explicit artifact inventory. `--search-budget` overrides the mode's
default; repeat `--assumption` as needed. It contains no automatic environment/config dump.

After initializing the ledger and collecting evidence:

```sh
python3 SKILL/scripts/research_run.py checkpoint RUN --artifact evidence.json --artifact evidence-brief.md --completed "Q1 evidence collected" --gaps "Q2 lacks independent evidence" --next-action "Retrieve the original Q2 study"
python3 SKILL/scripts/research_run.py status RUN
```

Artifact arguments refer to existing individual files relative to RUN, not directories,
globs, external tool caches, or symlinks. Repeat `--artifact` for any intentionally retained
source text or search log. Completed/gaps arguments replace their respective lists;
omitting them preserves the previous state. `--clear-gaps` clears resolved gaps.
Use the native file editor to revise scope, assumptions, or budgets in the manifest.

To resume, read status and the brief first, then the ledger only as needed. Inspect whether
time-sensitive evidence needs refreshing. Reuse completed work; do not rerun all searches.
The manifest is a checkpoint, not an automatic search scheduler or spending meter.

## Prevent accumulation

- Record compact facts once. The ledger is authoritative; the brief is a small derived view.
- Brief output is capped at 24,000 characters by default, with an explicit truncation notice.
  Truncation never changes stored evidence. Read omitted relevant records before drawing conclusions.
- Keep one working report and one brief. Use a final report plus archive for retention rather
  than appending a new full report after every iteration.
- Default to excerpts and locators. Retain full raw text only when it serves verification;
  avoid downloaded media, screenshots, PDFs, and duplicate HTML unless the task needs them.
- Check registered storage after each batch. The 64 MiB default is a warning threshold,
  not a hard cap. It does not include unregistered files or backend-managed caches.
- Do not delete host caches or other research runs. Their retention is separately managed.
- Store runs outside the distributable repository; do not commit reports or archives by default.

## Complete and archive

Finish the semantic and structural report audit before marking complete. Run completion
checks that report.md is nonempty; it cannot verify that a human/agent performed the audit.

```sh
python3 SKILL/scripts/research_run.py checkpoint RUN --status complete --artifact report.md --artifact evidence.json --artifact evidence-brief.md
python3 SKILL/scripts/research_run.py archive RUN --prune
```

This creates artifacts.zip using lossless LZMA compression when available. If the Python build
lacks LZMA, it uses deflate; if neither codec is available, it stores files in an uncompressed ZIP.
No dependency installation is needed. The command reports the chosen method. Restoring an
archive on another machine requires support for the codec that created it.
Text-heavy evidence often compresses well; PDFs, images, and already compressed files may
shrink little or grow. Actual byte counts are printed. No lossy rewriting of evidence occurs.

The archive includes all registered files, including a copy of the report and manifest.
Each archived member is decompressed and SHA-256 checked against its source before pruning.
`--prune` removes only registered, verified working files; report.md and run_manifest.json
remain readable. Unregistered files remain untouched. Without `--prune`, all originals
remain, so the operation adds storage rather than reclaiming it.

Use this only on an inactive, completed run. Existing archives are never overwritten.
The outer manifest records the archive checksum and member hashes. Keep it with the archive.
If a failure interrupts archiving, inspect the reported error and surviving files before retrying;
do not remove source artifacts to force a retry.

```sh
python3 SKILL/scripts/research_run.py restore RUN NEW_RUN
python3 SKILL/scripts/research_run.py checkpoint NEW_RUN --status active --next-action "Review the unresolved evidence gap"
```

Restore verifies checksums and requires a fresh destination. It restores original file bytes,
including the complete-state manifest. This is archival compression, not encryption or backup
to another device. Keep the readable report, manifest, and archive together in your backup.
