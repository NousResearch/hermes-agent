LIVE / DEAD PATH VERIFICATION for:

$ARGUMENTS

Determine whether the specified file, function, or code path is actually executed in production or is dead code.

STEP 1 — IMPORT CHAIN TRACE:
Starting from the target, trace the import chain upward:
grep -rn 'import.*from.*{target_file}' {source_dir}/ --include='*.ts'

For each importer, check if IT is imported by something:
grep -rn 'import.*from.*{importer_file}' {source_dir}/ --include='*.ts'

Continue until you reach either:
- A known entry point (API route, CLI entry, event handler, worker process) → LIVE
- No importers found → DEAD
- Only imported by dead code → DEAD

STEP 2 — CALLER TRACE:
If the target is a function, find all call sites:
grep -rn '{functionName}(' {source_dir}/ --include='*.ts'

For each call site, determine if the CALLER is on a live path (repeat Step 1 for the caller).

STEP 3 — KNOWN DEAD CODE CHECK:
Check the DO NOT Edit section in CLAUDE.md. Is the target in any listed dead code paths?
- If yes → DEAD (and note the live equivalent)
- If not listed → continue investigation

STEP 4 — VERDICT:

| Target | Import Chain | Callers | Entry Point Reached? | Verdict |
|--------|-------------|---------|---------------------|---------|
| {target} | {chain} | {N callers} | YES/NO | **LIVE** / **DEAD** / **PARTIAL** |

PARTIAL means: some code paths through this file are live, others are dead. Identify which.

If DEAD: recommend whether to leave it (harmless), add it to CLAUDE.md's DO NOT Edit list, or flag for cleanup in BACKLOG.md.
If LIVE: confirm the execution path from entry point to target.
