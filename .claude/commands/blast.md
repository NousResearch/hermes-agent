BLAST RADIUS ANALYSIS for:

$ARGUMENTS

Map every file, function, type, and pipeline stage that would be affected by this change. This is a pre-change safety analysis — run BEFORE making modifications.

STEP 1 — DIRECT DEPENDENCIES:
Find everything that directly imports or calls the target:
grep -rn 'import.*from.*{target}' {source_dir}/ --include='*.ts' --include='*.tsx'
grep -rn '{functionName}\(' {source_dir}/ --include='*.ts' --include='*.tsx'

STEP 2 — TRANSITIVE DEPENDENCIES:
For each direct dependent found in Step 1, find ITS dependents:
Repeat until you reach terminal consumers (API routes, UI components, event handlers, CLI entry points).

STEP 3 — TYPE DEPENDENCIES:
If the change modifies a type/interface:
grep -rn '{TypeName}' {source_dir}/ --include='*.ts' --include='*.tsx'
Every consumer of this type must be checked for compatibility.

STEP 4 — DATABASE DEPENDENCIES:
If the change affects a DB column or table:
- Find all queries referencing it
- Check any access control policies
- Check migrations that depend on current schema

STEP 5 — PIPELINE / WORKFLOW STAGE MAP:
Which stages of your application pipeline are in the blast radius?
Mark each: DIRECT (code is modified) or INDIRECT (consumes modified output)

| Stage | Impact | Reason |
|-------|--------|--------|
| {stage} | DIRECT/INDIRECT/NONE | {why} |

STEP 6 — RISK SUMMARY:
- Files in blast radius: {N}
- Stages affected: {list}
- Inviolable rules at risk: {list any rules from CLAUDE.md that govern modified code}
- Recommended test plan: {what to verify after the change}
