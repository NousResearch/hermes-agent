UPSTREAM AND DOWNSTREAM IMPACT ANALYSIS for:

$ARGUMENTS

Audit the full impact surface of the specified change, file, or function. Map everything that feeds INTO it (upstream) and everything that CONSUMES from it (downstream).

STEP 1 — IDENTIFY THE CHANGE:
- Read the target file/function/commit
- List every export, function signature, type, and constant that is modified or could be affected

STEP 2 — TRACE UPSTREAM (who feeds this?):
For each input parameter, imported value, or consumed data:
1. Where does it originate? Trace back to the source.
2. What transformations happen between origin and here?
3. Are there any lossy transformations? (type narrowing, field drops, truncation)
4. What stage of the pipeline does each upstream source belong to?

Map: Source → Transform → Transform → **THIS FUNCTION**

STEP 3 — TRACE DOWNSTREAM (who consumes this?):
For each output, return value, DB write, or event emission:
1. grep for all import sites of this file: grep -rn 'import.*from.*{file}' {source_dir}/ --include='*.ts'
2. grep for all call sites: grep -rn '{functionName}' {source_dir}/ --include='*.ts'
3. For each consumer, identify what STAGE of the pipeline it belongs to
4. Continue tracing until you reach a terminal: DB write, API response, rendered output, event emission
5. Flag any consumer that crosses a pipeline boundary

Map: **THIS FUNCTION** → Consumer A → Consumer B → Terminal output

STEP 4 — IDENTIFY PIPELINE BOUNDARIES:
List every pipeline boundary this change crosses or could affect.
(Replace with your actual stage names from CLAUDE.md → Project → Pipeline)

| Boundary | Crossed? | Reason |
|----------|----------|--------|
| Stage A → Stage B | YES/NO | {why} |
| Any stage → DB | YES/NO | {which tables} |
| Any stage → event emission | YES/NO | {which events} |

STEP 5 — INVIOLABLE RULE RISK SCAN:
Does this change interact with any inviolable rule from CLAUDE.md? Check each rule defined there.

STEP 6 — RISK RATING:

| Impact Zone | Files Affected | Pipeline Boundary? | Rule Risk? | Risk Level |
|------------|---------------|-------------------|------------|-----------|
| {zone} | {files} | YES/NO | {rule or NONE} | HIGH/MED/LOW |

OVERALL RISK: {HIGH/MEDIUM/LOW}
RECOMMENDED VERIFICATION: {what to check after implementation}

NOTE: Both `/upstream` and `/downstream` are aliases for this command.
