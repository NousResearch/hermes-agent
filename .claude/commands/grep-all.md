EXHAUSTIVE GREP COVERAGE for:

$ARGUMENTS

Perform exhaustive grep coverage before concluding anything about this target. This is NOT a quick search — it is a full codebase sweep. Do not propose a root cause or fix until this is complete.

STEP 1 — BROAD SWEEP:
Search for every reference, alias, import, export, handler, route, schema, feature flag, env var, test, and fallback related to this target.

Run ALL of these (adjust pattern from $ARGUMENTS):
grep -rn '{pattern}' {source_dir}/ --include='*.ts'
grep -rn '{pattern}' {ui_dir}/ --include='*.tsx'
grep -rn '{pattern}' tests/ --include='*.ts'
grep -rn '{pattern}' .claude/ --include='*.md'
grep -rn '{pattern}' {db_dir}/ --include='*.sql'   (if applicable)

STEP 2 — ALIAS AND VARIANT SWEEP:
If the pattern is a function name, also search for:
- Renamed imports: grep -rn 'as.*{pattern}\|{pattern}.*as' {source_dir}/ --include='*.ts'
- String references: grep -rn "'{pattern}'\|`{pattern}`\|\"{pattern}\"" {source_dir}/ --include='*.ts'
- Partial matches: grep -rn '{partial_pattern}' {source_dir}/ --include='*.ts'

If the pattern is a database table/column:
- ORM/client calls: grep -rn "from('{table}')" {source_dir}/ --include='*.ts'
- SQL strings: grep -rn '{pattern}' {source_dir}/ --include='*.ts' | grep -iE 'select|insert|update|delete'

STEP 3 — LIST EVERY HIT:
Present every single match. Do not summarize or skip "obvious" ones.
Group by file for readability.

STEP 4 — CLASSIFY EACH MATCH:
Label each match:
- WRITE — creates/modifies this value
- READ — consumes this value
- IMPORT — import/export statement
- TYPE — type definition or interface
- TEST — in a test file
- DEAD — in known dead code (check CLAUDE.md → DO NOT Edit)
- COMMENT — in a comment, not executable
- CONFIG — in configuration, env var, or feature flag

STEP 5 — LIVE PATH FILTER:
Flag any matches in known dead code paths (see CLAUDE.md → DO NOT Edit).

STEP 6 — SUMMARY:
Total matches: {N} across {N} files
Live matches: {N} | Dead matches: {N}
Write sites: {N} | Read sites: {N} | Import sites: {N}

If zero matches found, say so explicitly — do not speculate.
