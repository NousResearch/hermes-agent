ZERO TOLERANCE PATTERN SCAN

Run the project's inviolable-rule sweep. This audit should run at least once per session — at session start and before session end.

_Replace the patterns below with your project's actual inviolable rules from CLAUDE.md._

RULE 1 — {Your Cardinal Sin pattern}:
grep -rn '{pattern}' {source_dir}/ --include='*.ts' -A {context_lines} | grep -iE '{violation_pattern}'

If ANY match violates the rule: STOP. This is a P0. Report immediately with file:line.

RULE 2 — {Your second inviolable rule}:
grep -rn '{pattern}' {source_dir}/ --include='*.ts'

If ANY match is found: STOP. Report immediately.

RULE 3 — {Your third inviolable rule}:
grep -rn '{pattern}' {source_dir}/ --include='*.ts'
grep -rn '{pattern}' {ui_dir}/ --include='*.tsx'

If ANY match writes forbidden content to customer-facing output: STOP. Report immediately.

RULE 4 — Hardcoded config values (if applicable):
grep -rn '{config_value_pattern}' {source_dir}/ --include='*.ts' | grep -v '{config_file}\|\.md'

If ANY match is a hardcoded value that should come from config: report it.

REPORT FORMAT:
RULE 1: {CLEAN or VIOLATION at file:line}
RULE 2: {CLEAN or VIOLATION at file:line}
RULE 3: {CLEAN or VIOLATION at file:line}
RULE 4: {CLEAN or {N} violations found}

OVERALL: {ALL CLEAN or {N} VIOLATIONS FOUND}

NOTE: /zt is an alias for this command.
