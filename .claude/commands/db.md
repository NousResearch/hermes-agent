SCHEMA-CODE ALIGNMENT CHECK for:

$ARGUMENTS

Verify that the database schema matches what the code expects. This is a two-prong check.

PRONG 1 — INSPECT THE SCHEMA:
Run against your database (adjust for your DB client):

-- Columns and types
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = '{table}'
ORDER BY ordinal_position;

-- Constraints
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = '{table}'::regclass;

-- Check constraints
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = '{table}'::regclass AND contype = 'c';

-- Triggers
SELECT trigger_name, event_manipulation, action_statement
FROM information_schema.triggers
WHERE event_object_table = '{table}';

PRONG 2 — CROSS-REFERENCE AGAINST CODE:
Find all writes:
grep -rn '{table}' {source_dir}/ --include='*.ts' | grep -iE 'insert|upsert|update|\.from\('

Find all reads:
grep -rn '{table}' {source_dir}/ --include='*.ts' | grep -iE 'select|from|\.eq\(|\.match\('

Find all column references:
For each column found in Prong 1, grep for it in the codebase:
grep -rn '{column_name}' {source_dir}/ --include='*.ts'

PRONG 3 — ALIGNMENT CHECK:
For each column the code references:
- Does it exist in the schema? (ghost column = code references non-existent column)
- Does the type match? (code expects string but schema is integer)
- Does the nullable match? (code doesn't null-check but column is nullable)

For each column in the schema:
- Is it referenced by any code? (orphan column = schema has it, code ignores it)

For each CHECK constraint:
- Does the code respect it?

REPORT:
| Column | Schema Type | Code Expects | Match? | Issue |
|--------|-----------|-------------|--------|-------|
| {col} | {type} | {type} | YES/NO | {description} |

Ghost columns (code references, schema lacks): {list}
Orphan columns (schema has, code ignores): {list}
Type mismatches: {list}
Constraint violations: {list}
