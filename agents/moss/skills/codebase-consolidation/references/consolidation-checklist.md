# Consolidation Checklist

Reusable step-by-step checklist for merging and consolidating codebases.

## Pre-Merge

- [ ] Map both codebases: list all files, line counts, key function/class names
- [ ] Identify overlap vs complement (MERGE / REPLACE / COEXIST)
- [ ] Document the merge plan before touching files
- [ ] Check for cron jobs referencing scripts in the source repo

## During Merge

- [ ] Validate every import chain: existing code must still resolve
- [ ] Check config keys: every fallback reference must survive
- [ ] Check dict field access: merged configs must include all expected keys
- [ ] Write merged files to `/tmp/` first, validate, then copy

## Post-Merge

- [ ] Compile-check every .py file
- [ ] Runtime import verification
- [ ] Git commit with descriptive message
- [ ] Delete source repo (or archive with symlink migration)

## Repo Retirement (Archive Mode)

- [ ] Verify content exists in target repo(s)
- [ ] Identify salvageable work (plugins, bridges, configs)
- [ ] Check cron jobs and skill references for baked-in paths
- [ ] Confirm remote is unreachable (404/archived)
- [ ] Create kanban trail for any preserved work
