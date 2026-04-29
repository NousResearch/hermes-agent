FINALIZE AND SHIP

Complete all pending work, verify everything compiles, commit, and push to origin.

STEP 1 — BUILD CHECK:
Run your type checker or build:
{your build command from CLAUDE.md → Commands}

If ANY errors: fix them. Do not proceed until the build is clean.

STEP 2 — ZERO TOLERANCE SCAN:
Run /zt to check for any inviolable rule violations (see CLAUDE.md → Inviolable Rules).
If ANY violation: STOP and report. Do not commit violations.

STEP 3 — UNCOMMITTED CHANGES REVIEW:
git status
git diff --stat

Review every changed file. For each:
- Is it on the LIVE path? (not in any dead code path listed in CLAUDE.md → DO NOT Edit)
- Does it touch an inviolable rule?
- Does it cross a pipeline boundary?

STEP 4 — STAGE AND COMMIT:
git add {specific files — prefer explicit over git add .}
git commit -m "{type}({scope}): {description}"

Commit message rules:
- type: feat, fix, refactor, docs, test, chore
- scope: the module or subsystem being changed
- description: what changed and why
- Reference any relevant decision code or ticket

STEP 5 — PUSH:
git push origin {current-branch}

If push fails due to upstream changes:
- git pull --rebase origin {current-branch}
- Resolve any conflicts
- Re-run build check (Step 1)
- git push origin {current-branch}

STEP 6 — REPORT:
Commit hash: {hash}
Files changed: {N}
Insertions: {N} | Deletions: {N}
Build: clean
Zero tolerance: all clean
Push: success to origin/{branch}

RULES:
- NEVER push directly to main, master, or production branches without operator instruction.
- NEVER force push (--force is denied in settings.json).
- ALWAYS verify build clean before committing.
- ALWAYS run /zt zero tolerance scan before committing.
