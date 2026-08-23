# Codex Model/Effort Selection and Commit-Proof Tripwire

## Deliberate routing

For substantial coding, audit, or remediation work, invoke the actual Codex CLI when practical. Pin both model and reasoning effort:

| Task | Model capability | Effort |
|---|---|---|
| Mechanical isolated edit, docs, focused test | smallest capable model | low/medium |
| Multi-file implementation/debug | reliable implementation model | medium |
| Authority, permits, receipts, security, hostile audit, architecture | strongest reasoning model | high |
| Independent final review | strongest available, read-only | high |

Probe the exact selection before a long job. Codex CLI v0.144 accepts effort as configuration, not `--reasoning-effort`:

```bash
codex exec -m <model> -c model_reasoning_effort=high \
  'Read-only smoke probe. Reply exactly: OK. Do not edit files or run commands.'
```

## Agent isolation and final closure

- Never overlap writer agents on shared files.
- Read-only auditors may run alongside independent tests, but wait for every background agent before staging/commit/final claims.
- Codex reports are leads, not execution receipts. The controller re-runs compilation, tests, and release gates.
- After any late/background completion, inspect scoped Git state and rerun affected gates before accepting work.

## Commit-proof tripwire

A dirty subdirectory may belong to a larger dirty Git root. Never say “committed” based on an agent summary or passing tests.

1. Record full pre-commit HEAD at the actual Git root.
2. Stage only task-owned paths.
3. Verify `git diff --cached --name-only -- <scope>` and `git diff --cached --check`.
4. Commit.
5. Record full post-commit HEAD and prove it differs.
6. Verify with:

```bash
git log -1 --format='%H%n%s'
git show --stat --oneline --summary HEAD -- <scope>
git status --short -- <scope>
```

If SHA is unchanged, staged content remains, or scoped paths are still dirty, closure failed: do not report a commit or complete verification.

## Evidence boundary

Real external services remain unverified unless live receipts exist. Local mock/focused/workspace tests do not certify provider endpoints, federation, production network behavior, cancellations, or rate limits.
