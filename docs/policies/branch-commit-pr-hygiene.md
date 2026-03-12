# Branch, Commit, and PR Hygiene Policy

## Purpose
Keep delivery velocity high without sacrificing traceability, rollback safety, or review quality.

---

## 1) Branch Strategy

## Naming
Use:
- `feat/<scope>-<short-topic>`
- `fix/<scope>-<short-topic>`
- `refactor/<scope>-<short-topic>`
- `docs/<scope>-<short-topic>`
- `chore/<scope>-<short-topic>`

Examples:
- `feat/gateway-runtime-overrides`
- `fix/discord-attachment-cache-fallback`
- `docs/subagent-delivery-playbook`

## Rules
1. One branch per initiative slice.
2. Avoid mixed-purpose branches (feature + unrelated cleanup).
3. Rebase frequently to reduce drift.
4. Delete merged branches promptly.

---

## 2) Commit Standards

## Message format (Conventional Commits)
`type(scope): imperative summary`

Allowed types:
- `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

Examples:
- `feat(gateway): add /ask reasoning override receipt`
- `fix(discord): prevent attachment ingest fallback stall`
- `docs(runbook): add runtime override and RBAC operations guide`

## Commit body (when needed)
Include:
- why this change exists
- risk/impact
- test evidence

## Commit size
- Prefer small, reviewable commits (roughly one logical change).
- If change is large, split by behavior boundaries (parser, command wiring, tests, docs).

## Must not commit
- backup artifacts (`*.bak`, temp snapshots)
- secrets/tokens
- unrelated formatting churn
- dead experimental files

---

## 3) Pre-Commit Local Checks
Before every commit, Hermes should verify:
1. changed files match task scope
2. targeted tests pass
3. no obvious lint/type/syntax breakage
4. git diff has no accidental debug output

Suggested command pattern:

```bash
source .venv/bin/activate
python -m pytest <targeted-tests> -q
python -m py_compile <touched-python-files>
```

(Use `venv` if this checkout uses `venv` instead of `.venv`.)

---

## 4) PR Hygiene

Each PR should include:
1. Problem statement
2. Scope of change
3. Test evidence (exact commands + pass counts)
4. Risk assessment
5. Rollback plan
6. Follow-up items (if deferred)

PRs should be coherent and reviewable in under ~30 minutes.

---

## 5) Squash vs. Multi-Commit

## Keep multi-commit when:
- commits represent meaningful review units
- history helps future debugging/audit

## Squash when:
- branch contains noisy fixup churn
- commit sequence is not useful historically

If squashing, preserve key details in PR description.

---

## 6) Hotfix Exception Path
For urgent production fixes:
1. create `fix/hotfix-<topic>`
2. implement minimal safe patch
3. run focused verification
4. merge quickly
5. follow with hardening/tests/docs PR if needed

---

## 7) Manager Enforcement Checklist
Hermes should reject/repair before merge if:
- branch name violates policy
- commit message format is invalid
- tests are missing for behavioral changes
- docs are missing for operator-facing changes
- unrelated file churn appears
