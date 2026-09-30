# Gap Analysis Report — consolidated report and resume state

Written to `.claude/memory/gap-analysis-{YYYYMMDD}.md` at the end of the run. Doubles as the resume state: on restart, read the newest report and continue from `## Phase reached`.

```markdown
# Gap Analysis — {YYYYMMDD}

- Repository: <path> | Branch: <branch> | Commit: <sha>
- Phase reached: <discovery | inventory | verdicts | specs | gate | issues | orchestrator | done>
- Mode: analyze | full

## 1. Source inventory

| Source | Status | Notes |
| --- | --- | --- |
| .specs/ | present/absent | <n files> |
| docs/ | present/absent | |
| docs/architecture/ | present/absent | |
| .claude/CONTEXT.md | present/absent | |
| CLAUDE.md / AGENTS.md / README.md | present/absent | |
| tests / linters / CI | present/absent | |
| gh auth + remote | ok/blocked | |

## 2. AS-IS × TO-BE matrix

| Topic | AS-IS | TO-BE | Sources |
| --- | --- | --- | --- |

## 3. Candidates and verdicts

| Key | Category | Verdict | Priority | Spec | Issue | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| GAP-tests-missing-auth-coverage | tests | CONFIRMADO | high | .specs/SPEC-….md | #123 | src/auth/…, .specs/… |

Rejected, duplicated, and inconclusive candidates appear here too — never hidden.

## 4. Approval gate

- Decision: <approved/rejected> | By: <user> | Date: <…>
- Per-spec decisions if mixed: <list>

## 5. Issues

- Epic: gap-analysis-{YYYYMMDD} → <url>
- Slices: <key → #number/url>

## 6. Orchestrator handoff

- Pre-conditions: tree clean <y/n>, branch <name>, specs Approved <y/n>
- Commands run / delegated phases:
- Result: <PASS/FAIL> with evidence (build, lint, tests output refs)

## 7. Pendencies

- <anything left open: INCONCLUSIVO items awaiting decision, failed phases, skipped phases and why>
```
