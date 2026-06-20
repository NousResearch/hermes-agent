# Stephen's GitHub Standard

> Canonical operating standard for serious Stephen/Kovu/Hermes project work.

**Goal:** Make GitHub the source of truth for project history, planning, review, CI, releases, and handoffs.

**Core rule:** If a change matters, it should not live only on a local machine. Local-only is fine for scratch experiments; serious project work goes through issues, branches, commits, PRs, checks, and approved merges.

---

## 1. Required workflow

Every meaningful project change follows this lane:

1. **Issue** — capture the problem, desired outcome, acceptance criteria, risks, and affected repo.
2. **Plan doc** — for multi-step work, write a plan in `docs/plans/YYYY-MM-DD-short-name.md` before coding.
3. **Branch** — create a focused branch from the current default branch.
4. **Code** — make the smallest solid change that satisfies the plan/issue.
5. **Local tests** — run targeted tests first, then broader checks when appropriate.
6. **Commit** — commit clear, reviewable chunks with conventional-ish messages.
7. **Push** — push the branch to GitHub.
8. **PR** — open a PR linked to the issue with summary, tests, screenshots/logs if useful.
9. **CI** — let GitHub Actions run. Fix failures in follow-up commits.
10. **Review/fix** — review the diff, address feedback, and keep the PR clean.
11. **Merge approval** — do not merge without Stephen's explicit approval unless he has pre-authorized that repo/workflow.
12. **Release/tag if meaningful** — tag user-visible releases, deployable milestones, or compatibility boundaries.

Shortcut only for tiny documentation/config updates when Stephen explicitly says local direct edit is okay. Even then, check repo state first and commit/push if the repo is serious.

---

## 2. Repo structure baseline

Use this baseline for new or cleaned-up repos:

```text
repo-root/
├── README.md
├── LICENSE                         # if public or reusable
├── .gitignore
├── .env.example                    # safe names only, never real values
├── docs/
│   ├── plans/
│   │   └── YYYY-MM-DD-feature.md
│   ├── decisions/                  # optional ADRs / decision records
│   └── runbooks/                   # optional ops/deploy notes
├── src/ or app/ or backend/frontend
├── tests/
├── scripts/                        # safe local/dev automation
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── feature.md
│   │   └── bug.md
│   ├── pull_request_template.md
│   └── workflows/
│       └── ci.yml
└── CHANGELOG.md                    # for versioned/user-visible projects
```

Project-specific layouts are fine. The non-negotiables are: README, ignore file, safe env example when env vars exist, docs/plans for multi-step work, tests where code exists, and GitHub templates/workflows for serious repos.

---

## 3. Issue standard

Every non-trivial task starts as a GitHub issue.

### Feature issue template

Path: `.github/ISSUE_TEMPLATE/feature.md`

```markdown
---
name: Feature
about: Plan and track a new capability
labels: enhancement
---

## Goal
What should exist when this is done?

## User-facing behavior
What does Stephen or the end user see/do?

## Acceptance criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Tests or checks prove it works

## Scope
In:
-

Out:
-

## Risks / constraints
-

## Notes / links
-
```

### Bug issue template

Path: `.github/ISSUE_TEMPLATE/bug.md`

```markdown
---
name: Bug
about: Reproduce and fix broken behavior
labels: bug
---

## Problem
What is broken?

## Reproduction
1.
2.
3.

## Expected behavior
What should happen?

## Actual behavior
What happens instead?

## Evidence
Logs, screenshots, command output, failing test, URL, etc.

## Acceptance criteria
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Regression test or verification added
```

---

## 4. Plan document standard

Use a plan doc before implementing multi-step features, repo migrations, risky refactors, CI changes, or anything an agent may execute later.

Path format:

```text
docs/plans/YYYY-MM-DD-short-feature-name.md
```

Required header:

```markdown
# Feature Name Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task when appropriate.

**Goal:** One sentence.

**Architecture:** Two or three sentences explaining the approach.

**Tech Stack:** Main frameworks/libraries/tools.

---
```

Each task should include objective, exact files, test-first step when code changes, command to run, expected result, and commit message.

Plan docs are not bureaucracy. They are receipts so the next agent does not freestyle with the repo like it owes them money.

---

## 5. Branch naming

Branch names should be lowercase, slash-separated, and tied to an issue when possible.

Formats:

```text
feat/123-short-name
fix/123-short-name
chore/123-short-name
docs/123-short-name
refactor/123-short-name
spike/short-experiment
```

Rules:
- One branch per focused issue or plan.
- Branch from updated default branch: `main` unless the repo uses another default.
- Do not reuse stale branches for unrelated work.
- Do not delete remote branches without Stephen's explicit approval.

---

## 6. Commit naming

Use clear conventional-ish commits:

```text
feat: add playlist score card endpoint
fix: handle missing Gate.io funding data
docs: add GitHub standard rollout plan
test: cover alert threshold edge cases
refactor: split ingestion source adapters
chore: update CI dependencies
```

Rules:
- Commit only related files.
- Do not commit generated junk, caches, local DBs, logs, secrets, or `.env`.
- Commit messages should explain the change, not the effort.
- Prefer small reviewable commits over one giant mystery brick.

---

## 7. Pull request standard

Path: `.github/pull_request_template.md`

```markdown
## Summary
-

## Linked issue / plan
Closes #
Plan: docs/plans/YYYY-MM-DD-short-name.md

## What changed
-

## Tests / verification
- [ ] Command: `...`
      Result: `...`
- [ ] CI passed

## Screenshots / artifacts
If UI/media output changed, add screenshot, URL, or generated artifact path.

## Risk / rollback
Risk level: low / medium / high
Rollback: revert PR / disable flag / restore previous deploy

## Secrets check
- [ ] No secrets, tokens, credentials, private data, or real `.env` values committed
```

PR rules:
- Link the issue and plan doc when they exist.
- Include exact test commands and results.
- If CI fails, fix it before asking Stephen to merge.
- Do not merge without Stephen's explicit approval unless he has pre-authorized that repo/workflow.

---

## 8. CI baseline

Every serious repo should have a minimal GitHub Actions workflow.

Path: `.github/workflows/ci.yml`

### Generic baseline

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Detect project
        run: |
          echo "Repository contents:"
          find . -maxdepth 2 -type f | sort | sed 's#^./##' | head -200
```

### Python baseline

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f pyproject.toml ]; then pip install -e '.[dev]' || pip install -e .; fi
      - name: Run tests
        run: |
          if [ -d tests ]; then pytest -q; else python -m compileall .; fi
```

### Node/Vite/Next baseline

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: npm
      - name: Install dependencies
        run: npm ci
      - name: Lint
        run: npm run lint --if-present
      - name: Test
        run: npm test --if-present
      - name: Build
        run: npm run build --if-present
```

CI rules:
- Keep baseline simple first. Add matrix/deploy jobs only when useful.
- Use least-privilege permissions.
- No secrets in logs.
- PR must disclose any failing/skipped checks.

---

## 9. Secrets and environment rules

Non-negotiable:
- Never print, commit, paste, summarize, or expose real tokens, credentials, cookies, private keys, OAuth secrets, dashboard passwords, or private personal data.
- Never commit `.env`, `.env.local`, `.env.production`, local SQLite DBs with user/project data, browser profiles, or credential stores.
- Use `.env.example` with safe placeholder names only.
- Use GitHub Actions secrets for CI/deploy values.
- If a secret appears in output, redact it before sharing and rotate it if exposure was real.

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
*.sqlite
*.sqlite3
*.db
*.log
.DS_Store
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
node_modules/
dist/
build/
.next/
.vite/
coverage/
```

Safe `.env.example` style:

```dotenv
API_BASE_URL=https://example.com
GITHUB_TOKEN=replace_me
DATABASE_URL=sqlite:///local-dev.sqlite
```

Do not put real values in examples. Mira, examples are not confession booths.

---

## 10. Release and tagging rules

Tag when a change represents a meaningful user-visible or operational milestone:
- first working MVP
- deployed release
- breaking change
- stable integration point
- rollback point before risky migration

Tag format:

```text
v0.1.0
v0.2.0
v1.0.0
v1.1.0-rc1
```

Use semantic versioning loosely:
- MAJOR: breaking change or major product milestone
- MINOR: new capability
- PATCH: bugfix or small polish

Release checklist:
- [ ] PR merged with approval
- [ ] CI passing on default branch
- [ ] Version/tag chosen
- [ ] Release notes generated or written
- [ ] Deployment/runbook updated if needed

Commands:

```bash
git checkout main
git pull --ff-only
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --generate-notes
```

Do not create public releases/tags for Stephen's serious repos without approval unless the task explicitly requests it.

---

## 11. Primo enforcement rules for future agent work

These rules apply to Primo/Hermes/agents working on Stephen's projects.

### Before touching files

1. Identify the repo root.
2. Run:

```bash
git status --short
git branch --show-current
git remote -v
git log --oneline -5
```

3. Check for existing uncommitted work. If it is not yours, do not overwrite it.
4. Confirm the default branch and whether local branch is behind/ahead.
5. Inspect project instructions: `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, README, docs.

### During work

- Use GitHub as source of truth for serious projects.
- Create/update issue and plan when appropriate.
- Work on a focused branch.
- Keep changes small and reviewable.
- Run relevant local tests before committing.
- Push branch and open/update PR for meaningful work.
- Record exact commands and results in PR or kanban handoff.

### Never without Stephen's explicit approval

- Merge PRs.
- Delete repos.
- Delete branches, especially remote branches.
- Force-push shared branches.
- Rewrite public history.
- Publish releases/tags.
- Expose or rotate credentials unless specifically requested.
- Speak/post/send externally as Stephen unless he approved the exact content/action.

### No local-only serious changes

If the task changes a real project in a meaningful way, the expected end state is:
- branch pushed to GitHub,
- PR opened or updated,
- tests/CI status reported,
- Stephen has a clear review link.

If that cannot happen because the repo is not normal, auth is missing, CI is absent, or the repo is unsafe to modify, report it clearly and recommend the next concrete follow-up. Do not pretend the workflow happened.

---

## 12. Repo rollout checklist

Use this checklist when upgrading an existing project to the standard:

- [ ] Confirm repo root and GitHub remote.
- [ ] Check `git status --short`; protect existing work.
- [ ] Confirm default branch and current branch.
- [ ] Add/update README if missing.
- [ ] Add/update `.gitignore` for stack and secrets.
- [ ] Add/update `.env.example` if env vars are used.
- [ ] Add `.github/ISSUE_TEMPLATE/feature.md`.
- [ ] Add `.github/ISSUE_TEMPLATE/bug.md`.
- [ ] Add `.github/pull_request_template.md`.
- [ ] Add minimal `.github/workflows/ci.yml`.
- [ ] Add `docs/plans/` and first plan if needed.
- [ ] Run local tests/build.
- [ ] Commit, push branch, open PR.
- [ ] Wait for CI and fix failures.
- [ ] Ask Stephen for merge approval.

---

## 13. Quick command flow

```bash
# orient
git status --short
git branch --show-current
git remote -v
git log --oneline -5

# sync default branch
git checkout main
git pull --ff-only

# create issue if needed
gh issue create --title "Short title" --body-file /tmp/issue.md --label enhancement

# create branch
git checkout -b feat/123-short-name

# work + test
pytest -q
npm test --if-present
npm run build --if-present

# commit + push
git add path/to/files
git commit -m "feat: short description"
git push -u origin feat/123-short-name

# PR
gh pr create --fill --base main --head feat/123-short-name

# inspect CI
gh run list --limit 10
gh pr checks
```

If any command reveals danger — dirty working tree, unknown remote, secrets, missing auth, failing critical tests — stop and report the real state instead of plowing ahead.

---

## 14. Definition of done

A GitHub-standard task is done when:
- The issue/plan/branch/PR path exists for meaningful work.
- Local tests were run and results recorded.
- CI status is known.
- Secrets were protected.
- Stephen has a clear link or local artifact to review.
- No merge/delete/release action happened without approval.

For this canonical document specifically, done means this file exists at:

```text
~/github-standard-rollout/GITHUB_STANDARD.md
```

and its content covers workflow, repo structure, templates, CI, branch/commit/release rules, secrets rules, and Primo enforcement rules.


---

## Hermes Agent adoption note

This repository treats GitHub as the source of truth for durable project work: issues capture intent, `docs/plans/` captures multi-step implementation plans, `docs/decisions/` captures lasting architectural/product decisions, branches isolate work, PRs carry review and CI, and merges require Stephen's explicit approval unless pre-authorized.
