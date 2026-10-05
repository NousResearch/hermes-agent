---
name: hermes-upstream-pr-workflow
description: Use when making a PR to NousResearch/hermes-agent. Pull upstream main, branch cleanly, follow Teknium-style issue/PR format, validate, scan for credentials, commit, push, and open the PR.
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, hermes-agent, pull-request, upstream, credentials]
    related_skills: [github-pr-workflow, github-issues, hermes-agent-skill-authoring]
---

# Hermes Upstream PR Workflow

## Overview

Use this workflow for contributions to `NousResearch/hermes-agent`. It is optimized for small, reviewable upstream PRs that match the maintainer style: Teknium-style issue and PR bodies, clean branch from latest upstream `main`, no credentials in commits, validation before push, and a final PR link.

## When to Use

Use when asked to:

- make a PR for Hermes Agent itself
- update an in-repo Hermes skill
- create an upstream issue + PR
- follow Teknium-style issue or PR formatting
- avoid committing credentials

Do not use for unrelated repos unless explicitly asked to apply the same pattern.

## Required Skills to Load

Before acting, load:

- `hermes-agent`
- `github-pr-workflow`
- `github-issues`
- `hermes-agent-skill-authoring` if editing or creating skills

## Source of Truth: CONTRIBUTING.md

`CONTRIBUTING.md` in the repo root is the canonical contribution guide. Always check it first for:

- Contribution priorities
- Skill vs. tool decisions
- Bundling guidance
- Development setup
- Test commands
- Code style

This skill is a focused workflow companion; it does not replace `CONTRIBUTING.md`.

## Contribution Priorities (from CONTRIBUTING.md)

1. **Bug fixes** — crashes, incorrect behavior, data loss. Always top priority.
2. **Cross-platform compatibility** — macOS, Linux distros, WSL2 on Windows.
3. **Security hardening** — shell injection, prompt injection, path traversal.
4. **Performance and robustness** — retry logic, error handling, graceful degradation.
5. **New skills** — but only broadly useful ones. See below.
6. **New tools** — rarely needed. Most capabilities should be skills.
7. **Documentation** — fixes, clarifications, new examples.

## Should it be a Skill or a Tool?

From `CONTRIBUTING.md`:

### Make it a Skill when:

- The capability can be expressed as instructions + shell commands + existing tools
- It wraps an external CLI or API that the agent can call via `terminal` or `web_extract`
- It does not need custom Python integration or API key management baked into the agent

### Make it a Tool when:

- It requires end-to-end integration with API keys, auth flows, or multi-component configuration managed by the agent harness
- It needs custom processing logic that must execute precisely every time
- It handles binary data, streaming, or real-time events that cannot go through the terminal

## Should the Skill be Bundled?

Bundled skills (in `skills/`) ship with every Hermes install. They should be **broadly useful to most users**.

If your skill is official and useful but not universally needed (e.g., a paid service integration, a heavyweight dependency), put it in **`optional-skills/`** — it ships with the repo but is not activated by default.

If your skill is specialized, community-contributed, or niche, it is better suited for a **Skills Hub**.

## Delegation Guidance

Upstream PRs to `NousResearch/hermes-agent` are non-trivial by definition. The default orchestrator should not perform the issue/PR/repo operations itself when the request involves implementation, validation, or review.

If the user invokes plan mode for an upstream Hermes PR, do not execute this workflow. Instead, write the plan file so it explicitly names `hermes-upstream-pr-workflow`, lists the required companion skills (`hermes-agent`, `github-pr-workflow`, `github-issues`), and spells out the issue/PR commands, credential scan, tests, and verification steps to run later.

Before any git edit, commit, push, GitHub issue creation, or PR creation:

1. Split the request into independent workstreams.
2. Create Kanban tasks assigned to suitable profiles.
3. Let workers perform the repo edits, issue/PR body drafting, validation, commit/push/PR execution, and review as appropriate.
4. The orchestrator may do only lightweight prerequisite discovery needed to route, monitor workers, summarize results, and run deterministic verification after workers finish.
5. If a worker stalls, reclaim or reassign the task. Do not silently finish the worker's job yourself.

Typical graph for this class of task:

```text
T1 fixer     make the bounded repo/skill/docs change, validate, commit
T2 explorer  inspect existing issue/PR style or source references, if needed
T3 oracle    review final diff and PR/issue body when risk warrants it  parents: T1, T2
```

## Workflow

The following steps describe what the delegated worker(s) should do, or what the orchestrator should ask them to do.

### 1. Inspect repo and auth

```bash
git status --short --branch
git remote -v
git branch --show-current
git log -1 --oneline
gh auth status
```

If the working tree is dirty, stop and decide whether to commit, stash, or use a worktree. Do not overwrite uncommitted work.

### 2. Pull newest upstream main into a new branch

Prefer branching from `upstream/main`, not local divergent `main`:

```bash
git fetch upstream main
git checkout -B <branch> upstream/main
```

Use branch prefixes:

- `docs/` for docs or skills
- `fix/` for bug fixes
- `feat/` for features
- `test/` for tests only

### 3. Match Teknium issue/PR style

Inspect recent Teknium issues and PRs before writing bodies:

```bash
gh issue list --author teknium1 --state all --limit 5 --json number,title,body,labels,state,createdAt
gh pr list --author teknium1 --state all --limit 5 --json number,title,body,state,createdAt
```

Typical issue structure:

```markdown
## Summary
<concise problem>

## Repro / Example / Why this matters
<clear concrete scenario>

## Proposed fix
<numbered list>

## Scope
<what this does and does not do>

## Validation
<expected checks>
```

Typical PR structure:

```markdown
## Summary
<what changed and why>

## Changes
- <bullet>
- <bullet>

## Validation
- <command/result>

Closes #<issue-number>
```

### 4. Make the smallest useful change

For in-repo skills, edit files under `skills/<category>/<skill>/SKILL.md`. Use `patch` for small changes. Avoid broad rewrites.

For any Hermes upstream PR, prefer the smallest useful change:

- one skill or one tool file per PR when possible
- one bug fix or one docs clarification per PR
- one new bundled skill or one new tool per PR
- keep diffs focused; avoid bundling unrelated changes

### 5. Validate before commit

For skill edits, validate frontmatter:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
p = Path('skills/<category>/<skill>/SKILL.md')
content = p.read_text()
assert content.startswith('---')
end = content.find('\n---\n', 3)
assert end != -1
fm = yaml.safe_load(content[3:end])
assert isinstance(fm, dict)
assert fm.get('name')
assert fm.get('description') and len(fm['description']) <= 1024
assert content[end+5:].strip()
assert len(content) <= 100_000
print('ok')
PY
```

Run relevant tests with the wrapper, never direct pytest on Linux:

```bash
scripts/run_tests.sh tests/tools/test_skill_manager_tool.py tests/tools/test_skill_size_limits.py -q
```

If the wrapper fails because local dependencies are missing, install the missing dev dependency in `.venv`, then rerun the wrapper. Record both failed attempt and final pass honestly only if relevant.

### 6. Credential scan before commit and PR

Scan the diff before staging/committing:

```bash
git diff | grep -Ei 'gho_|github_pat_|sk-[A-Za-z0-9]|BEGIN (RSA|OPENSSH|PRIVATE)|api[_-]?key\s*[:=]|token\s*[:=]|password\s*[:=]|secret\s*[:=]' && {
  echo 'Potential credential in diff' >&2
  exit 1
} || true
```

Also inspect `git status --short` and only stage intended files.

Never commit `.env`, `auth.json`, credentials, tokens, debug dumps with secrets, or local profile config unless explicitly intended and sanitized.

### 7. Create issue

Write body to `/tmp/..._issue.md` and create via `gh issue create`:

```bash
gh issue create \
  --title "<type>(<scope>): <short-description>" \
  --body-file /tmp/issue.md \
  --label type/docs \
  --label tool/skills
```

Use labels that exist. Check with:

```bash
gh label list --limit 200
```

### 8. Commit, push, PR

```bash
git add <intended files>
git commit -m "<type>(<scope>): <short-description>"
git push -u origin HEAD
```

Create the PR against upstream:

```bash
gh pr create \
  --repo NousResearch/hermes-agent \
  --base main \
  --head <fork-owner>:<branch> \
  --title "<type>(<scope>): <short-description>" \
  --body-file /tmp/pr.md
```

Verify:

```bash
gh pr view <pr-number-or-url> --repo NousResearch/hermes-agent --json number,title,url,state,headRefName,baseRefName,author
git status --short --branch
```

## Generic PR Mechanics

For branch creation, commit conventions, CI monitoring, auto-fix loops, and merge operations, see the `github-pr-workflow` skill. This skill covers the Hermes-specific upstream conventions; `github-pr-workflow` covers the generic GitHub lifecycle.

## Verification Checklist

- [ ] Started from latest `upstream/main`
- [ ] New topic branch created
- [ ] Only intended files changed
- [ ] Skill frontmatter valid if a skill was edited
- [ ] Relevant tests run through `scripts/run_tests.sh`
- [ ] Diff credential scan passed
- [ ] Issue created with Teknium-style sections
- [ ] Commit created with conventional message
- [ ] Branch pushed to fork
- [ ] PR opened against `NousResearch/hermes-agent:main`
- [ ] PR body includes validation and `Closes #...`
- [ ] Final response includes issue URL, PR URL, branch, commit, tests, and credential-scan status
