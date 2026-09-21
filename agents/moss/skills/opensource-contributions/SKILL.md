---
name: opensource-contributions
description: >-
  Make good open source contributions — check CONTRIBUTING.md first, follow
  project norms, be a good citizen. Covers bug reports, feature requests, and
  pull requests with a defensible default posture when the project hasn't
  documented expectations.
license: MIT
compatibility: Hermes Agent
metadata:
  tags: [opensource, contributing, github, etiquette, PRs, issues]
  related_skills: [github-issues, github-pr-workflow, github-code-review]
  version: "1.6.0"
  source_repo: https://agentskills.io
---

# Open Source Contributions

## The Golden Rule

**Make it easy for maintainers to absorb your contribution.** Maintainers are often under-resourced volunteers or small teams. Every friction point you remove — unclear reproduction steps, missing tests, bad commit messages, force-pushed history — is time they don't have to spend figuring out what you did and why.

## When to Use This Skill

- Filing a bug report or feature request on a public repository
- Preparing a pull request for any open source project
- Working on a project without documented contributing guidelines
- Reviewing your own PR before submission

## Sign-Off Rule

All issues, PRs, and comments are signed off as **Sahil Saghir**. No agent disclosure. The work stands on its own merit.

## Quick Phase Overview

| Phase | What It Covers |
|-------|----------------|
| **0 — Before You Start** | Read CONTRIBUTING.md, check existing issues/PRs, triage bugfix candidates, large-change discussion etiquette |
| **1 — Filing Issues** | Bug report and feature request templates, agent disclosure, maintainer workflow ("issue first") |
| **2 — Pull Requests** | Branch naming conventions, scope assessment, studying existing implementations, commit messages, PR templates, documentation audits, CI setup |
| **3 — After Submitting** | CI monitoring, responding to review feedback, what to do if your PR goes stale or gets closed |
| **3.5 — Follow-up After Scope Feedback** | Systematic call-site audits, filing comprehensive issues, complementary PRs |
| **4 — Release Process** | Version bumping, tagging, GitHub Releases vs tags |

## Phase 0 — Before You Start

### Step 0.1: Read the project's CONTRIBUTING.md
Use `curl` to fetch from the target repo, or `read_file` on a local clone. If no CONTRIBUTING.md exists, use the default posture: file an issue before a PR for non-trivial changes, follow existing commit style, keep PRs focused.

### Step 0.2: Check existing issues and PRs
```
gh issue list --repo <owner/repo> --search "<keyword>" --state all
gh pr list --repo <owner/repo> --search "<keyword>" --state all
```
Always check: has someone already reported this? Is a fix already in flight? If yes, comment on the existing one instead of creating duplicates.

### Step 0.3: Triage the candidate
- Genuine bug with reproduction steps? → file a bug issue
- Feature request? → file a feature issue
- Scope larger than a focused PR? → discuss in an issue first
- Trivial fix (typo, comment, single line)? → skip the issue, submit PR directly

## Phase 1 — Filing Issues

### Bug Report Template
```
## Description
<What happens>

## Reproduction
<Exact steps to trigger the bug>

## Expected behaviour
<What should happen instead>

## Environment
- OS: <linux/mac/win>
- Python version: <3.11/3.12>
- <project> version: <v1.2.3>
```

### Feature Request Template
```
## Problem
<What problem does this solve>

## Proposed solution
<What should be added/changed>

## Alternatives considered
<Other approaches and why they're less good>
```

### Maintainer workflow
For non-trivial changes, file the issue first and WAIT for maintainer feedback before coding. This prevents wasted work.

## Phase 2 — Pull Requests

### Quality Gates (MANDATORY — run BEFORE opening PR)

These gates must pass before any PR is opened. If any gate fails, fix the issue and re-run the check.

**Gate S1 — No secrets or tokens**
```
grep -rE "(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|xox[baprs]-[a-zA-Z0-9-]+)" .
```
Zero matches required. Never commit API keys, tokens, or credentials.

**Gate S2 — No personal references**
```
grep -riE "(Sahil.?SS9|KenseiAgent|kensei\.local|ops-lead@kensei)" .
```
Zero matches required. The PR must read as if written by an independent community contributor.

**Gate C1 — Conventional Commits**
```
git log origin/main..HEAD --oneline | grep -vE "^(fix|feat|docs|test|refactor|chore)(\(.+\))?: "
```
Zero matches required. Every commit must follow Conventional Commits format.

**Gate C2 — British English, no em-dashes**
```
git show --name-only | xargs grep -n "—" 2>/dev/null
```
If matches found: replace em-dashes with commas, semicolons, or colons.

**Gate C3 — Signed off as Sahil Saghir**
Confirm the issue or PR body is signed off as Sahil Saghir. No agent disclosure.

**Gate T1 — Tests pass**
Run the target project's test suite for the files you changed. At minimum:
```
pytest tests/<your-test-files> -v
```
All tests must pass. If pre-existing failures exist, note them in the PR body.

**Gate B1 — Branch is based on current main**
```
git fetch origin && git merge-base --is-ancestor origin/main HEAD && echo "OK" || echo "REBASE NEEDED"
```
Branch must be a direct descendant of the target's current main.

**Gate F1 — No unrelated changes**
```
git diff --stat origin/main..HEAD
```
Review the file list. If there are files you didn't intend to change, remove them.

### Branch naming
- `fix/description` — bug fixes
- `feat/description` — new features
- `docs/description` — documentation
- `test/description` — tests

### Before committing
1. Read CONTRIBUTING.md again — check for project-specific rules
2. Study how existing code is structured — match the style
3. Write tests that fail before the fix and pass after
4. Run existing tests — your change must not break anything
5. Check cross-platform impact if relevant

### Commit messages
Conventional Commits: `<type>(<scope>): <description>`
Sign commits: `git commit -s -m "fix(scope): description"`

### PR body
- What changed and why
- How to test (reproduction steps for bugs, usage examples for features)
- What platforms you tested on
- Reference any related issues

### Self-review checklist
Before opening the PR, verify:
1. No personal/infrastructure references in body, commits, or comments
2. Agent disclosure is present on related issues
3. Branch is based on the target's current main
4. All tests pass locally
5. No unrelated changes mixed in

## Phase 3 — After Submitting

### Monitor CI: `gh pr checks --watch`
If CI fails: read the logs, fix the issue, push a fixup commit. Do NOT force-push after review has started.

### Respond to reviews
- Respond to every comment
- Push fixup commits for requested changes
- If stalled 2+ weeks: "Friendly bump — any thoughts on this?"

### If your PR gets closed
- Read the close reason carefully
- If closed by mistake: comment asking to reopen
- If closed with feedback: address feedback, open new PR referencing old one

## Phase 3.5 — Follow-up After Scope Feedback

When a maintainer says "this is good but X should also be updated":
1. Audit ALL call sites that match the pattern — not just the one you touched
2. File a comprehensive issue listing every file/function affected
3. Submit a PR that addresses the full audit

## Pitfalls

1. **Force-push after review kills context.** After review starts, push new commits — don't amend existing ones.
2. **Cross-fork PRs can't be reopened if the head branch is deleted.** Never delete a branch that is the head of an open PR.
3. **The "I'll just fix it quickly" trap.** Scope creep during PR review. Say no: file a follow-up issue instead.
4. **Silent label failures.** Some repos auto-label. Don't fight their automation.
5. **CI debugging.** Read the full log. The first error is usually the real cause.
6. **Personal references.** Never link your own repos or mention your own infrastructure in upstream PRs.
