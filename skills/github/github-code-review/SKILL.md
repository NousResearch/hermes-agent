---
name: github-code-review
description: "Review PRs: diffs, inline comments via gh or REST."
version: 1.2.0
author: "A-KH17, Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Code-Review, Pull-Requests, Git, Quality]
    related_skills: [github-auth, github-pr-workflow]
---

# GitHub Code Review Skill

Perform code reviews on local changes before pushing, or review open PRs on GitHub and
post comments or formal reviews. Most of this skill uses plain `git` — the `gh`/`curl`
split only matters for PR-level interactions. It does not author PRs
(`github-pr-workflow`) or triage issues (`github-issues`).

## When to Use

- "Review my changes before I push" — local `git diff main...HEAD` review, no API needed.
- "Review PR #N" or a PR URL — fetch, inspect, test, and review a GitHub pull request.
- "Comment on / approve / request changes on PR #N" — post inline comments or a formal review.

## Prerequisites

- Authenticated with GitHub (see `github-auth` skill)
- Inside a git repository

### Setup (for PR interactions)

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py")
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

## How to Run

Run every shell command via `terminal`. Read surrounding code with `read_file` and find
related call sites with `search_files` — diffs alone can miss issues visible only in
context. Pick the matching workflow under `## Procedure` (local pre-push or PR review)
and follow it end to end. Full posting payloads (`gh` and REST/`curl`) live in
[references/github-rest-api.md](references/github-rest-api.md).

## Quick Reference

| Task | Core command (via `terminal`) |
|------|-------------------------------|
| Local diff (pre-push) | `git diff main...HEAD` (+ `--stat`, `--name-only`) |
| Staged changes only | `git diff --staged` |
| PR details / diff / CI | `gh pr view N` · `gh pr diff N` · `gh pr checks N` |
| Check out a PR | `git fetch origin pull/N/head:pr-N && git checkout pr-N` (or `gh pr checkout N`) |
| Env/auth setup | `source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"` |
| Inline comment | `gh api repos/$OWNER/$REPO/pulls/N/comments --method POST` (head SHA as `commit_id`) |
| Formal review | `gh pr review N --approve\|--request-changes\|--comment --body "..."` |
| Top-level summary | `gh pr comment N --body "..."` (format: `references/review-output-template.md`) |
| Cleanup | `git checkout main && git branch -D pr-N` |

## Procedure

### A. Local pre-push review

When the user asks to "review the code" or "check before pushing" (pure `git`, no API):

1. Get the big picture first and state the scope ("N commits, M files") — a scan over an
   empty diff means "nothing to review", not "clean":
   ```bash
   git diff main...HEAD --stat
   git log main..HEAD --oneline
   ```
2. Review file by file — use `read_file` on changed files for full context:
   `git diff main...HEAD -- src/auth/login.py`
3. Check for common issues:
   ```bash
   # Debug statements, TODOs, console.logs left behind
   git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|XXX\|debugger"
   # Large files accidentally staged
   git diff main...HEAD --stat | sort -t'|' -k2 -rn | head -10
   # Secrets or credential patterns
   git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"
   # Merge conflict markers
   git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
   ```
4. Apply the review checklist (below).
5. Present findings in the structured format from
   [references/review-output-template.md](references/review-output-template.md)
   (Critical / Warnings / Suggestions / Looks Good). If critical issues are found, offer
   to fix them before the user pushes.

### B. PR review (end to end)

When the user asks to "review PR #N", "look at this PR", or gives a PR URL:

1. **Set up environment** — the Setup block above, or
   `source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"`.
2. **Gather PR context** (metadata, changed files, CI state):
   `gh pr view N`, `gh pr diff N --name-only`, `gh pr checks N`.
3. **Check out the PR locally** for full access to `read_file`, `search_files`, and the
   ability to run tests:
   `git fetch origin pull/N/head:pr-N && git checkout pr-N` (shortcut: `gh pr checkout N`).
4. **Read the full diff** against the base branch (`git diff main...HEAD`, or file by
   file for large PRs); for each changed file use `read_file` to see the surrounding
   context.
5. **Run automated checks locally** against the checked-out PR code — the project's test
   suite (`python -m pytest`, `npm test`, `cargo test`, …) and linter (`ruff check .`,
   `eslint`, `clippy`, …).
6. **Apply the review checklist** (below).
7. **Post the review** — `gh pr review N --approve|--request-changes|--comment`, or the
   atomic multi-comment REST payload in
   [references/github-rest-api.md](references/github-rest-api.md). Inline comments need
   the PR head SHA as `commit_id`; `line` is the NEW-file line number, and deleted lines
   use `side: "LEFT"` with the OLD-file line number. Anchors must sit inside a diff
   hunk — GitHub rejects off-diff lines with HTTP 422. Re-check against
   `gh pr diff N -- path/to/file`, fall back to a file-level comment
   (`subject_type: "file"`), or fold the point into the review body.
8. **Post a top-level summary comment** so the author gets the full picture at a glance —
   format from `references/review-output-template.md`.
9. **Clean up:** `git checkout main && git branch -D pr-N`.

**Decision — Approve vs Request Changes vs Comment:**

- **Approve** — no critical or warning-level issues; only minor suggestions or all clear.
- **Request Changes** — any critical or warning-level issue that should be fixed before merge.
- **Comment** — observations and suggestions, nothing blocking (also use when unsure or
  the PR is a draft). Comment is the ONLY valid verdict for an empty diff: never Approve
  or Request Changes when the PR changes 0 files — note the empty diff and its likely
  causes (already merged, branch reset to base, wrong base branch, commits never pushed)
  in a top-level comment instead.

### Review checklist

When performing a code review (local or PR), systematically check:

**Correctness** — does the code do what it claims; edge cases handled (empty inputs,
nulls, large data, concurrent access); error paths handled gracefully.

**Security** — no hardcoded secrets, credentials, or API keys; input validation on
user-facing inputs; no SQL injection, XSS, or path traversal; auth/authz checks where
needed.

**Code Quality** — clear naming; no unnecessary complexity or premature abstraction;
DRY; focused functions (single responsibility).

**Testing** — new code paths tested; happy path and error cases covered; tests readable
and maintainable.

**Performance** — no N+1 queries or unnecessary loops; appropriate caching; no blocking
operations in async code paths.

**Documentation** — public APIs documented; non-obvious logic has comments explaining
"why"; README updated if behavior changed.

## Pitfalls

- Never Approve code you haven't actually read, and never call a scan over an empty diff
  "clean" — state the scope so "no findings" is meaningful.
- Never Approve or Request Changes on an empty diff — a top-level Comment is the only
  vehicle there.
- Don't post anything to GitHub without confirmed auth, and never echo token values into
  outputs or logs.
- Inline anchors outside a diff hunk fail with HTTP 422 — verify the line against the PR
  diff first, then fall back (nearest changed line → file-level comment → review body).
- Run tests against the checked-out PR code, not the user's working tree — and always
  clean up (`git checkout main`, delete `pr-N` branches) afterward.
- Use `read_file` for context around every changed hunk; diffs alone miss issues.

## Verification

Before delivering, confirm:
- [ ] Scope stated (N commits, M files) and full diff read with surrounding context.
- [ ] Tests/linter run against the checked-out PR code — or explicitly reported as not run.
- [ ] Review checklist applied across all six categories.
- [ ] Every posted item confirmed by the API response (comment or review URL).
- [ ] Verdict matches the findings — empty diff → Comment only.
- [ ] Cleanup done: original branch restored, `pr-N` branch deleted.
