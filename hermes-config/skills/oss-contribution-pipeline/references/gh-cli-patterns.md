# GitHub CLI Patterns for the Moss Pipeline

Commands refined through real usage. These are non-obvious and easy to get wrong.

## Duplication Detection — Check if Issue Already Has a PR

**Always search BOTH patterns.** `gh search prs "Fixes #N"` catches PRs with standard linking syntax. `gh search prs "#N"` (without `Fixes`) catches PRs that mention the issue in the title/body without the keyword. Some PRs use `Resolves #N`, `Closes #N`, or just reference the number inline.

```bash
# Standard — matches "Fixes #52042" in body
gh search prs "Fixes #52042" --repo NousResearch/hermes-agent --state open --json number,url,author,title

# Broad — matches bare "#52042" anywhere in title or body
gh search prs "#52042" --repo NousResearch/hermes-agent --state open --json number,url,author,title
```

If either returns a match by someone other than `Sahil-SS9`, skip the issue.

## Issue State Check

```bash
gh issue view <number> --repo NousResearch/hermes-agent --json state,title
```

`state` can be `OPEN` or `CLOSED`. `closedByPullRequests` is NOT a valid field on issues — only on PRs. Do not request it.

## PR Creation from Fork

When creating a PR from a fork (separate remote), use `--head`:

```bash
gh pr create \
  --repo NousResearch/hermes-agent \
  --head Sahil-SS9:<branch-name> \
  --title "fix(area): description (#<number>)" \
  --body "Fixes #<number>\n\n## Description\n..." \
  --label "type/bug"
```

**Note:** Label permission errors on fork PRs are expected — the PR still creates successfully, the label assignment just fails silently. Check `gh pr view <number> --json state` to confirm.

## PR Activity — Tracked Authors Only

Use `--author` at the API level, not post-fetch Python filtering:

```bash
gh pr list --repo NousResearch/hermes-agent \
  --state all --author Sahil-SS9 \
  --json number,title,state,author,createdAt,updatedAt,mergedAt,closedAt,url,labels,comments,reviews \
  --limit 50
```

The `--author` flag is per-user. To track multiple authors, loop or parallelise the calls — do NOT batch and filter in Python (you'll lose PRs beyond the --limit window).

## Test Baseline Validation — Detecting Introduced Failures

When a subagent reports a test failure as "pre-existing" or "unrelated to its change", verify rather than trust. Run the failing test(s) against the unmodified baseline:

```bash
# Capture baseline test state before any changes
git stash --message "pre-fix-baseline-$(date +%s)"
pytest <path-to-failing-test> -v -q 2>&1 | tail -5
echo "Exit: $?"
git stash pop
```

If the baseline passes, the failure was introduced by the subagent's changes. Do NOT accept "pre-existing" as a diagnosis without this check.

This is especially important when the subagent changed files that are only tangentially related to the fix — the test that breaks may be testing a different function that the subagent inadvertently modified.

**Pipeline integration:** The pipeline prompt should include a step: "Before delegating the fix, record the current test state for affected modules. After the subagent reports results, re-run any failing tests against the baseline to confirm they are truly pre-existing."

## Issue Listing with Classification Labels

```bash
gh issue list --repo NousResearch/hermes-agent \
  --state open \
  --json number,title,labels,createdAt,url,body \
  --limit 30
```

Classification logic (from `moss-issue-watch.py`):
- Label `type/bug` → `bug` (actionable)
- Label `type/security` → `security` (actionable)
- Label `needs-repro` with reproduction steps in body → `bug`
- Label `needs-repro` without reproduction steps → skip
- Label `type/feature` with short body → skip (vague feature request)
- Priority: P1 > P2 > P3 from labels; default to P3

## Author Configuration

Git commits should be authored as the human, not the agent:

```bash
# Set at repo level (overrides any stale global/local config)
cd ~/repos/hermes-agent-upstream
git config user.name "Sahil-SS9"
git config user.email "218421507+Sahil-SS9@users.noreply.github.com"
```

This email must match the upstream AUTHOR_MAP entry in `scripts/release.py`. Verify with `git log --format="%an <%ae>" -1` before pushing. If the commit email doesn't match, the CI `contributor-check / check-attribution` gate will fail.

**Pitfall:** A stale `user.email` set at the repo level (e.g. `sahil@example.com`) will override the global config. Always verify at the repo level before committing. After fixing, verify with: `git config user.email` inside the repo.

```bash
# Inside the upstream clone — check what email will actually be used
git config user.email
# Expected: 218421507+Sahil-SS9@users.noreply.github.com
```

If it's wrong, fix it at repo level:
```bash
git config user.email "218421507+Sahil-SS9@users.noreply.github.com"
git config user.name "Sahil-SS9"
```

Then verify with `git log --format="%an <%ae>" -1` before pushing. The pipeline cron's workdir is this clone, so repo-level config is what matters.

**Recovery if pushed with wrong email:** amend + force-push (safe on a solo fork branch):
```bash
git commit --amend --author="Sahil-SS9 <218421507+Sahil-SS9@users.noreply.github.com>" --no-edit
git push fork HEAD --force-with-lease
```

## Branch Naming Convention

```
fix/issue-<number>-<kebab-slug>
```

Slug from issue title: lowercase, replace non-alphanumeric with hyphens, collapse repeated hyphens, truncate at ~6 words.
