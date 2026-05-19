---
name: automated-pr-reviewer
description: "Automated PR reviewer: scans for authorized '@jules' PR comments and triggers static code reviews."
version: 1.0.2
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Code-Review, Automation, Pull-Requests, Cron, review]
    related_skills: [github-code-review, cronjob]
---

# Automated PR Reviewer Workflow

This skill sets up a scheduled workflow that monitors GitHub repository Pull Request comments for the exact phrase `@jules`. A matching comment is only a review request when it was posted by a trusted repository participant and the PR branch is from the same repository. Authorized requests are reviewed using a static diff approach, and the PR is labeled only after the review is safely completed.

## How It Works

1. **Trigger**: A scheduled job runs periodically (e.g., via cron).
2. **Scan**: It queries GitHub for open PR comments mentioning `@jules` (all pages).
3. **Authorize**: It confirms at least one matching comment was made by an `OWNER`, `MEMBER`, or `COLLABORATOR` and that the comment is a genuine trigger request (not just a mention of `@jules`).
4. **Trust Source**: It only reviews PRs whose head branch comes from the same repository, skipping forks by default.
5. **Filter**: It filters out PRs that already have the `jules-reviewed` label.
6. **Action**: For each authorized PR, the agent performs a static code review from the diff.
7. **Mark Done**: Once the review completes, it ensures the `jules-reviewed` label exists, adds it to the PR, and posts the review comment.

## Security Requirements

- Do not treat the search result alone as authorization. Always validate the triggering comment's `author_association` before reviewing.
- Do not fetch, checkout, build, test, lint, or otherwise execute code from automated PR review requests.
- Do not review fork PRs in this automation unless a separate sandboxed workflow has explicitly opted in to that risk.
- Do not add labels or acknowledgements until the safe static review has completed.
- Only treat a comment as a trigger when `@jules` appears as a standalone token (not as part of a username like `@jules-bot`).

## Setting Up the Automation

Use the built-in `cronjob` tool to schedule this workflow. For example, to run it every hour:

```python
cronjob(
    "create",
    {
        "name": "jules-github-pr-reviewer",
        "schedule": "0 * * * *",
        "command": "hermes run automated-pr-reviewer",
    },
)
```

## The Workflow Script

When invoked, the agent should run the following bash script to find authorized review requests. It writes PR numbers to a temporary file and prints each safe diff without checking out PR code.

```bash
#!/bin/bash
set -euo pipefail

# Ensure GH CLI is installed and authenticated
if ! command -v gh &>/dev/null || ! gh auth status &>/dev/null; then
  echo "GitHub CLI (gh) is not installed or not authenticated."
  exit 1
fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
PRS_TO_REVIEW=$(mktemp "${TMPDIR:-/tmp}/hermes-prs-to-review.XXXXXX")
trap 'rm -f "$PRS_TO_REVIEW" "$PRS_TO_REVIEW.candidates"' EXIT

echo "Scanning $REPO for authorized '@jules' PR review requests..."

# Search all pages for open PRs with matching comments, excluding PRs already reviewed.
gh api --paginate -X GET search/issues \
  -f q="repo:$REPO is:pr is:open in:comments \"@jules\" -label:jules-reviewed" \
  --jq '.items[].number' > "$PRS_TO_REVIEW.candidates"

while read -r PR_NUMBER; do
  [ -n "$PR_NUMBER" ] || continue

  # Require the PR branch to come from this repository. Forks are skipped because
  # automated review must not process attacker-controlled code paths locally.
  HEAD_REPO=$(gh api "repos/$REPO/pulls/$PR_NUMBER" --jq '.head.repo.full_name')
  if [ "$HEAD_REPO" != "$REPO" ]; then
    echo "Skipping PR #$PR_NUMBER: head repository '$HEAD_REPO' is not trusted."
    continue
  fi

  # Require at least one genuine trigger comment (exact @jules token, not @jules-bot etc.)
  # from a trusted repository participant. Negative lookahead/lookbehind ensure @jules
  # is not part of a longer username (e.g. @jules-bot, @jules_jr are rejected).
  TRUSTED_COMMENT_COUNT=$(gh api --paginate "repos/$REPO/issues/$PR_NUMBER/comments" \
    --jq '.[] | select((.body // "") | test("(?<![a-zA-Z0-9_-])@jules(?![a-zA-Z0-9_-])")) | select(.author_association == "OWNER" or .author_association == "MEMBER" or .author_association == "COLLABORATOR") | .id' \
    | wc -l | tr -d ' ')

  if [ "$TRUSTED_COMMENT_COUNT" -eq 0 ]; then
    echo "Skipping PR #$PR_NUMBER: no trusted @jules trigger comment found."
    continue
  fi

  echo "$PR_NUMBER" >> "$PRS_TO_REVIEW"
done < "$PRS_TO_REVIEW.candidates"

rm -f "$PRS_TO_REVIEW.candidates"

if [ ! -s "$PRS_TO_REVIEW" ]; then
  echo "No authorized PRs to review."
  exit 0
fi

# Ensure the jules-reviewed label exists before we attempt to apply it.
gh api -X POST "repos/$REPO/labels" \
  -f name="jules-reviewed" -f color="0e8a16" \
  --silent 2>/dev/null || true

while read -r PR_NUMBER; do
  echo "Authorized request for PR #$PR_NUMBER. Generating static diff..."

  # Review the patch without checking out or executing PR code.
  gh pr diff "$PR_NUMBER"

  echo "After completing the static review, post the review and then run:"
  echo "  gh pr edit $PR_NUMBER --add-label jules-reviewed"
  echo "Do not label or acknowledge PR #$PR_NUMBER before the review is complete."
done < "$PRS_TO_REVIEW"

echo "Review pass complete."
```

## Agent Instructions

When you are asked to "check for PRs to review" or when this skill is run:

1. Run the bash snippet above to identify authorized PR numbers and display their diffs.
2. For each authorized PR number output by the script, analyze only the static diff and any needed base-branch context.
3. Do not checkout the PR branch and do not run tests, linters, package scripts, build commands, hooks, or other commands from the PR.
4. Post the review as a PR comment or PR Review based only on the static diff and permitted base-branch context gathered without checking out or executing PR code.
5. After the review is posted successfully, add the `jules-reviewed` label to prevent re-processing.
