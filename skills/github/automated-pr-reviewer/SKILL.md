---
name: automated-pr-reviewer
description: "Automated PR reviewer: scans for '@jules code review' comments and triggers code reviews."
version: 1.0.0
author: badMade
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Code-Review, Automation, Pull-Requests, Cron]
    related_skills: [github-code-review, cronjob]
---

# Automated PR Reviewer Workflow

This skill sets up a scheduled workflow that monitors GitHub repository Pull Request comments for the exact phrase `@jules`. When triggered, it delegates the review of that PR to the agent using the existing `github-code-review` skill, and leaves a label on the PR to prevent duplicate reviews.

## How It Works

1. **Trigger**: A scheduled job runs periodically (e.g., via cron).
2. **Scan**: It queries GitHub for PR comments mentioning `@jules`.
3. **Filter**: It filters out PRs that already have the `reviewed` label.
4. **Action**: For each matching PR, the agent performs a comprehensive code review.
5. **Mark Done**: Once reviewed, it adds the `reviewed` label to the PR and posts the review comment.

## Security Requirements

- Do not treat the search result alone as authorization. Always validate the triggering comment's `author_association` before reviewing.
- Do not fetch, checkout, build, test, lint, or otherwise execute code from automated PR review requests.
- Do not review fork PRs in this automation unless a separate sandboxed workflow has explicitly opted in to that risk.
- Do not add labels or acknowledgements until the safe static review has completed.
- Only treat a comment as a trigger when `@jules` appears as a standalone token (not as part of a username like `@jules-bot`).

## Setting Up the Automation

Use the built-in `cronjob` tool to schedule this workflow. For example, to run it every hour:

```python
cronjob("create", {
    "name": "jules-github-pr-reviewer",
    "schedule": "0 * * * *",
    "command": "hermes run automated-pr-reviewer"
})
```

## The Workflow Script

When invoked, the agent should run the following bash script to find authorized review requests. It writes PR numbers to a temporary file and prints each safe diff without checking out PR code.

```bash
#!/bin/bash
set -euo pipefail

main() {
  # Ensure we are in a git repository and GH CLI is authenticated
  if ! command -v gh &>/dev/null || ! gh auth status &>/dev/null; then
    echo "GitHub CLI (gh) is not installed or not authenticated."
    # gracefully return inside the main function
    return 1 2>/dev/null || true
  fi

  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
  # Use mktemp to prevent concurrent execution collisions
  PRS_TO_REVIEW=$(mktemp "${TMPDIR:-/tmp}/hermes-prs-to-review.XXXXXX")
  trap 'rm -f "$PRS_TO_REVIEW" "$PRS_TO_REVIEW.candidates"' EXIT

  echo "Scanning $REPO for authorized '@jules' PR review requests..."

  # Search all pages for open PRs with matching comments, excluding PRs already reviewed.
  gh api --paginate -X GET search/issues \
    -f q="repo:$REPO is:pr is:open in:comments \"@jules\" -label:reviewed" \
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
    # gracefully return inside the main function
    return 0 2>/dev/null || true
  fi

  # Ensure the reviewed label exists before we attempt to apply it.
  gh api -X POST "repos/$REPO/labels" \
    -f name="reviewed" -f color="0e8a16" \
    --silent 2>/dev/null || true

  while read -r PR_NUMBER; do
    echo "Authorized request for PR #$PR_NUMBER. Generating static diff..."

    # Review the patch without checking out or executing PR code.
    gh pr diff "$PR_NUMBER"

    echo "After completing the static review, post the review and then run:"
    echo "  gh pr edit $PR_NUMBER --add-label reviewed"
    echo "Do not label or acknowledge PR #$PR_NUMBER before the review is complete."
  done < "$PRS_TO_REVIEW"

  echo "Review pass complete."
}

# Execute main function
main
```

## Agent Instructions

When you are asked to "check for PRs to review" or when this skill is run:

1. Run the bash snippet above to identify the PR numbers.
2. For each authorized PR number output by the script, analyze only the static diff and any needed base-branch context.
3. Do not checkout the PR branch and do not run tests, linters, package scripts, build commands, hooks, or other commands from the PR.
4. Post the review as a PR comment or PR Review based only on the static diff and permitted base-branch context gathered without checking out or executing PR code.
5. After the review is posted successfully, add the `reviewed` label to prevent re-processing.