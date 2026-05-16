---
name: automated-pr-reviewer
description: "Automated PR reviewer: scans for '@jules code review' comments and triggers code reviews."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Code-Review, Automation, Pull-Requests, Cron, review]
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

When invoked, the agent should run the following bash script to find and process the comments:

```bash
#!/bin/bash

# Ensure we are in a git repository and GH CLI is authenticated
if ! command -v gh &>/dev/null || ! gh auth status &>/dev/null; then
  echo "GitHub CLI (gh) is not installed or not authenticated."
  # gracefully return
  return 1 2>/dev/null || true
fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Scanning $REPO for '@jules' comments..."

# Search for open PRs
gh api -X GET search/issues -f q="repo:$REPO is:pr is:open in:comments \"@jules\" -label:reviewed" \
  --jq '.items[].number' > /tmp/prs_to_review.txt

if [ ! -s /tmp/prs_to_review.txt ]; then
  echo "No new PRs to review."
  # gracefully return
  return 0 2>/dev/null || true
fi

while read PR_NUMBER; do
  echo "Found request for PR #$PR_NUMBER. Starting review..."

  # Checkout PR locally to do the review
  git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER
  git checkout pr-$PR_NUMBER

  # Note for Agent: At this point, the agent should use the `github-code-review`
  # skill's methodology to review `git diff main...HEAD`.

  # 2. Add label to prevent duplicates
  # Ensure the label exists first
  gh api -X POST repos/$REPO/labels -f name="reviewed" -f color="0e8a16" --silent || true

  # Add label to PR
  gh pr edit $PR_NUMBER --add-label "reviewed"

  # 3. Reply to the PR acknowledging completion
  # Note for Agent: Make sure the actual code review is also submitted using the github-code-review standard.
  gh pr comment $PR_NUMBER --body "Code review completed by @jules. Added the \`reviewed\` label."

  echo "Completed PR #$PR_NUMBER"

  # Clean up branch
  git checkout -
  git branch -D pr-$PR_NUMBER

done < /tmp/prs_to_review.txt

echo "Review pass complete."
```

## Agent Instructions

When you are asked to "check for PRs to review" or when this skill is run:

1. Run the bash snippet above to identify the PR numbers.
2. For each PR number output by the script, read the diff using `git diff`.
3. Analyze the diff using your code review capabilities.
4. Post the review as a PR comment or PR Review (as detailed in `github-code-review`).
5. Ensure the label `reviewed` is added to prevent re-processing.
