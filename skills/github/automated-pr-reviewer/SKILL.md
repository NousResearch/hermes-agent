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
    related_skills: [github-code-review]
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
    "prompt": "Check for PRs to review using the automated-pr-reviewer skill.",
    "skills": ["automated-pr-reviewer"]
})
```

## The Workflow Script

When invoked, the agent should run the following bash script to find and process the comments:

```bash
#!/bin/bash

# Ensure we are in a git repository and GH CLI is authenticated
if ! command -v gh &>/dev/null || ! gh auth status &>/dev/null; then
  echo "GitHub CLI (gh) is not installed or not authenticated."
  exit 1
fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Scanning $REPO for '@jules' comments..."

# Search for open PRs and output their numbers
gh api -X GET search/issues -f q="repo:$REPO is:pr is:open in:comments \"@jules\" -label:reviewed" \
  --jq '.items[].number'

## Agent Instructions

When you are asked to "check for PRs to review" or when this skill is run:

1. Run the bash snippet above to identify the PR numbers.
2. For each PR number output:
   a. Fetch and checkout the PR: `git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER -f && git checkout pr-$PR_NUMBER`.
   b. Analyze the diff using your code review capabilities (refer to `github-code-review`).
   c. Post the review and add the `reviewed` label: `gh pr edit $PR_NUMBER --add-label "reviewed"`.
   d. Clean up: `git checkout - && git branch -D pr-$PR_NUMBER`.