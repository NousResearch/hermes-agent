---
name: open-source-maintainer-applications
description: "Prepare evidence-based OSS support applications."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [open-source, maintainer, grants, applications, github]
    related_skills: [github-pr-workflow, github-issues, github-repo-management]
---

# Open Source Maintainer Applications Skill

Use this skill when helping a user apply to maintainer programs, OSS grants, cloud or API credit programs, contributor funds, or similar support opportunities. It turns public contribution evidence into accurate application material.

This skill does not replace live verification. Program criteria, PR states, and release inclusion can drift, so re-check current public sources before final wording.

## When to Use

- Use when the user asks for an OSS program, grant, credit, fellowship, or maintainer-support application.
- Use when GitHub contribution evidence must be summarized accurately.
- Use when a closed PR may still have upstream credit through salvage, cherry-pick, or co-author metadata.
- Do not use for private employment claims or unverifiable metrics.

## Prerequisites

- The user has identified a target program or wants help choosing one.
- The relevant GitHub handle and repositories are known or can be discovered.
- `terminal` can run GitHub CLI commands when local authentication is available, or the GitHub connector can inspect public repository metadata.

## How to Run

Build the application from two tracks:

- Program fit: what the program says it supports.
- Contribution proof: what public evidence shows the user actually did.

Use live checks for claims that can change:

```bash
gh search prs --repo OWNER/REPO --author USER --limit 100 --json number,title,state,url,createdAt,updatedAt
gh pr view NUMBER --repo OWNER/REPO --json number,title,state,url,mergedAt,mergeCommit,reviewDecision
gh release list --repo OWNER/REPO --limit 100
```

## Quick Reference

| Evidence | Safe Wording |
| --- | --- |
| Open PR | "submitted upstream PRs" |
| Merged PR | "merged" only when `mergedAt` is present |
| Closed but salvaged | "salvaged or cherry-picked into commit ..." |
| Co-author credit | "co-authored upstream work" |
| Release inclusion | Claim only after tag or release containment is checked |

## Procedure

1. Confirm the user's GitHub handle and the repositories that matter.
2. Read the target program's official page, terms, and application prompts.
3. Collect PR, issue, commit, and release evidence with URLs.
4. Separate direct merges, open submissions, closed work, salvaged commits, and co-author credit.
5. Draft a one-page narrative around maintainer workload, not hype.
6. Add an evidence appendix with exact links and current states.

## Pitfalls

- Do not call a closed PR merged unless public metadata says it merged.
- Do not claim release inclusion without checking the relevant tag or release line.
- Do not inflate application claims with private or unverified work.
- Do not quote stale counts from prior sessions without re-checking.
- Do not confuse "built a fork" with "maintains a workflow that helps OSS maintainers."

## Verification

- Every material claim has a URL or command-backed evidence note.
- Open, merged, closed, salvaged, and co-authored work are labeled separately.
- Dates and program eligibility are current at the time of submission.
- The final application explains what support will unlock for real OSS maintenance.

## References

- `references/codex-for-oss-hermes-evidence.md`
