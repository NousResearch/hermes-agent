---
name: bounty-target-execution
description: "Use when evaluating or executing an authorized bounty target from Algora, GitHub issues, or similar platforms, requiring target verification, TAKE gating, read-only scouting, explicit confirmation before external actions, command planning, and communication drafts."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Algora, Bounty, Safe-Execution, Pull-Requests, DevOps, Compliance]
    category: github
    related_skills: [github-bounty-workflow, github-auth, github-issues, github-pr-workflow, github-code-review]
---

# Bounty Target Execution

## Purpose

Use this skill to move from a candidate bounty into a safe execution plan. It
keeps Algora/GitHub target verification, TAKE gating, dev command planning,
DevOps validation, and maintainer communication separate so external actions
only happen after explicit user approval.

## Two-Team Architecture

### Team A: github-bounty-income-30m

Purpose: read-only scouting, scoring, and target discovery.

Allowed:

- Inspect public Algora/GitHub bounty pages.
- Read GitHub repositories, issues, comments, labels, README, CONTRIBUTING,
  SECURITY, and public metadata.
- Score candidates and produce reports.
- Produce a TAKE dossier for user review.

Not allowed:

- Claim bounties.
- Fork repositories.
- Clone into execution workspaces unless explicitly part of read-only local
  inspection approved by the user.
- Create branches, commits, pushes, or PRs.
- Comment on issues/PRs or contact maintainers.
- Start bounty implementation work.

### Team B: github-bounty-execution-team

Purpose: development, DevOps, testing, and communication drafting for a bounty
target that the user has already confirmed as TAKE.

Allowed after TAKE confirmation:

- Clone or prepare an isolated local workspace.
- Analyze the codebase and issue requirements.
- Write code and tests locally.
- Run setup, test, lint, build, and DevOps validation commands.
- Produce PR drafts, claim/comment drafts, status updates, and handoff notes.

Still requires explicit confirmation for each externally visible action:

- Claiming or requesting assignment.
- Posting comments or messages.
- Forking when it creates a public artifact.
- Pushing branches.
- Opening or submitting PRs.
- Triggering remote CI intentionally.

### Handoff Contract

Team A must hand Team B a TAKE dossier before execution starts:

```text
Source team: github-bounty-income-30m
Recommended team: github-bounty-execution-team
TAKE status: recommended / not recommended / needs user decision
Score:
Platform:
Target URL:
Repository:
Issue / bounty ID:
Reward:
Current status:
Duplicate PR / active attempt check:
Feasibility notes:
Risks:
Suggested first commands:
External actions requiring confirmation:
```

## Required Gates

### 1. Target Verification

Before any action beyond read-only inspection, verify:

```text
Platform:
Target URL:
Repository:
Issue / bounty ID:
Current status:
Reward / bounty terms:
Eligibility rules:
Assignment or claim requirements:
Required deliverable:
Deadline, if any:
Public vs private communication channel:
```

Use read-only commands first:

```bash
gh repo view OWNER/REPO
gh issue view ISSUE_NUMBER --repo OWNER/REPO --comments
gh issue list --repo OWNER/REPO --search "bounty OR algora OR reward OR paid" --state open
git ls-remote https://github.com/OWNER/REPO.git
```

For Algora targets, inspect the visible Algora bounty page and the linked
GitHub issue. Confirm the bounty is still open and not already solved or
effectively claimed before proceeding.

### 2. TAKE Gate

Before the TAKE gate, do only read-only scouting and planning.

Do not claim, comment, request assignment, fork, create branches, commit, push,
open PRs, submit forms, or contact maintainers until the user explicitly
confirms that exact action.

Ask for confirmation using this format:

```text
TAKE gate required.

I verified:
- Platform:
- Repo:
- Target:
- Status:
- Expected deliverable:
- External actions needed:

Please confirm one option:
1. TAKE: engage with this bounty target.
2. SCOUT ONLY: continue read-only investigation.
3. STOP: do not proceed.
```

Only proceed with externally visible actions if the user says `TAKE` or gives
equivalent explicit approval for the specific action.

## Read-Only Scouting

Allowed before TAKE confirmation:

- Read issue, comments, labels, bounty terms, README, CONTRIBUTING, SECURITY.
- Inspect repository structure with remote metadata or an already-present local checkout.
- Draft a plan, commands, comments, PR body, or claim message without posting.
- Identify required dev, test, packaging, and CI commands.
- Search for duplicate PRs, active attempts, maintainer guidance, and payout terms.

Do not perform these without explicit confirmation:

- Claim bounty or comment on issues/PRs.
- Request assignment.
- Fork repository.
- Clone or fetch if the user has not approved local workspace creation.
- Create branches.
- Commit, push, or open PRs.
- Submit reports or forms.
- Trigger remote CI intentionally.
- Contact maintainers externally.

## Execution Plan Template

Before implementation, produce:

~~~markdown
## Bounty Execution Plan

### Target
- Platform:
- Repo:
- Issue/Bounty:
- Status:
- Deliverable:

### Assumptions
- ...

### Read-Only Findings
- ...

### Proposed Work
- ...

### Dev Commands
```bash
# install / setup
...

# targeted tests
...

# validation
...
```

### DevOps / CI Commands
```bash
# only if needed
...
```

### External Actions Requiring Confirmation
- [ ] claim/comment
- [ ] fork
- [ ] clone/fetch
- [ ] branch
- [ ] commit
- [ ] push
- [ ] PR
~~~

## Communication Drafts

Prepare drafts before posting.

### Claim / Assignment Draft

```markdown
Hi, I would like to take this bounty. I reviewed the linked issue and plan to
work on <short plan>. I will keep the change focused and include a clear test
plan.
```

### Progress Comment Draft

```markdown
Quick update: I confirmed <finding> and am working on <fix approach>. I will
post a PR with targeted tests once validated.
```

### PR Body Draft

```markdown
## Summary
- ...

## Bounty Context
- Linked bounty/issue:
- Public-safe because:

## Test Plan
- [ ] ...

## Notes
Submitted for maintainer review under the linked bounty terms. No payout is
guaranteed.
```

## Safety Rules

- Treat scope as deny-by-default.
- Prefer local reproduction over live probing.
- Do not disclose private vulnerabilities publicly.
- Never include secrets, tokens, cookies, payout credentials, private keys, or
  personal data in comments, commits, logs, screenshots, or PRs.
- Keep diffs small and reviewable.
- Avoid noisy formatting and unrelated refactors.
- If the bounty is closed, assigned, stale, ambiguous, or duplicate, stop and
  ask the user.

## Completion Checklist

- [ ] Target verified on Algora/GitHub.
- [ ] Bounty status checked.
- [ ] TAKE gate completed before external actions.
- [ ] Read-only scouting summarized.
- [ ] Dev/devops command plan prepared.
- [ ] External actions explicitly confirmed.
- [ ] Communication draft prepared before posting.
- [ ] Tests/validation plan documented.
- [ ] Secrets and sensitive data checked.
