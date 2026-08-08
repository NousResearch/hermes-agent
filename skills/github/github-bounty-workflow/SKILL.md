---
name: github-bounty-workflow
description: "Use when working on authorized GitHub bounty, security advisory, or paid issue workflows that need scope checks, code changes, private disclosure, PR evidence, and payout-safe handoff."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Bug-Bounty, Security, Pull-Requests, Disclosure, Compliance]
    related_skills: [github-auth, github-issues, github-pr-workflow, github-code-review]
---

# GitHub Bounty Workflow

## Overview

Use this workflow to turn an authorized GitHub bounty, paid issue, or security
advisory opportunity into a small, reviewable code change plus the evidence a
program owner needs to evaluate it.

This skill does not promise income. No payout is guaranteed. It helps prepare a
submission that may qualify for review when the work is in-scope, useful, and
accepted by the program owner.

## When to Use

- A user asks to earn or submit a GitHub bounty through code changes.
- A repository has a paid issue, bounty label, GitHub Security Advisory path,
  or published bug bounty rules.
- A vulnerability or hardening opportunity needs private vulnerability reporting
  before any public issue or PR.
- A bounty task needs a clean branch, tests, evidence, and a concise submission.

Do not use this for out-of-scope testing, spam submissions, duplicate farming,
credential abuse, bypassing rate limits, extortion, or public disclosure of a
private vulnerability.

## Required Preflight

Before testing or changing code, record these facts:

```text
Bounty / issue:
Repository:
Program URL or SECURITY.md:
Explicit authorization:
In-scope assets:
Out-of-scope assets:
Safe harbor terms:
Disclosure channel:
Rate limits / prohibited testing:
Data sensitivity:
Expected deliverable:
Payout contact, if public and user-provided:
```

Rules:

- Continue only with explicit authorization from the program, maintainer, issue,
  or written scope.
- Treat scope as deny-by-default. If an asset is not clearly in-scope, do not
  test it.
- Prefer a minimal local reproduction over live probing.
- Do not open a public issue for a suspected private vulnerability.
- Use private vulnerability reporting, a GitHub Security Advisory draft, the
  platform report form, or the contact listed in `SECURITY.md`.
- Store secrets only in the approved secret store. Never put GitHub tokens,
  platform API keys, private keys, seed phrases, bank data, tax IDs, or payout
  credentials in the repo, report, logs, tests, screenshots, or commit history.
- A public payout address may be copied into the final handoff only when the
  user explicitly provides it and labels it public.

## Task Selection

Pick small work that can be verified quickly:

1. Confirm the issue or advisory is still open and not already solved.
2. Prefer fixes with a narrow diff: validation, auth checks, dependency
   hardening, crash fixes, docs for security-sensitive misconfiguration, CI
   checks, or regression tests.
3. Avoid speculative rewrites, noisy formatting, generated churn, and unrelated
   refactors.
4. If the repo has no bounty program, frame the work as an ordinary PR, not a
   bounty claim.

Fast triage commands:

```bash
git status --short --branch
git remote -v
gh issue list --search "bounty OR paid OR reward OR security" --state open
gh issue view <number> --comments
test -f SECURITY.md && sed -n '1,220p' SECURITY.md
```

## Implementation Loop

Use a conservative branch and evidence-first workflow:

```bash
git switch -c fix/<short-bounty-topic>

# Reproduce or inspect the issue.
# Add the smallest failing test or check first.
# Implement the smallest fix.

pytest <targeted-tests>
npm test -- <targeted-tests>
git diff --check
git status --short
```

When the change is ready:

```bash
git add <changed-files>
git commit -m "fix: <short description>"
git push -u origin HEAD
```

If the finding is not private, open a PR:

```bash
gh pr create \
  --title "fix: <short description>" \
  --body-file /tmp/pr-body.md \
  --base main \
  --head "$(git branch --show-current)"
```

If it is a private vulnerability, submit privately first and wait for guidance
before creating a public PR.

## Evidence Pack

Prepare an evidence pack that is useful without being needlessly weaponized:

```text
Summary:
Impact:
Affected component:
Affected versions / commits:
Scope authorization:
Reproduction steps:
Expected result:
Actual result:
Fix summary:
Tests run:
Residual risk:
Disclosure channel used:
Timeline:
Payout note: no payout is guaranteed; include only public payout metadata the user explicitly provided.
```

Good evidence includes:

- exact commit SHA and branch name
- test names and command output summaries
- before/after behavior
- screenshots only when they do not expose secrets or private user data
- links to private reports or advisories when available to the maintainer

## Private Report Template

Use this for Security Advisory, HackerOne/Bugcrowd-style, email, or other
private channels:

```markdown
## Summary
<One paragraph.>

## Authorization and Scope
- Program / policy: <link>
- In-scope asset: <asset>
- Safe harbor / rules considered: <notes>

## Impact
<Who is affected and what can happen.>

## Reproduction
1. <Step>
2. <Step>
3. <Observed result>

## Fix
<Patch or PR link if the program allows it.>

## Validation
- <test command>: <result>

## Disclosure Notes
I have not opened a public issue for this finding. Please advise whether you
prefer a private patch, coordinated public PR, or advisory flow.
```

## PR Description Template

Use this when the work is safe to submit publicly:

```markdown
## Summary
- <What changed>
- <Why it matters>

## Scope / Bounty Context
- Related issue or program: <link>
- I assume this is public-safe because: <reason>

## Test Plan
- [ ] <command>

## Notes
No payout is guaranteed. This PR is submitted for maintainer review under the
linked program or issue terms.
```

## Review Gate

Before publishing, check:

- Does the report mention explicit authorization and in-scope assets?
- Does it avoid exploit instructions beyond what is necessary to reproduce?
- Does it avoid secrets, private keys, seed phrases, bank data, tax IDs, and
  raw tokens?
- Is the fix small enough for maintainers to review?
- Are tests targeted and repeatable?
- If private, was the private channel used before any public PR or issue?
- If public, is the issue already public and non-sensitive?

## Common Pitfalls

1. **Guaranteed-income language.** Say "may qualify" or "eligible for review",
   not that the user will earn money.
2. **Public disclosure by default.** Security bugs start private unless the
   program explicitly says otherwise.
3. **Scope drift.** Forks, cloud deployments, dependencies, and third-party
   services may not be covered.
4. **Secret leakage.** Redact tokens, cookies, account identifiers, logs, and
   screenshots before attaching evidence.
5. **Noisy diffs.** Bounty reviewers reward clear fixes more than broad churn.
6. **Duplicate reports.** Search existing issues, advisories, and recent PRs
   before investing time.

## Verification Checklist

- [ ] Authorization and scope recorded.
- [ ] Public vs private disclosure path chosen.
- [ ] Failing reproduction or test captured before the fix where practical.
- [ ] Minimal code/docs/CI change completed.
- [ ] Targeted tests pass.
- [ ] Evidence Pack completed.
- [ ] Private Report Template or PR Description Template filled in.
- [ ] Secrets and payout-sensitive data redacted.
- [ ] Branch, commit, and PR/report link recorded.
