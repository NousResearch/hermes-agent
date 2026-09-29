---
name: forgejo
description: Deliver Forgejo issues as tested pull requests.
version: "1.0.0"
author: "cloudkoopa, Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [forgejo, git, issues, pull-requests, coding]
    category: software-development
    related_skills: [codebase-inspection, test-driven-development]
---

# Forgejo Skill

Use this skill to carry a Forgejo issue through a disposable coding task and
create one pull request. It composes configured `forgejo_*` MCP tools with
`terminal`, `read_file`, `search_files`, and `patch`; it does not merge,
deploy, or create an authorization layer.

## When to Use

Use when the user names an existing Forgejo issue and requests implementation.
Do not use for reviewing an existing pull request or for live deployment.

## Prerequisites

- Forgejo MCP tools are configured and expose the required issue, repository,
  branch, and pull-request operations.
- `terminal.backend: agent_sandbox` is configured with the reviewed immutable
  image and namespace permissions.
- The task-bound checkout/publication credential path is configured and
  verified. Never place a token in a prompt, command, file, or log.

## How to Run

Read `references/issue-to-pr.md` before starting. Ask no chat-side approval for
ordinary Forgejo actions that the configured credential permits. Stop before
checkout or publication when the credential path is missing or unverified.

## Quick Reference

| Step | Required result |
|---|---|
| Read | Issue, comments, repository instructions, and relevant source are read. |
| Isolate | One disposable Agent Sandbox owns the checkout. |
| Change | The smallest applicable change is made. |
| Test | Applicable repository tests run and their real result is recorded. |
| Publish | One branch and one Forgejo pull request are created and linked. |
| Report | Matrix receives the URL, branch, changed files, and test result. |

## Procedure

Follow the complete procedure in `references/issue-to-pr.md`. The first release
stops after pull-request creation. no merge is performed. Do not merge or perform a merge, do not merge via automation, deploy, mutate Kubernetes, or
change Hermes identity/PVC/Hindsight state.

## Pitfalls

- Do not treat a client-side tool filter as Forgejo authorization.
- Do not use the Hermes PVC or host filesystem as a checkout.
- Do not reuse a stale Sandbox or an unrelated branch.
- fail closed and do not report success after a failed test, checkout, push, or pull request.
- Do not print credentials or claim production verification.

## Verification

Report each completed step, changed files, test result, and the exact failed step when the workflow stops.
The final Matrix report must include the pull-request URL, branch, changed
files, test command/result, and any remaining failure. A pull request is not a
merge or deployment result.
