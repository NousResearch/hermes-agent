---
name: github-auth
description: "Use Hermes-managed GitHub authentication for repository, issue, and PR workflows."
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Authentication, Git, Skills, Security]
    related_skills: [github-pr-workflow, github-code-review, github-issues, github-repo-management]
---

# Hermes GitHub Workflow Authentication

Use Hermes' canonical GitHub workflow before performing GitHub operations. The workflow is provider-neutral: it reads the logical `GITHUB_TOKEN` through Hermes' active secret scope. It does not call Bitwarden, 1Password, or another provider directly, and it does not implement credential precedence.

## Canonical sequence

1. Detect whether the current directory or request is GitHub-related.
2. Resolve repository context from the local Git remote when present.
3. Resolve `GITHUB_TOKEN` through Hermes' secret scope.
4. For authenticated operations, verify identity with GitHub's `GET /user` preflight.
5. Check repository authorization for the requested operation.
6. Use the shared safe GitHub API/Git transport path.
7. Verify the operation independently.

Never ask the user to paste a PAT before Hermes' configured secret scope has been checked.

## Credential rules

- Do not call a provider-specific CLI or API from this skill.
- Do not implement provider precedence here.
- Do not use `git config --global credential.helper store`.
- Do not put tokens in remotes, `.git/config`, command arguments, shell history, or persistent files.
- Do not use `gh auth login` as a prerequisite.
- `gh` authentication detection may be reported by `hermes doctor`, but this workflow's canonical credential path is Hermes secret scope.
- Public repository reads may proceed without a credential when the requested GitHub operation permits unauthenticated access.

## Failure handling

Report one classified failure and its remedy. Do not cycle through unrelated `gh`, SSH, Git credential-store, and token-prompt paths.

- No credential: tell the user to configure `GITHUB_TOKEN` in a supported Hermes secret source and run `hermes secrets status`.
- Invalid/expired credential: tell the user to repair or rotate the configured GitHub credential.
- Rate limited: distinguish quota exhaustion from invalid credentials and report the retry window when available.
- Permission denied: identify the repository operation that lacks authorization.
- Network failure: report connectivity rather than blaming credentials.

## Git transport safety

Use the shared workflow transport. It must run Git non-interactively, clear ambient credential helpers for the invocation, disable system Git configuration, and deliver credentials ephemerally. Verify that credentials do not enter the remote URL, `.git/config`, argv, environment, or persistent helper stores.

## Scope

This skill provides the authentication/bootstrap contract. Domain skills own their operations:

- `github-repo-management` — repository and remote management;
- `github-issues` — issue operations;
- `github-pr-workflow` — branches, PRs, checks, and merges;
- `github-code-review` — review workflow;
- `codebase-inspection` — repository inspection.

All writes remain subject to the normal Hermes approval and explicit-user-intent rules.

## Diagnostics

Use the GitHub section of `hermes doctor`. PR1 does not add a separate `hermes github doctor` command. Diagnostic output must never include raw credentials or provider-specific secret identifiers.

## Legacy instructions removed

The following are intentionally not supported by this skill:

- persistent Git credential-store setup;
- token-bearing remote URLs;
- PATs pasted into chat;
- provider-specific secret retrieval instructions;
- `gh auth login` as the default workflow.

Use Hermes' configured secret source and the canonical workflow instead.
