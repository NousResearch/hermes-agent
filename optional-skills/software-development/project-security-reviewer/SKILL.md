---
name: project-security-reviewer
description: "Detect a repository's toolchain and perform an evidence-based security review. Use for general project security reviews, code audits, pre-merge checks, or when a user wants a review adapted to an unfamiliar local project or Git repository."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Security, Code-Review, Audit, Software-Development, Toolchain]
    category: software-development
    related_skills: [github-code-review]
---

# Project Security Reviewer

Adapt a security review to the repository rather than applying one generic test command. Detect project manifests first, select only compatible local tooling, and produce a Markdown report backed by observed evidence.

## When to Use

- User asks to audit, security-review, or check a project before merge or release.
- User provides an unfamiliar repository and wants the review adapted to its stack.
- User asks for a general code-security review rather than a framework-specific audit.

Do not use when the user only wants a GitHub PR diff review; use `github-code-review` instead. Do not replace a domain-specific review when a dedicated skill exists.

## Required Input

- Accept a local repository path, the current directory, or a Git URL.
- If the user gives only a project name, ask for its local path or Git URL. Do not guess the source repository.
- Before cloning a Git URL, ask for confirmation because cloning writes files locally.

## Procedure

1. Resolve the repository root and record the reviewed commit or branch.
2. Detect manifests and recommended workflows without executing project code:

   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/detect_project.sh <project-root>
   ```

3. Read the top-level README, dependency manifests, lockfiles, CI configuration, and security policy before selecting checks.
4. Use the detected workflow from [toolchain workflows](references/toolchain-workflows.md). Prefer commands already declared by the project and tools already on `PATH`.
5. Delegate when a specialized skill applies:

   - `foundry.toml` — use an installed Foundry-specific review workflow; when `foundry-security-reviewer` is available, use it for build, Forge tests, coverage, gas snapshots, and Solidity analyzers.
   - A GitHub PR-only request — use `github-code-review`.
   - A deployed web application rather than source code — use `web-pentest` when appropriate and authorized.

6. Do not install packages, run deployment commands, mutate databases, use production credentials, or transmit source code to external scanners without explicit approval.
7. Run relevant tests, linters, and installed dependency/security scanners. Capture failures as evidence; do not mark an unrun check as passed.
8. Inspect authentication, authorization, input validation, secrets, dependency risk, unsafe deserialization, injection paths, concurrency, and the project's domain-specific trust boundaries.
9. Separate confirmed vulnerabilities from hypotheses and false positives. Include a concrete exploit path, affected file/function, impact, and remediation for every confirmed finding.

## Output Format

Return a Markdown report suitable for a PR comment:

```markdown
## Project Security Review

**Scope:** `<repo>@<commit>`
**Detected stack:** `<toolchains>`

### Checks Run
- [x] `<command>` — result
- [ ] `<command>` — not run, reason

### Findings
- **HIGH** — `path/to/file:line` — impact, reachable exploit path, remediation

### Test and Dependency Coverage
- Relevant tests and lint/security-tool results
- Known gaps or unavailable scanners

### Residual Risk
- Assumptions, unreviewed components, and false-positive decisions

### Recommendations
- [ ] Concrete follow-up, owner, or regression-test action
```

## Pitfalls

- A project name is not a review target. Require a local path or an explicit repository URL.
- Monorepos can contain several manifests. Review each independent deployable component and state the scope.
- Test commands can create files, contact services, or require credentials. Inspect scripts and configuration first.
- Dependency-audit commands can need registry/network access; run them only when permitted and record unavailable results.
- Static-analysis findings are leads, not proof. Confirm reachability and impact in source before reporting severity.

## Verification

- Verify the report names the actual revision and every detected toolchain.
- Verify every checked item includes command output or a concise result.
- Verify every finding maps to source evidence and a remediation or regression-test recommendation.
