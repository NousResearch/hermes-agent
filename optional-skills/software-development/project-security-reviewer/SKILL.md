---
name: project-security-reviewer
description: "Review unfamiliar projects for security risks."
version: 1.0.0
author: Ahmet Osrak (Osraka), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Security, Code-Review, Audit, Software-Development, Toolchain]
    category: software-development
    related_skills: [github-code-review]
---

# Project Security Reviewer Skill

Adapt a security review to the repository's detected toolchain instead of
running one generic command set. This skill inspects local project evidence
and reports confirmed findings, test gaps, and unavailable checks; it does
not install dependencies, contact production systems, or replace a
domain-specific audit skill.

## When to Use

- The user requests a security review of a local project or Git repository.
- The repository has an unfamiliar or mixed toolchain.
- The user wants evidence suitable for a pre-merge or release review.

Do not use this for a GitHub PR-only diff review; use `github-code-review`.
Do not replace a domain-specific skill when one is available.

## Prerequisites

- A local repository path or an explicit Git URL.
- The `terminal`, `read_file`, and `search_files` tools.
- Bash on Linux or macOS for the bundled manifest detector.
- Optional project tools such as Foundry, Cargo, pytest, or dependency scanners
  when they are already installed and the project documents their use.

If the user provides only a project name, ask for its local path or Git URL.
Ask for confirmation before cloning a Git URL because cloning writes locally.

## How to Run

From the target repository, invoke the detector through the `terminal` tool:

```text
bash ${HERMES_SKILL_DIR}/scripts/detect_project.sh <project-root>
```

Then use `read_file` and `search_files` to inspect the README, manifests,
lockfiles, CI configuration, security policy, and the workflows listed by the
detector. Run only compatible commands that are already available on `PATH`.

## Quick Reference

| Stage | Action |
| --- | --- |
| Resolve | Record the repository path and reviewed commit or branch. |
| Detect | Run `scripts/detect_project.sh` without executing project code. |
| Inspect | Read manifests, lockfiles, CI, README, and security policy. |
| Select | Choose documented tests, linters, and installed scanners. |
| Review | Check auth, input boundaries, secrets, dependencies, and concurrency. |
| Report | Separate confirmed findings from hypotheses and unavailable checks. |

## Procedure

1. Resolve the repository root and record the reviewed revision.
2. Run `scripts/detect_project.sh <project-root>` through `terminal`.
3. Read the top-level README, dependency manifests, lockfiles, CI files, and
   security policy before selecting commands.
4. Follow the matching workflow in
   [references/toolchain-workflows.md](references/toolchain-workflows.md).
   Prefer commands declared by the project and tools already on `PATH`.
5. If `foundry.toml` is present, delegate build, Forge tests, coverage, gas
   snapshots, and Solidity analyzers to `foundry-security-reviewer` when it is
   installed.
6. If the request is a GitHub PR-only review, use `github-code-review` instead.
7. Do not install packages, run deployments, mutate databases, use production
   credentials, or send source code to external scanners without approval.
8. Run relevant tests, linters, and installed scanners. Record failures as
   evidence and never mark an unrun check as passed.
9. Inspect authentication, authorization, validation, secrets, unsafe
   deserialization, injection paths, concurrency, and trust boundaries.
10. For every confirmed finding, record the affected path or function, impact,
    reachable exploit path, and remediation or regression-test recommendation.

## Pitfalls

- A project name is not a review target; require a local path or explicit URL.
- Monorepos can contain several independently deployable components. State the
  component scope instead of treating every manifest as one application.
- Tests can create files, contact services, or require credentials. Inspect
  their configuration before running them.
- Dependency-audit commands may need network access. Run them only when
  permitted and record unavailable results.
- Static-analysis output is a lead, not proof. Confirm reachability and impact
  before reporting severity.

## Verification

Before returning the report, verify that it names the actual revision and every
detected toolchain. Each check must include its command and result, while every
finding must map to source evidence and a remediation or regression test.

Use this report shape:

```markdown
## Project Security Review

**Scope:** `<repo>@<commit>`
**Detected stack:** `<toolchains>`

### Checks Run
- [x] `<command>` — result
- [ ] `<command>` — not run, reason

### Findings
- **HIGH** — `path/to/file:line` — impact, exploit path, remediation

### Test and Dependency Coverage
- Relevant tests and scanner results
- Known gaps or unavailable tools

### Residual Risk
- Assumptions and unreviewed components

### Recommendations
- [ ] Concrete follow-up or regression-test action
```
