---
name: claude-code
description: "Delegate coding to Claude Code CLI (features, PRs)." Use when working with claude code.
version: 2.2.0
author: Hermes Agent + Teknium
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, Code-Review, Refactoring, PTY, Automation]
    related_skills: [codex, hermes-agent, opencode]
---

## Routing

Use this skill when its description matches the request. Read [references/full-guide.md](references/full-guide.md) before acting; it contains the complete domain, safety, troubleshooting, and verification guidance preserved from the prior skill. Follow any more-specific referenced documents linked there.

## Purpose

Provide bounded, evidence-producing autonomous-agent work for the scope named by this skill.

## Prerequisites

- Read the full guide and project-local instructions before execution.
- Confirm approved credentials, tools, runtime, and writable scope; never guess endpoints or secrets.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `run_script()` | Execute a repository-provided validator, test, or smoke script after inspecting it | Exact documented path and arguments only |
| `skillevaluator quality-check` | Validate this skill and preserve the report | Skill directory |

Example: `run_script("./scripts/<validator>.sh", "--help")`; inspect the script first and record stdout, stderr, and exit status.

## Troubleshooting

| Error | Response |
|---|---|
| Connection, authentication, or protocol mismatch | Verify the live server/configuration, capture stderr and exit status, and stop rather than masking it with retries. |
| Timeout or missing dependency | Use a bounded retry only for documented transient failures; otherwise preserve evidence and diagnose installation/configuration. |
| Unexpected or partial result | Quarantine it, reconcile scope and source paths, and do not promote it as verified. |

## Limitations

Receipts, registrations, process listings, generated plans, and zero-test runs do not by themselves prove execution, source truth, deployment, or correctness. Live provider/model/runtime behavior must be verified at the time of use.

## Procedure

1. Establish scope, authority, prerequisites, and approved tools.
2. Inspect the live environment and source/configuration before making claims or changes.
3. Execute the bounded workflow in the full guide, keeping credentials and sensitive data out of commands, logs, prompts, and receipts.
4. Capture IDs, paths, exit codes, timestamps, and artifacts; distinguish proposed, implemented, deployed, and independently verified results.
5. On timeout, authentication/protocol mismatch, missing dependency, or unexpected output, preserve stderr and evidence, stop masking retries, and diagnose the boundary.

## Dated claims

Any version, port, capacity, model, provider, platform behavior, incident, or other time-sensitive statement in the reference is evidence-dated, not a current guarantee. Before relying on it, re-check the live configuration, command help, endpoint, process, or authoritative source and record the verification date and result. If live verification is unavailable, label the statement unverified and do not use it as an admission gate.

## Verification

Run the repository/project validator or quality evaluator available in the current environment; inspect its documented arguments first and preserve stdout, stderr, and exit status. Verify scope, changed files, dependency paths, and the real consumer/runtime. Report separately: verified complete; implemented but not deployed/proven; blocked; not started. Never claim success from a generated plan, registration, process listing, receipt integrity flag, or zero-test run alone.

## Reference

- Complete preserved guidance: [references/full-guide.md](references/full-guide.md)
