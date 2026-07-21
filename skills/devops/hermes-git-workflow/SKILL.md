---
name: hermes-git-workflow
description: "Use when Hermes maintains Git repositories and needs clean commits, pushes, remotes, and provider-level provenance for shared bot-account work."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [git, github, ssh, automation, hermes, workflow]
    related_skills: [github-pr-workflow, hermes-pr-provenance]
---

# Hermes Git Workflow

## Overview

Hermes frequently pushes through a shared machine user or bot account while the actual code was written by a routed provider family. This skill keeps Git hygiene and provenance consistent across repositories without depending on a project-specific closeout process.

## When to Use

Use this skill when Hermes:

- Creates commits or pushes branches.
- Initializes or fixes a repository remote.
- Works in a repo where a human reviews changes through GitHub.
- Needs to distinguish the GitHub actor from the provider route that wrote the change.

If a repository has stronger local workflow rules, follow those first and use this skill only for the portable baseline.

## Core Rules

- Keep the working tree understandable: inspect status before and after edits.
- Prefer the repository's documented branch, commit, and closeout workflow.
- Push when the repo workflow expects remote visibility, especially for PR-based review.
- Prefer SSH remotes when the environment already has an authenticated deploy/user key.
- Do not embed secrets in remotes, commit messages, PR bodies, or logs.
- Include provider-level `Writer:` trailers for Hermes-authored commits.

## Provider Provenance

When Hermes writes code, docs, or repo updates, the GitHub account often remains the same while the active provider route changes. Preserve that distinction with provider-level commit trailers:

```text
Writer: codex
Refs: #123
```

Rules:

- `Writer:` is the stable provider family used by automation and issue ledgers.
- Use values such as `codex`, `grok`, `claude`, `gemini`, `openrouter`, `human`, or the repository's existing writer vocabulary.
- For GPT-5.5 through OpenAI Codex, use `Writer: codex`.
- Do not add `Writer-Model:` by default; exact model/version tracking is deferred unless the user or repository explicitly asks for it.
- Keep `Writer:`, `Refs:`, and any `Co-Authored-By:` trailers contiguous at the end of commit messages with no blank lines between them.
- For PRs, include a short `## Provenance` block with GitHub actor, PR creator, implementer, writer trailers, and task ledger.
- For Beads/issue-led repos, keep fields like `implemented_by`, `closed_by`, and `pr_created_by` provider-level where the repo supports metadata.
- Load `hermes-pr-provenance` for the full portable PR/GitHub/Beads contract.

## Commit Example

```bash
git add src/auth/login.py tests/test_login.py
git commit -m "fix: correct redirect URL after login

Preserves the ?next= parameter instead of always redirecting to /dashboard.

Writer: codex
Refs: #42"
git push -u origin HEAD
```

## Provenance Helper

Before opening or updating a PR, run:

```bash
hermes-provenance-check --base origin/main --head HEAD
```

Use `--emit-pr-block` to generate a starter `## Provenance` section from commit trailers.

## Pitfalls

- Leaving changes uncommitted in a repo where the human expects to pull/review remote work.
- Using HTTPS remotes without credential helpers configured.
- Forgetting to fetch/rebase before pushing a long-lived branch.
- Encoding exact model names in `Writer:`; this can break workflows that parse `Writer:` as a small provider enum.
- Adding model-level provenance by default; keep the portable baseline provider-level unless explicitly requested.

## Verification Checklist

- [ ] `git status --short` inspected before final handoff.
- [ ] Commit messages include provider-level `Writer:` trailers when Hermes authored work.
- [ ] `Refs:` points to a GitHub issue, Beads id, PR, or task id when available.
- [ ] `hermes-provenance-check` passes for the branch or intentional exceptions are documented.
- [ ] Remote push completed when the workflow expects GitHub/remote visibility.
