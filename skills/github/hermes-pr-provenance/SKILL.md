---
name: hermes-pr-provenance
description: "Use when Hermes creates or updates Git commits, GitHub PRs/issues, or Beads records so provider-level writer provenance is recorded consistently across projects."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, git, pull-requests, issues, beads, provenance, hermes]
    related_skills: [github-pr-workflow, github-issues, hermes-git-workflow]
---

# Hermes PR Provenance

## Overview

Hermes often pushes through one shared GitHub actor, such as `hermesbot` or `hermes agent`, while the actual work was authored by a routed provider family such as Codex, Grok, Claude, or Gemini. This skill standardizes that distinction for any repository without importing Tailboard-specific DP-11 rules.

The universal rule is:

- GitHub actor/account tells who pushed or opened the PR.
- `Writer:` tells which Hermes provider family wrote the code or update.
- PR and Beads metadata mirror the same provider-level writer.
- Exact model/version tracking is intentionally deferred by default.

## When to Use

Use this skill for any Hermes-authored or Hermes-updated repository work:

- Creating commits.
- Pushing branches.
- Opening or updating GitHub PRs.
- Creating, commenting on, or closing GitHub issues.
- Claiming, updating, closing, or cross-linking Beads issues.
- Summarizing provenance in handoffs or PR descriptions.

Do not use this as a replacement for repository-specific ship/closeout rules. If a repository has stronger local rules, follow them first and add this provider-level provenance where compatible.

## Writer Vocabulary

Use provider/route-family values, not exact model versions.

Preferred universal values:

| Writer value | Use when |
|---|---|
| `codex` | OpenAI Codex / GPT-Codex route authored the change. |
| `grok` | xAI / Grok route authored the change. |
| `claude` | Claude wrote/reviewed through an approved headless Claude route. |
| `gemini` | Gemini route authored the change. |
| `openrouter` | OpenRouter-hosted route authored the change and no narrower repo vocabulary exists. |
| `human` | A human authored the change and Hermes is only recording or pushing it. |
| `user` | Use only if the target repo already prefers `user` over `human`. |

If the repository already defines a writer vocabulary, use the repository vocabulary. Do not introduce exact model strings into `Writer:`.

For GPT-5.5 through OpenAI Codex, record:

```text
Writer: codex
```

## Commit Trailer Contract

Every Hermes-authored commit should end with contiguous Git trailers.

Minimum shape when an issue or task exists:

```text
Writer: codex
Refs: #123
```

For Beads-only tasks, use the Beads id in `Refs:` when there is no GitHub issue:

```text
Writer: codex
Refs: tb-123
```

Rules:

- Keep `Writer:` provider-level.
- Do not add `Writer-Model:` by default.
- Do not put model versions in `Writer:`.
- Keep `Writer:`, `Refs:`, and `Co-Authored-By:` trailers contiguous at the end of the commit message.
- Do not insert blank lines between trailers.
- Use `Refs:` with the colon; omitting the colon can break trailer parsing.
- `Co-Authored-By:` is optional and should identify a stable agent/provider persona, not a rotating exact model version.

Example commit message:

```text
fix: preserve redirect URL after login

Preserves the ?next= parameter instead of always redirecting to /dashboard.

Writer: codex
Refs: #42
```

If multiple providers materially authored one commit and splitting commits is not practical, use repeated `Writer:` trailers:

```text
Writer: codex
Writer: grok
Refs: #42
```

Prefer one writer per commit when possible.

## PR Provenance Block

Hermes-created or Hermes-updated PRs should include a concise provenance section.

Template:

```markdown
## Provenance

- GitHub actor: <github-login-or-bot-account>
- PR created by: <writer-provider>
- Implemented by: <writer-provider-or-comma-list>
- Writers from commit trailers: <writer-provider-or-comma-list>
- Task ledger: <Beads tb-id(s), GitHub issue(s), or none>
```

Example:

```markdown
## Provenance

- GitHub actor: hermesbot-almace
- PR created by: codex
- Implemented by: codex
- Writers from commit trailers: codex
- Task ledger: Beads tb-123; GitHub #42
```

Do not include exact model names unless the user explicitly asks for model-level tracking in that repository.

## Beads Integration

Use Beads when a repository has a `.beads/` directory or otherwise documents Beads as the task ledger. Do not force Beads into repositories that do not use it.

Detection:

```bash
if command -v bd >/dev/null 2>&1 && [ -d .beads ]; then
  echo "Beads repo detected"
fi
```

Recommended provider-level metadata, where the repo/tooling accepts it:

```bash
bd update <id> --claim
bd update <id> --set-metadata implemented_by=codex
bd update <id> --set-metadata pr_created_by=codex --set-metadata github_pr="#123"
bd update <id> --status closed --set-metadata closed_by=codex
```

Guidelines:

- Keep Beads fields provider-level: `implemented_by=codex`, `closed_by=codex`, `pr_created_by=codex`.
- Do not add model-specific fields like `implemented_models` unless the repository explicitly asks for model-level provenance.
- Do not overwrite existing metadata from another provider or human; append/update only according to the repo's Beads conventions.
- If the repo has a local Beads skill or close protocol, follow that first.
- If `bd update --set-metadata` is unavailable, add a Beads comment/note or include the provenance in the PR body instead.
- Keep Git commit trailers as the durable source of writer provenance even when Beads metadata is also updated.

## GitHub Issues and Comments

For substantial issue comments or status updates written by Hermes, include provider attribution when useful:

```markdown
Hermes update: investigated and opened PR #123.

Provenance: Writer: codex
```

Avoid noisy provenance on tiny comments unless the repository expects it.

## Validator / Helper

Use the bundled helper before opening or updating a PR:

```bash
hermes-provenance-check --base origin/main --head HEAD
```

Common variants:

```bash
# Validate a specific commit range.
hermes-provenance-check --range main..HEAD

# Check a PR body markdown file too.
hermes-provenance-check --base origin/main --pr-body /tmp/pr-body.md

# Print a starter provenance block from detected commit writers and refs.
hermes-provenance-check --base origin/main --emit-pr-block

# Allow a repository-specific writer enum in addition to the defaults.
hermes-provenance-check --allowed-writer cursor --allowed-writer chat
```

The helper fails when Hermes-authored commits are missing `Writer:`, when `Refs:` is required but absent, when trailers are not in the final contiguous trailer block, or when exact-model provenance appears without an explicit opt-in.

## Verification Checklist

Before handing off or merging:

- [ ] Each Hermes-authored commit has at least one provider-level `Writer:` trailer.
- [ ] Relevant commits include `Refs:` for the GitHub issue, PR, Beads id, or task id.
- [ ] Trailers are contiguous with no blank lines between them.
- [ ] The PR body has a `## Provenance` block when Hermes created or materially updated the PR.
- [ ] Beads metadata is provider-level if Beads is present and supports metadata.
- [ ] `hermes-provenance-check` passes, or any intentional exception is documented.
- [ ] No exact model/version was recorded unless the user explicitly requested it.
- [ ] Repository-specific closeout rules still take precedence.

## Common Pitfalls

1. Confusing GitHub actor with writer provider. The bot account may push every commit, but `Writer:` captures which Hermes route authored it.
2. Encoding exact model names in `Writer:`. This breaks repos that parse `Writer:` as a small enum.
3. Adding `Writer-Model:` everywhere. Exact model provenance is deferred unless explicitly requested.
4. Forgetting `Refs:`. Many audit and task-ledger workflows rely on commit-to-task linkage.
5. Adding blank lines between trailers. Git trailer parsing expects a contiguous trailer block.
6. Applying Tailboard DP-11 to every repo. Tailboard can keep its own richer rules; this skill is the portable baseline.
