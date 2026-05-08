# GitHub as Source of Truth

Date: 2026-05-08
Status: Accepted

## Context

Hermes Agent is a serious project with code, docs, skills, plugins, CI, and release-sensitive behavior. Durable work needs a source-of-truth workflow that preserves planning, review, test evidence, and handoffs instead of relying on local-only edits.

## Decision

Use GitHub as the source of truth for meaningful Hermes Agent work:

- Track non-trivial changes in GitHub issues.
- Write implementation plans in `docs/plans/` for multi-step, risky, or delegated work.
- Capture durable architectural or workflow decisions in `docs/decisions/`.
- Use focused branches from the current default branch.
- Open PRs for review and CI before merge.
- Do not merge without Stephen's explicit approval unless a repo/workflow has been pre-authorized.
- Keep secrets out of git; update examples with placeholders only.

## Consequences

- Local scratch work remains fine for exploration, but serious changes need branch/commit/PR history.
- Agents should inspect repo state before editing, avoid unrelated dirty files, and include verification commands in PRs.
- CI failures should be fixed on the same PR branch with follow-up commits.
