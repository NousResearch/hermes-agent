---
name: coder
description: "Use for write-capable engineering work: code changes, tests, refactors, repo maintenance, Hermes pilot implementation, local tooling, code review fixes, and verified PR-ready changes."
version: 1.0.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [coder, engineering, code, tests, repo-maintenance, hermes-pilot, review, yallaplay]
    related_skills: [codebase-readonly, knowledge, infrastructure]
---

# Coder Skill

## Overview

Use this skill when the task requires changing files or preparing implementation work. The goal is careful senior-engineer behavior: inspect first, make the smallest reversible change, verify with real commands, and avoid unrelated churn.

This is the write-capable counterpart to `codebase-readonly`. Use `codebase-readonly` first when the task is only investigation or source provenance.

## When to Use

- Code changes, scripts, tests, refactors, repo maintenance, or local tooling.
- Scheduler, bot, Hermes profile, dashboard, gateway, or pilot implementation work.
- Fixes discovered during debugging or code review.
- Test creation, lint/build repairs, dependency/config changes, or generated artifacts the user explicitly wants.
- Sibling repo edits only when the user explicitly asks and scope is clear.

Do not use this for pure source lookup, analytics provenance, or read-only explanation; use `codebase-readonly`. When analytics, liveops, collaboration, infrastructure, observability, or knowledge is the primary surface, load that domain skill too.

## Required Context

Before editing, inspect relevant durable context when it exists:

- `yallaplay-wiki/engineering/index.md`
- `yallaplay-wiki/engineering/data_warehouse.md` for warehouse pipeline/source mechanics.
- `yallaplay-wiki/reference/analytics_rules.md` when code changes affect telemetry or metrics.
- Feature/system pages under `yallaplay-wiki/products/**/engineering.md`, `design.md`, and `facts.md`.
- Existing neighboring files, tests, manifests, and project docs in the target repo.

For repo files, use `search_files` and `read_file` first. Trace symbol definitions and usages rather than guessing imports, shapes, or APIs.

## Write Workflow

1. **Classify the surface.** Name whether this is Hermes pilot code, wiki tooling, generated artifacts, profile state, or a sibling source checkout. Completion: target repo/path, branch/dirty state, and risk level are clear.
2. **Inspect before editing.** Read manifests, neighboring files, tests, and relevant wiki pages. Completion: the symbol/file/API shape is verified from source.
3. **Plan the smallest reversible change.** Avoid drive-by refactors, broad renames, and unrelated formatting. Completion: every intended touched file has a reason.
4. **Edit with repo style.** Prefer `patch` for focused edits; use `write_file` for new files or deliberate whole-file rewrites. Completion: diff is focused and every modified file is intentional.
5. **Verify.** Run the narrowest useful test, lint, compile, dry-run, or temporary ad-hoc script. Completion: real tool output backs the claim, or a concrete blocker is reported.
6. **Review the diff.** Check `git diff`/`git status` before finalizing. Completion: no secrets, generated churn, or unrelated changes are included.
7. **Honor repo-specific commit policy.** If the active repo’s project context asks for local commits, stage only intended files and create focused conventional commits after complete, verified units. Completion: commit SHA(s), verification command(s), and intentionally uncommitted files are known.
8. **Capture reusable knowledge.** If the task reveals a stable workflow, schema behavior, or codepath fact, add it to `yallaplay-wiki/` or a skill rather than memory. Completion: reusable fact is saved or intentionally skipped as one-off.

## Profile, Bot, and Scheduler State

- `bot/jobs_state.json` and scheduler runtime state are owned by the scheduler; do not edit directly.
- Local Hermes profile state (`~/.hermes/profiles/...`) affects the running agent; treat writes there as operational changes, not ordinary repo edits.
- Project-owned YallaPlay skills live under `skills/yallaplay/` and should be symlinked into `~/.hermes/profiles/<profile>/skills/claudio-authored/` so repo and TUI stay in sync while showing a clear authored-skills category.
- Ask before visible or production writes. Profile skill symlinks are acceptable when the user asks for a skill to be usable in the active profile.

## Sibling Source Edits

Sibling backend/Unity checkouts are higher-risk than this pilot repo.

Rules:

- Check dirty state and current branch before editing.
- Prefer isolated worktrees for concurrent sibling work.
- Do not commit, push, or rewrite history unless the user explicitly asks.
- Never print secrets or token-bearing config while inspecting.
- Keep fixes minimal and verify with the target repo’s own tests/build commands where available.

## Legacy Capability Migration Audits

When comparing the legacy Claudio repo (`yallaplay-analytics-agent-gpt`) with the Hermes pilot, audit by capability class rather than listing every script. Start from the legacy `CLAUDE.md`, `COMMANDS.md`, `docs/README.md`, and `jobs/README.md`, then compare against the pilot `README.md`, `docs/capability-map.md`, `docs/pilot-plan.md`, `tools/README.md`, and `tools/*.py`. Classify each area as ported, native-Hermes-covered, legacy-fallback-only, missing/not yet ported, or intentionally out of scope.

See `references/legacy-capability-audit.md` for the compact inventory and reporting pattern from the first migration audit.

## Safety

- Refuse broad deletion of repos, Hermes homes, wiki content, tool directories, credentials, or generated operational state.
- Do not read or print secrets (`vars.toml`, `.env`, tokens, OAuth caches) unless explicitly required and scoped.
- Do not commit, branch, push, or rewrite history unless asked or the active repo's project context explicitly requires local commits.
- Avoid production writes unless a documented gated workflow exists and the user confirms the exact action.
- Before destructive commands, narrow the path and prefer reversible file moves over deletion.

## Common Pitfalls

1. **Skipping read-only diagnosis.** Understand the current behavior before editing. Load/use `codebase-readonly` if the fix is not yet clear.
2. **Testing by description.** A summary is not verification; run a command or report the blocker.
3. **Broken skill/profile link.** Editing `skills/yallaplay/<name>/SKILL.md` should affect `/skills` via the `claudio-authored` profile symlink; if not, check the symlink and run `/reload-skills` or start a new session.
4. **Accidental generated churn.** Keep generated logs/outputs/caches out of commits unless the user asked for the artifact.
5. **Overloading memory.** Task outcomes and PR numbers are not memory; durable procedures belong in skills or wiki pages.

## Verification Checklist

- [ ] Relevant source/wiki context was inspected.
- [ ] Dirty state was checked before risky edits or sibling repo changes.
- [ ] Every modified file is necessary for the requested change.
- [ ] No secrets or unrelated profile/runtime state were committed or exposed.
- [ ] A focused test/lint/compile/dry-run/ad-hoc verification was run, or the blocker is explicit.
- [ ] `git status`/`git diff` was reviewed before final response.
- [ ] The active repo's commit policy was followed, including focused local commits when project context requires them.
- [ ] Reusable engineering knowledge was captured in `yallaplay-wiki/` or a skill when applicable.
