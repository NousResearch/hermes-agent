---
name: codebase-readonly
description: "Use for read-only codebase investigation: repo exploration, debugging, source lookup, C# sibling source tracing, event/property provenance, and implementation behavior questions without making code changes."
version: 1.0.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [codebase, read-only, repo-search, debugging, csharp, analytics-provenance, yallaplay]
    related_skills: [knowledge, analytics]
---

# Codebase Read-Only Skill

## Overview

Use this skill when the task is to understand code, not change it. The goal is source-grounded investigation: trace definitions and usages, identify where events/properties are emitted or filled, explain runtime behavior, and leave the working tree untouched.

This is the low-risk engineering path for analytics, liveops, observability, and product questions that need source as evidence but do not require implementation work.

## When to Use

- Repo search, source reading, symbol tracing, or implementation behavior questions.
- Debugging and root-cause exploration before deciding whether a code change is needed.
- Analytics provenance checks, e.g. where an event fires, how a property is populated, or which enum/config value is sent.
- Sibling backend/Unity source lookup for event emission, formulas, payout logic, item names, config keys, or enums.
- Code review-style explanation when the user asks what the code currently does.

Do not use this for edits, refactors, test creation, commits, PRs, or generated code changes; use `coder` for write-capable implementation work. When analytics, liveops, collaboration, infrastructure, or observability is the primary domain, load that domain skill too.

## Read-Only Contract

- Do not call `patch`, `write_file`, `skill_manage`, `git commit`, `git checkout`, `git reset`, formatters, migrations, package updates, or any command intended to mutate source or runtime state.
- Prefer `search_files` and `read_file` for source inspection.
- Use `terminal` only for read-only discovery commands such as `git status`, `git branch`, `git log`, `python --version`, `hermes skills list`, or scripts with documented dry-run/read-only behavior.
- Avoid tests/builds unless the user asks; they can create caches or generated files. If you must run one for diagnosis, check `git status --short` before and after and report any generated files.
- Do not edit sibling checkouts. If investigation shows a fix is needed, stop and recommend switching to `coder`.

## Required Context

Before source tracing, inspect relevant durable context when it exists:

- `yallaplay-wiki/engineering/index.md`
- `yallaplay-wiki/engineering/data_warehouse.md` for warehouse pipeline/source mechanics.
- `yallaplay-wiki/reference/warehouse/index.md` for warehouse table reference.
- `yallaplay-wiki/reference/analytics_rules.md` when telemetry or metrics are involved.
- Feature/system pages under `yallaplay-wiki/products/**/engineering.md`, `design.md`, `analytics.md`, and `facts.md`.

Use wiki context to aim the source search, but treat source as the tiebreaker for actual runtime behavior.

## Investigation Workflow

1. **Classify the question.** Name the target surface: Hermes pilot, wiki tooling, backend, Unity client, data pipeline, config, or sibling source lookup. Completion: repo/path and desired evidence are clear.
2. **Find anchors.** Search for event names, property names, symbols, config keys, enum values, API routes, or file names. Completion: candidate files and symbol names are identified from source, not guessed.
3. **Trace definitions and usages.** Read the emitter/filler and its nearby callers/calculations. Completion: the data path is explainable from source lines.
4. **Check context and edges.** Look for feature flags, app/platform branches, environment guards, null/default handling, version gates, and fallback paths. Completion: material conditions and caveats are known.
5. **Report with evidence.** Cite files and concise line references, state confidence, and call out unresolved gaps. Completion: the user can verify the answer from source.
6. **Escalate when needed.** If a fix, migration, or test is required, summarize the proposed change and switch to `coder` only after the user’s task requires write access.

## Analytics Provenance Pattern

For “where does this event/property come from?” questions:

1. Search exact event/property strings first.
2. If not found, search enum/constants and serialization layers that transform names.
3. Trace from event construction to send/enqueue layer, then backward to the state source.
4. Verify app/game/platform gates and default values.
5. Map source property names to warehouse/event column names only after checking analytics wiki/schema notes.

## Sibling C# Source Lookup

Use source as the canonical tiebreaker for:

- Event emission and analytics parameter names.
- Enum values, item names, formula edges, payout logic, and config keys.
- Runtime behavior not fully captured in wiki pages.

Rules:

- Prefer wiki/config/schema knowledge for behavioral questions first; use C# source to verify or settle ambiguity.
- Do not edit sibling checkouts without switching to `coder` and checking dirty state and branch/base first.
- For concurrent sibling work, prefer isolated worktrees when available rather than sharing one checkout.
- Never commit or push sibling repo changes unless the user explicitly asks.

## Common Pitfalls

1. **Accidental writes during investigation.** Tests, builds, formatters, package managers, and codegen may mutate files. Avoid them unless requested and verify status before/after.
2. **String-only conclusions.** Finding a string is not enough; trace construction, call sites, and guards.
3. **Guessing schema mappings.** Event/source names may be transformed before landing in Snowflake. Check wiki/schema notes and source serialization.
4. **Stopping at the first match.** Analytics events and config keys often have client, backend, and warehouse layers. Trace the relevant layer for the question.
5. **Overloading memory.** Task outcomes and transient findings do not belong in memory; durable reusable source facts belong in `yallaplay-wiki/` when they will help future work.

## Verification Checklist

- [ ] Relevant wiki/source context was inspected.
- [ ] No write tools or mutating commands were used.
- [ ] If any command could generate files, `git status --short` was checked before and after.
- [ ] Answer cites source files/line references or explicitly states why source evidence is incomplete.
- [ ] Proposed fixes are clearly separated from read-only findings.
