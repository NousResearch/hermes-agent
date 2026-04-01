---
name: hermes-repo-map
description: Build and use a compact architectural map of the hermes-agent codebase before planning fixes, reviews, or cross-cutting changes. Prefer this when Hermes is improving Hermes itself.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hermes-agent, architecture, repo-map, codebase-synopsis, planning]
    related_skills: [codebase-inspection, writing-plans, systematic-debugging, github-pr-workflow]
---

# Hermes Repo Map

Use this skill when working on `NousResearch/hermes-agent` itself and the task would benefit from a compact architectural map before implementation.

## Why this skill exists

Hermes often has to rediscover the same repository structure through repeated `search_files` and `read_file` calls. That works, but it is slow and token-hungry for Hermes self-development work.

This skill provides a repeatable way to rebuild a compact repo map on demand without making it part of the default system prompt for every session.

## When to use

Use this skill when:
- fixing or reviewing Hermes bugs that span multiple subsystems
- writing an issue, PR plan, or architecture note for Hermes
- evaluating whether a proposed change belongs in `run_agent.py`, `model_tools.py`, `toolsets.py`, `hermes_cli/*`, or `tools/*`
- comparing Hermes against other agent/codebase-map implementations
- refreshing yourself on the Hermes architecture after starting in a fresh session or worktree

Usually load this before or alongside `hermes-agent-dev` for architecture-sensitive work.

## Core rule

Treat the repo map as task-local working context, not a live mutable system prompt feature.

Do NOT propose designs that:
- rebuild the system prompt every turn
- mutate prompt/tool availability mid-session
- make cache-unstable context injection the default behavior

Hermes relies on a stable per-session prompt prefix for cost and latency. This skill is for explicit, opt-in repo understanding during Hermes self-work.

## What to build

Create a compact map with four layers:

1. Entry points
- where sessions start
- where CLI/gateway dispatch happens
- where tools are discovered and executed

2. Core subsystems
- agent loop
- prompt building/caching/compression
- CLI/config
- gateway/session persistence
- tools/registry/toolsets

3. Key invariants
- system prompt is built once per session and kept stable except compression rebuilds
- tool availability is resolved before the conversation, not mutated mid-session
- profile-aware paths must use `get_hermes_home()` / `display_hermes_home()`
- CLI/gateway config loading differs and must be checked separately

4. Likely touch points for the current task
- exact files most likely to change
- adjacent tests to read/run
- files that look relevant but should probably NOT be changed

## Recommended workflow

### 1. Confirm you are actually in Hermes

Look for:
- `run_agent.py`
- `model_tools.py`
- `toolsets.py`
- `hermes_cli/`
- `tools/`
- `tests/`

If those are absent, do not pretend this is the Hermes repo.

### 2. Read the fixed anchor files first

Always start with these anchor files unless the task is extremely narrow:
- `AGENTS.md`
- `run_agent.py`
- `model_tools.py`
- `toolsets.py`
- `agent/prompt_builder.py`
- `hermes_cli/config.py`
- `tools/registry.py`

Then read task-specific files.

### 3. Produce a compact repo map in your own working notes/response

Keep it short and operational. Good format:
- `cli.py` / `hermes_cli/main.py` — interactive and subcommand entrypoints
- `run_agent.py` — core loop, system prompt assembly, session state
- `model_tools.py` — tool discovery, schemas, dispatch
- `toolsets.py` — tool grouping and platform composition
- `agent/prompt_builder.py` — skills/context prompt assembly and prompt-side snapshots
- `hermes_cli/config.py` — defaults, migrations, config metadata
- `tools/registry.py` — central tool registration
- `gateway/run.py` + `gateway/session.py` — messaging platforms and session persistence
- `tests/...` — target subsystem tests

### 4. Add task-local hotspots

After the anchor map, identify:
- exact files likely to change
- exact tests likely to fail or need updates
- any cross-cutting constraints

### 5. Only then propose code changes

Do not jump straight from user request to patching files when the change is architectural or cross-cutting.

## Competitor research guidance

When the task is specifically about repo maps, codebase synopses, or architecture summaries, compare Hermes against at least 2 current implementations.

### Recommended comparisons
- `Aider-AI/aider`
- `RooCodeInc/Roo-Code`
- optionally another agent/tooling repo if directly relevant

### What to verify

Do not rely on reputation. Inspect the current tree and identify the actual mechanism.

Check:
- where the map/index implementation lives
- whether it is syntax/symbol based, graph ranked, embeddings based, or just memoized docs
- whether it is always-on or on-demand
- how it handles caching, invalidation, and refresh
- whether it tracks filesystem changes/watchers
- how it enforces token/context budgets
- what gets injected into prompts versus queried as a tool/service
- whether the output is file-level, symbol-level, scope-aware, or search-result oriented

### Competitor scorecard

For each competitor, explicitly score these dimensions in your notes:
- extraction: AST/tree-sitter tags, regex/grep, embeddings, or hybrid
- ranking: PageRank/graph, search score, heuristics, or none
- caching: in-memory, on-disk, invalidation by mtime/content hash, or none
- update model: rebuild per turn, background watcher, on-demand manual refresh, or startup precompute
- output shape: directory map, symbol list, scope-aware code elision, semantic search hits, or architecture prose
- prompt strategy: automatic injection, tool-returned context, or external service query
- Hermes fit: copy now, copy later, or avoid

### Specific patterns to harvest

From Aider, look for:
- compact structural summaries rather than full file dumps
- symbol-aware extraction
- explicit token-budgeting
- relevance ranking tied to active files / current task
- scope-aware elision instead of dumping large code blocks

From Roo Code, look for:
- separation of config/state/orchestration
- disciplined indexing lifecycle and invalidation
- clear separation between indexing concerns and prompt/context concerns
- optional heavier infrastructure that can remain outside the default prompt path

### Hermes-specific interpretation

Prefer conclusions like:
- Aider-style compact structural summaries fit Hermes well
- Aider-style always-rebuilt per-turn prompt injection conflicts with Hermes cache goals unless carefully constrained
- Roo-style indexing architecture is useful for future deeper systems, but too heavy for a first Hermes repo-map implementation
- Hermes should favor on-demand skills or tools over default mutable prompt context
- Hermes should explicitly preserve stable per-session prompt identity while still improving repo understanding

## Suggested output template

Use this shape in your own reasoning or user-facing summary:

```text
Hermes repo map
- Entry points: ...
- Prompt/cache invariants: ...
- Tooling/config invariants: ...
- Task hotspots: ...
- Relevant tests: ...
- Competitor scorecard:
  - Aider: extraction / ranking / caching / update model / output / prompt strategy / Hermes fit
  - Roo Code: extraction / ranking / caching / update model / output / prompt strategy / Hermes fit
- Recommended Hermes-specific approach: ...
```

## Hermes anchor map

Start from this baseline and refine per task:
- `run_agent.py` — main synchronous agent loop, cached system prompt, session state
- `model_tools.py` — tool discovery/imports, schema collection, dispatch plumbing
- `toolsets.py` — core/shared toolsets and platform composition
- `agent/prompt_builder.py` — skills system prompt, context-file prompt, snapshot caching
- `agent/prompt_caching.py` — Anthropic cache-control breakpoint strategy
- `agent/context_compressor.py` — context pressure handling and rebuild path
- `hermes_cli/main.py` — top-level CLI command surface
- `hermes_cli/config.py` — persistent config defaults/migration/env metadata
- `cli.py` — interactive CLI orchestration and slash-command handling
- `tools/registry.py` — tool registry contract
- `tools/*.py` — tool implementations
- `gateway/run.py` — gateway message loop and command dispatch
- `gateway/session.py` — gateway conversation persistence
- `hermes_state.py` — SQLite session metadata and search backing store
- `tests/` — subsystem regression coverage

## Pitfalls

- Do not confuse `cli.py` with `hermes_cli/main.py`; both matter, but for different entry surfaces.
- Do not add default repo maps to every session's system prompt without proving the cache/cost tradeoff works.
- Do not assume CLI and gateway use the same config-loading path.
- Do not hardcode `~/.hermes`; use profile-aware helpers.
- Do not treat a competitor's vector index or live mutable session state as Hermes-safe by default.
- Do not skip reading adjacent tests before architectural edits.

## Good outcomes

A good use of this skill should leave you able to answer:
- Which Hermes subsystem owns this behavior?
- Which file is the right place for the change?
- Which files are adjacent but should stay untouched?
- Which tests prove the change?
- Which competitor ideas fit Hermes, and which do not?
