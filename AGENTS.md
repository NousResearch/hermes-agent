# Hermes Agent - Development Guide

Hermes is a personal AI agent from Nous Research that runs the same core across the CLI, TUI, desktop app, messaging gateway, scheduled jobs, plugins, skills, memory, terminal, and browser.

This file is the repo's compact bootloader for coding agents. Keep it under the default `context_file_max_chars` cap. Do not turn it back into a full contribution guide: route details to the canonical docs below.

**Never give up on the right solution.**

## Canonical docs

- Contributor workflow, setup, project layout, tool/skill authoring, cross-platform rules, security, dependency pinning, and PR process: `CONTRIBUTING.md`.
- Public developer docs: `website/docs/developer-guide/contributing.md`.
- Prompt/context semantics: `website/docs/developer-guide/prompt-assembly.md` and `website/docs/user-guide/features/context-files.md`.
- Agent loop, role alternation, tool execution, compression, and persistence: `website/docs/developer-guide/agent-loop.md`.
- Prompt caching and context compression: `website/docs/developer-guide/context-compression-and-caching.md`.
- Skills: `website/docs/developer-guide/creating-skills.md` plus bundled `SKILL.md` files.

If this file conflicts with those docs, prefer the more specific doc and update this bootloader only with stable, high-consequence rules.

## Non-negotiable architecture primitives

1. **Prompt caching is sacred.** A conversation expects a stable cached prefix. Do not mutate past context, swap toolsets, or rebuild the system prompt mid-conversation except through the existing context compression path.
2. **The core is a narrow waist.** New core model tools are expensive because their schemas ride on every model call. Prefer, in order: extend existing code, CLI command plus skill, service-gated tool, plugin, MCP/catalog entry, then new core tool as a last resort.
3. **Role alternation must hold.** Never create two adjacent messages with the same role, and never inject synthetic user messages mid-loop. Preserve provider-specific message invariants.
4. **Profiles are isolated.** Use `hermes_constants.get_hermes_home()` / profile-aware helpers for all Hermes state. Tests must never write to the real `~/.hermes/`.
5. **Config is for behaviour; `.env` is for secrets.** User-facing behavioural settings belong in `config.yaml`, not new `HERMES_*` env vars. API keys, tokens, passwords, and connection secrets belong in `.env` or credential stores.
6. **Edges can grow; the waist stays small.** Platforms, providers, desktop/TUI/dashboard features, skills, and plugins can expand aggressively when they use the existing setup/config UX and do not bloat the core model surface.

## Context-file semantics

Hermes context loading is not Codex's nested `AGENTS.md` merge chain.

- Startup project context priority is first-match: `.hermes.md` / `HERMES.md` from cwd up to git root, then `AGENTS.md` from cwd only, then `CLAUDE.md` from cwd only, then Cursor rules.
- `SOUL.md` is identity/tone only and is loaded separately from `HERMES_HOME`.
- Subdirectory `AGENTS.md`, `CLAUDE.md`, and `.cursorrules` files may be discovered later as tool-result hints when the agent touches paths under those directories. They are not preloaded into the cached system prompt.
- Do not add `.hermes.md` unless the repo needs Hermes-only instructions that should deliberately outrank this portable file.

## Contribution decision rules

Use the full rubric in `CONTRIBUTING.md`. The short form:

- Fix real reported behaviour, reproduce first, and cover the whole bug class including sibling call paths.
- Preserve contributor authorship when salvaging external work.
- Prefer behaviour contracts and invariants over snapshot/change-detector tests.
- Use E2E-shaped validation for config propagation, security boundaries, remote backends, file/network I/O, and resolution chains.
- Reject speculative hooks, unused managers, dead code wired without E2E proof, core tools that duplicate terminal/file/skills, and integrations for third-party products that should live as standalone plugins.

## Development workflow

- Work from a clean branch or an isolated worktree. Do not trample unrelated local changes.
- Search before adding a new abstraction; extend existing infrastructure when possible.
- Keep changes narrow. Large refactors are acceptable only when the declared request is the extraction and the diff stays mechanical.
- For Python, use the existing project environment or create a venv with Python 3.11. Do not install packages into the system Python.
- For UI/TypeScript work, follow the existing package manager and scripts in the touched app/package. Do not introduce a new JS package manager.

## Verification gates

Before calling work done:

1. Reproduce the bug or inspect the current behaviour when the task is a fix.
2. Run the smallest meaningful tests for the touched area.
3. For code changes, run the repository test wrapper before PR unless a documented blocker prevents it:
   ```bash
   scripts/run_tests.sh
   ```
4. For docs-only changes, at minimum run:
   ```bash
   git diff --check
   ```
5. Report exact commands and outcomes. Do not fabricate output when a tool, install, or network call fails.

## Persistent pitfalls

- Do not hardcode `~/.hermes`; use profile-aware helpers.
- Do not add new `simple_term_menu` usage.
- Do not use `\033[K` erase-to-EOL in spinner/display code.
- Treat `_last_resolved_tool_names` in `model_tools.py` as process-global state.
- Do not hardcode cross-tool references in schema descriptions.
- Gateway approval/control commands must bypass both gateway message guards.
- Squash-merging stale branches can silently revert recent fixes; inspect ancestry before merging.
- Do not add pagination/offset escape hatches to instructional loaders that agents must read fully, such as skills and playbooks.

## If you need more detail

Load the relevant skill or read the canonical docs. This file should answer: what is Hermes, what must not break, where to look next, and what evidence proves the change works. It should not teach the entire codebase.
