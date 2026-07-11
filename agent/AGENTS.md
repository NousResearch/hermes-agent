# Agent Core Subtree Instructions

This file scopes core-agent guidance to `agent/` work. Root `AGENTS.md` still contains the non-negotiable project rules.

## Core invariants

- Prompt caching is sacred: do not change the system prompt, toolset, memories, or past context mid-conversation outside the established compression path.
- Preserve strict message role alternation.
- Context compression is the only normal mechanism that rewrites conversation context.
- Environment hints must describe the executable backend the tools actually touch. For remote terminal backends, do not leak host-only filesystem assumptions into prompts.
- Context-file caps are resolved once per conversation and must remain stable for cache safety.

## State and paths

- Use `get_hermes_home()` for Hermes state and `display_hermes_home()` for user-facing paths.
- Profile behavior depends on `HERMES_HOME` being applied before imports; avoid hardcoded home paths and import-time side effects that bypass profile scoping.

## Verification

- For provider/model routing, compression, memory, or prompt-builder changes, prefer E2E-style tests against a temp `HERMES_HOME` over mocks.
- Run tests with `scripts/run_tests.sh ...`, never direct `pytest`.
