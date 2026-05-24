# Agent Internals Guide

This directory owns provider adapters, prompt assembly, memory orchestration,
compression, LSP, context engines, delegation support code, and runtime helpers.

## Prompt Assembly

`agent/system_prompt.py` and `agent/prompt_builder.py` assemble prompt parts into
stable, context, and volatile sections. Preserve prompt caching:

- stable: identity, tool-use guidance, provider-specific guidance.
- context: caller system message and project context files.
- volatile: memory snapshots, external memory prefetch, plugin context.

Context file loading is deliberately bounded. Startup loading chooses the first
project context source from `.hermes.md`/`HERMES.md`, `AGENTS.md`, `CLAUDE.md`,
or `.cursorrules`; `SOUL.md` from `HERMES_HOME` is independent. When `AGENTS.md`
is the selected source, Hermes loads the git-root-to-current-working-directory
hierarchy, with more specific files later in the prompt. Deeper subdirectory
hints in `agent/subdirectory_hints.py` are discovered lazily from tool arguments
and appended to tool results instead of rewriting the system prompt.

Context files are security-scanned before injection. Keep new context loaders
bounded, scanned, and cache-aware.

## AIAgent Lifecycle

`run_agent.py` delegates most initialization to `agent/agent_init.py` and the
conversation loop to `agent/conversation_loop.py`.

Memory, skills, context engines, plugin hooks, and compression are optional
subsystems. Failures should degrade gracefully unless the caller explicitly
requested the feature.

The agent loop is synchronous at the top level. Tool execution can fan out
internally, but parent-visible ordering and budget accounting must remain
coherent.

## Memory

Hermes has two memory layers:

- built-in curated files via `tools/memory_tool.py`: `MEMORY.md` and `USER.md`
  under `get_hermes_home() / "memories"`.
- external memory providers via `agent/memory_manager.py` and
  `agent/memory_provider.py`.

Built-in memory is loaded as a frozen snapshot at session start. Mid-session
writes are durable on disk but do not rewrite the active system prompt.

Only one external memory provider is active at a time. Providers can prefetch
context, sync turns, react to explicit memory writes, and run session-end
extraction. Provider failures should not break the primary agent loop.

Cron sessions intentionally run with `skip_memory=True` to avoid corrupting user
representations from scheduled-job prompts.

## Context Engines

`agent/context_engine.py` defines the pluggable context-engine lifecycle.
Compression and other engines can receive `on_session_start` and
`on_session_end`. Keep engine failures non-fatal unless the active operation
cannot proceed safely.

## LSP

The LSP layer provides semantic diagnostics after file writes and patches.
Management lives under `agent/lsp/`; integration into file writes lives in
`tools/file_operations.py`.

Current LSP support is diagnostic-oriented. Do not assume symbol navigation
tools exist unless you add and expose them explicitly.

Config lives under `lsp:` in `DEFAULT_CONFIG`:

- `enabled`
- `wait_mode`
- `wait_timeout`
- `install_strategy`
- `servers`

LSP is gated on git workspace detection and degrades silently when unavailable.

## Delegation Support

`delegate_task` lives in `tools/delegate_tool.py`, but active child-agent state,
interrupt propagation, budgets, provider routing, and callbacks interact with
agent internals.

Subagents get isolated conversation context and their own iteration budget.
They are not durable. For work that must outlive the current turn, use cron,
kanban, or a background terminal process with notifications.

Subagent summaries are self-reports. Parent agents should ask for verifiable
handles such as paths, commands, IDs, or URLs when correctness matters.

## Tool-use Guidance

`agent/prompt_builder.py` contains model-family execution guidance for tool
persistence, mandatory tool use, and completion discipline. When adding new
guidance, keep it:

- short enough to preserve prompt budget,
- model-family gated when possible,
- behavior-focused rather than a snapshot of current model names.

## Curator

`agent/curator.py` reviews and maintains agent-created skills. It only touches
skills with `created_by: "agent"` provenance. Bundled and hub-installed skills
are out of scope.

Curator never deletes; it archives and backs up. Pinned skills are exempt from
automatic state transitions and LLM review.
