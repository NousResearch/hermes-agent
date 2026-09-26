# Agent core

The root `AGENTS.md` and `CODING_STANDARDS.md` apply. Read `website/docs/developer-guide/agent-loop.md`, `website/docs/developer-guide/prompt-assembly.md`, `website/docs/developer-guide/context-compression-and-caching.md`, `website/docs/developer-guide/provider-runtime.md`, `website/docs/developer-guide/session-storage.md`, and `website/docs/developer-guide/subagent-lifecycle-api.md` for maintained architecture.

## Turn and message invariants

`run_agent.py` is the `AIAgent` facade. Construction lives in `agent/agent_init.py`. `agent/conversation_loop.py` coordinates topical `turn_*.py` phases through `agent/turn_facade.py` and the turn lease.

- Keep the system prompt byte-stable for a conversation. Compression is the sanctioned cache break. Skills and other mid-session additions use user messages or tool results.
- Preserve provider-valid role order. `/steer` may add a standalone user row after an assistant tool call and its tool result. It never mutates the persisted tool row.
- Agent-level tools are dispatched through `INLINE_TOOL_EXECUTORS` in `agent/inline_tool_executors.py`, before `handle_function_call()`. Add an entry instead of a name branch.
- `_last_resolved_tool_names` in `model_tools.py` is process-global. Delegation saves and restores it, so readers must tolerate a temporarily stale value while a child runs.

## Project context

`agent/prompt_builder.py` loads the first non-empty project-context type in this order:

1. nearest `.hermes.md` or `HERMES.md`, walking from the working directory to the Git root.
2. the `AGENTS.md` directory chain from Git root to working directory.
3. working-directory `CLAUDE.md` or `claude.md`.
4. working-directory `.cursorrules` plus `.cursor/rules/*.mdc`.

For every directory in an `AGENTS.md` chain, the first non-empty readable `AGENTS.override.md`, `AGENTS.md`, or `agents.md` wins. Sections merge root first and working directory last, identical content is deduplicated, and no-Git workspaces stay working-directory-only. Deeper rules therefore have later precedence. A fallback working directory inside the Hermes install tree has no project-context authority unless the caller explicitly selected it or enabled the CLI-style fallback.

The cap comes from positive `context_file_max_chars` config when present. Otherwise `_dynamic_context_file_max_chars` uses 6% of the model window with a 20,000-character floor, 500,000-character ceiling, and 20,000-character unknown-window fallback. Each file section and the merged `AGENTS.md` chain receive the cap independently. `_truncate_content` keeps head and tail, inserts a marker with the `read_file` path, logs, and queues a context-local warning.

`agent/subdirectory_hints.py` handles directories entered after startup. It attaches local context to a tool result rather than rebuilding the prompt, stays inside the working tree, honors overrides and denied paths, deduplicates content, and uses its own fixed preview cap. Keep the user-facing contract synchronized with `skills/autonomous-ai-agents/hermes-agent/references/project-context-files.md` and `website/docs/developer-guide/prompt-assembly.md`.

## Compression and auxiliary work

Manual compression on every client goes through `agent/conversation_compression_manual.py::compress_now`. Automatic compression prunes tool results, chooses boundaries, and summarizes through the auxiliary route. Provider-native compaction remains provider-specific. A failed summary must not commit an unfenced prune. See `website/docs/developer-guide/context-compression-and-caching.md` for the full state machine.

Curator, vision, embedding, title generation, session search, and compression resolve through `agent/auxiliary_client.py`. Physical attempts emit `pre_auxiliary_call` and `post_auxiliary_call`. They do not emit main-turn `pre_api_request` or `post_api_request` events.

## Profile and lifecycle scope

Memory, context-engine, image-generation, and curator work may run outside a turn. The caller binds the owning profile before session end, eviction, shutdown, ticker work, and background callbacks. Start scoped background work with `agent.memory_provider.spawn_context_thread`.

`agent.secret_scope.get_secret` fails closed after multiplexing activates. A child `UnscopedSecretError` identifies a missing spawn or scope binding. It is not grounds for an `os.getenv` fallback. Delegated child markers carry the fenced Kanban board root, not a boolean.

## Tests

Agent tests live in `tests/agent/` and run through `scripts/run_tests.sh`. Assert message order, stable prompt bytes, context precedence, and real resolution behavior. Patch the binding the phase reads. Do not snapshot prompt prose.