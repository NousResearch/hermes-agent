# Hermes performance PR candidates - 2026-05-15

Context:

- Workspace: `C:\Users\wesley.simplicio\Pictures\m\hermes-agent`
- Goal: identify 10 performance improvements that can become focused PRs.
- Codex CLI: `codex-cli 0.130.0` was available and used as a second-pass reviewer via `codex exec`.
- Hermes has `/goal` loop support in `cli.py`, `gateway/run.py`, and `hermes_cli/goals.py`. Codex CLI itself does not expose Hermes `/goal`; `codex exec` was used non-interactively.

Local measurements:

- First cold-ish `import model_tools`: 13.461s; repeat after OS/import cache: 2.357s.
- Built-in tool import timing: 29 modules, 3.050s total; biggest local cost was `tools.browser_tool` at 2.681s.
- `discover_plugins()`: 0.829s in this checkout, with platform/backend plugin imports visible in import-time output.
- `get_tool_definitions(quiet_mode=True)`: cold 0.3097s, warm 0.000486s.
- `SessionDB()` init: 0.5522s; `list_sessions_rich(20)`: 0.0005s; `search_sessions("test")`: 0.0003s in this local state.

External references consulted:

- OpenAI latency optimization: https://developers.openai.com/api/docs/guides/latency-optimization
- OpenAI prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Codex CLI docs: https://developers.openai.com/codex/cli
- Claude Code best practices: https://code.claude.com/docs/en/best-practices
- Claude prompt caching lessons: https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything
- Anthropic prompt caching docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- SQLite WAL docs: https://www.sqlite.org/wal.html
- Python profiling docs: https://docs.python.org/3.15/library/profiling.html
- Python sqlite3 docs: https://docs.python.org/3/library/sqlite3.html
- React useMemo/memo docs: https://react.dev/reference/react/useMemo and https://react.dev/reference/react/memo

## 10 PR candidates

1. Lazy-load built-in tools by enabled toolset.
   - Files: `model_tools.py`, `tools/registry.py`, `toolsets.py`.
   - Why: importing all self-registering tool modules at `model_tools` import time dominates startup. The local timing showed `tools.browser_tool` alone at 2.681s.
   - PR shape: generate or cache a manifest of tool name -> module/toolset/schema metadata, then import handlers only when the toolset is enabled or the tool is dispatched.

2. Move bundled platform plugin loading out of `model_tools` import.
   - Files: `model_tools.py`, `hermes_cli/plugins.py`, `gateway/platform_registry.py`.
   - Why: `model_tools` imports general plugin discovery, and bundled platform plugins load gateway/aiohttp stacks even for plain CLI tool schema setup.
   - PR shape: keep manifest introspection, but load `kind: platform` only from gateway startup or when platform registry asks for it.

3. Cache plugin manifest and entry-point scans with fingerprints.
   - Files: `hermes_cli/plugins.py`.
   - Why: `discover_plugins()` took 0.829s locally. Most runs do not change plugin manifests or pip entry points.
   - PR shape: cache parsed manifests keyed by plugin directories, manifest mtimes/sizes, `plugins.enabled`, and package entry-point metadata.

4. Cache `resolve_toolset()` / `validate_toolset()` by registry generation.
   - Files: `toolsets.py`, `model_tools.py`.
   - Why: `get_tool_definitions` already has a strong outer cache, but cold schema assembly still pays repeated registry/toolset resolution.
   - PR shape: memoize resolution results until `registry._generation` changes.

5. Preserve prompt-cache prefix by moving volatile prompt data into messages.
   - Files: `run_agent.py`, `agent/prompt_builder.py`.
   - Why: OpenAI and Anthropic both emphasize stable prompt prefixes; Claude Code notes timestamps/tool-order changes can break caching. Hermes already caches the system prompt per session, but fresh sessions may still embed volatile data.
   - PR shape: audit `_build_system_prompt()` for session id, timestamps, dynamic tool/platform status, and move truly volatile reminders into the latest user message.

6. Extend and cheapen the existing skill index snapshot.
   - Files: `agent/prompt_builder.py`, `agent/skill_utils.py`, maybe `utils.py`.
   - Why: Hermes already has an in-process LRU plus `.skills_prompt_snapshot.json` for the local skills dir, but external skill dirs are still scanned directly and snapshot validation rebuilds a full mtime/size manifest.
   - PR shape: include external dirs in the snapshot manifest and cache directory fingerprints so validation can skip expensive walks when root mtimes/sizes are unchanged.

7. Batch message persistence for completed turns.
   - Files: `run_agent.py`, `hermes_state.py`.
   - Why: `_flush_messages_to_session_db()` can append many messages one by one. Each message hits FTS triggers and session counters.
   - PR shape: add `append_messages(session_id, messages)` that performs one `BEGIN IMMEDIATE` transaction, bulk inserts, and updates counters once.

8. Denormalize session list preview/last-active metadata.
   - Files: `hermes_state.py`, `tui_gateway/server.py`, dashboard session APIs.
   - Why: session picker queries contain correlated subqueries for preview/last active. Local state was tiny, but this can grow with long-lived users.
   - PR shape: maintain `last_message_at`, `preview`, and maybe `root_session_id` summary fields on write/compression.

9. Debounce config/MCP reloads in the TUI gateway.
   - Files: `ui-tui/src/app/useConfigSync.ts`, `tui_gateway/server.py`, `tui_gateway/entry.py`.
   - Why: Codex CLI and Claude Code both design around avoiding unnecessary tool/prompt churn. Hermes comments already warn that MCP reload invalidates prompt cache.
   - PR shape: replace polling/unconditional reload behavior with config fingerprints and reload only for MCP-relevant key changes.

10. Make `/goal` loop cheaper per turn.
    - Files: `hermes_cli/goals.py`, `cli.py`, `gateway/run.py`.
    - Why: each continuation currently persists state and calls an auxiliary judge. For long goals this adds a full extra LLM request per turn.
    - PR shape: skip judge call when response clearly asks user input or explicitly says complete/block; coalesce state writes; expose a `goals.judge_every_n_turns` option for low-cost modes.

Suggested first PR:

Start with item 2 or item 7. Item 2 is a startup win with a visible local measurement trail. Item 7 is narrower and easier to test safely because it can preserve behavior while reducing SQLite write amplification.
