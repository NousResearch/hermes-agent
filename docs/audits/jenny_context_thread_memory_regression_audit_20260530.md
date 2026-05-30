# Jenny Context Thread Memory Regression Audit

Date: 2026-05-30
Mode: read-only audit plus non-invasive pure tests

## Executive Summary

Jenny's Discord thread context is routed primarily by `SessionSource` fields and
`build_session_key()`. For normal Discord thread messages, the stable key is the
thread ID twice: `agent:main:discord:thread:<thread_id>:<thread_id>`. That gives
separate project threads separate gateway sessions as long as the Discord adapter
always supplies both `chat_id` and `thread_id`.

The highest-risk regression area is not the core happy-path key itself. It is the
split between durable routing state and durable transcript state:

- `sessions.json` stores `session_key -> session_id`.
- `state.db` stores session rows and transcripts.
- If `sessions.json` is missing, pruned, stale, or not updated after compression,
  the same Discord thread can start a fresh session even though the old transcript
  still exists in `state.db`.

Project walling is only partly centralized. Hermes passes rich scope data to
memory providers (`gateway_session_key`, `thread_id`, `chat_id`, active profile),
but provider-specific code decides how strongly to enforce isolation. Built-in
`MEMORY.md` and `USER.md` are profile-scoped, not Discord-thread-scoped, so they
can intentionally or accidentally bridge otherwise separate project threads that
share one Hermes profile.

No production state, Discord API, gateway restart, memory file, or live
integration was touched during this audit.

## Files And Functions Inspected

- `plugins/platforms/discord/adapter.py`
  - `DiscordAdapter._handle_message()`
  - `DiscordAdapter._build_slash_event()`
  - `DiscordAdapter._dispatch_thread_session()`
  - `DiscordAdapter._text_batch_key()`
  - channel prompt/skill resolution helpers
- `gateway/session.py`
  - `SessionSource`
  - `build_session_key()`
  - `is_shared_multi_user_session()`
  - `SessionStore.get_or_create_session()`
  - `SessionStore._generate_session_key()`
  - `SessionStore._save()` / `_ensure_loaded_locked()`
  - `SessionStore.prune_old_entries()`
  - `SessionStore.suspend_recently_active()`
  - transcript load/rewrite methods
- `gateway/run.py`
  - `GatewayRunner.__init__()`
  - `GatewayRunner._session_key_for_source()`
  - `GatewayRunner._handle_message_with_agent()`
  - `GatewayRunner._run_agent()`
  - session split persistence after compression
  - in-memory `_session_sources` and `_agent_cache`
- `gateway/session_context.py`
  - task-local ContextVars for platform/chat/thread/session metadata
- `agent/agent_init.py`
  - built-in memory loading
  - external memory provider initialization
  - compression configuration
- `agent/system_prompt.py`
  - system prompt layering and cache behavior
- `agent/conversation_loop.py`
  - system prompt restore/build
  - external memory prefetch injection
- `agent/memory_manager.py`
  - memory provider prefetch/sync hooks
- `tools/memory_tool.py`
  - profile-scoped file-backed memory store
- `agent/context_compressor.py`
  - summarization template, fallback behavior, protected windows
- `agent/conversation_compression.py`
  - session rotation and memory-provider session switch on compression
- Relevant tests under `tests/gateway/`, `tests/agent/`, and `tests/run_agent/`

## Current Routing Model

Discord messages enter through `DiscordAdapter._handle_message()` in
`plugins/platforms/discord/adapter.py`. A Discord thread is detected with
`isinstance(message.channel, discord.Thread)`. For real thread messages:

- `thread_id = str(message.channel.id)`
- `parent_channel_id = self._get_parent_channel_id(message.channel)`
- `effective_channel = message.channel`
- `source.chat_id = str(effective_channel.id)`
- `source.chat_type = "thread"`
- `source.thread_id = thread_id`
- `source.guild_id`, `source.parent_chat_id`, and `source.message_id` are also set

For normal Discord thread messages, this means `chat_id == thread_id`, and
`build_session_key()` returns:

```text
agent:main:discord:thread:<thread_id>:<thread_id>
```

Discord slash command events also set `chat_type="thread"` and
`thread_id=str(interaction.channel_id)` when invoked inside a thread. However,
`_build_slash_event()` does not include `guild_id`, `parent_chat_id`, or
`message_id` in the source. The `/thread` starter path in
`_dispatch_thread_session()` likewise routes to `chat_id=thread_id` and
`thread_id=thread_id`, but does not populate `parent_chat_id`/`guild_id`.

Thread gating and allow-list checks consider both thread ID and parent channel ID.
Session scope does not: `build_session_key()` uses `chat_id`, `thread_id`, and
optionally participant ID. It does not include parent channel ID or guild ID.
That is acceptable for Discord snowflakes because IDs are globally unique, but it
means walling cannot be expressed by parent/guild in the session key itself.

## Current Memory And Context Loading Model

Before each gateway response, `GatewayRunner._handle_message_with_agent()`:

1. Calls `SessionStore.get_or_create_session(source)`.
2. Builds `SessionContext` from the source, gateway config, and session entry.
3. Sets task-local ContextVars for routing metadata.
4. Builds an ephemeral session context prompt with `build_session_context_prompt()`.
5. Auto-loads channel/thread skill bindings only on new sessions.
6. Loads transcript history with `SessionStore.load_transcript(session_id)`.
7. Runs session hygiene compression if the transcript is too large.
8. Passes transcript history into `AIAgent.run_conversation()`.

Inside the agent, the system prompt is built/restored in this order:

1. Stable identity/tool/platform/skill guidance.
2. Caller system prompt and context files from `TERMINAL_CWD`.
3. Built-in memory snapshot from profile-scoped `memories/MEMORY.md`.
4. Built-in user profile snapshot from profile-scoped `memories/USER.md`.
5. External memory provider system prompt block.
6. date/session/model/provider line.

The gateway's session context prompt and per-channel prompt are passed as
`ephemeral_system_prompt`; they are added at API-call time and are not stored in
the persisted system prompt. External memory provider prefetch is injected into
the current user message at API-call time and is not persisted into the transcript.

Built-in memory writes are durable immediately, but the built-in memory snapshot
in the system prompt is frozen for the agent/session. It refreshes on a new
session or after system-prompt invalidation, such as compression.

## Current Isolation And Walling Model

The hard session boundary is `build_session_key()`:

- DMs: keyed by platform, DM chat ID, and optional thread ID.
- Non-DM channels/groups: keyed by platform, chat type, chat ID, optional thread
  ID, and optionally participant ID.
- Threads are shared across participants by default because
  `thread_sessions_per_user=False`.
- If `thread_sessions_per_user=True`, participant ID is appended.

For Jenny's project threads, the intended isolation unit appears to be the
Discord thread session key, not the parent channel. The active Hermes profile is
also included in system prompt guidance and passed to memory providers as
`agent_identity`, but the gateway session key itself does not include profile.

Likely leakage routes:

- Built-in `MEMORY.md` and `USER.md` are profile-global, not thread-global.
- External memory providers receive scope inputs but enforce isolation
  provider-by-provider.
- Context files come from `TERMINAL_CWD`, not Discord thread metadata. All Discord
  project threads sharing the same gateway `terminal.cwd` see the same repo
  context files unless differentiated by profile, channel prompt, skill binding,
  or memory provider.
- Parent/guild IDs are prompt metadata, not session-key dimensions.
- Missing `thread_id` splits a normal Discord thread scope into a different key
  that can behave like a per-user thread session.

## Current Summarization And Compression Model

The default context engine is `ContextCompressor`. It is lossy by design:

- Default threshold is `compression.threshold` or model-specific override,
  usually around 50% of context.
- Default `target_ratio` is 0.20.
- Default protected head is `protect_first_n=3`.
- Default protected tail is `protect_last_n=20`.
- Summary model is resolved through the auxiliary compression client; failures
  can fall back to the main model.
- If `abort_on_summary_failure` is false, historical fallback behavior can insert
  a marker and drop the middle window; comments indicate newer abort behavior is
  also tracked via `_last_compress_aborted`.

Compression creates a continuation session in `state.db`, links it to the old
session as `parent_session_id`, rebuilds the system prompt, and notifies external
memory providers with `on_session_switch(..., reason="compression")`. The gateway
then updates the same `session_key` to the new `session_id` in `sessions.json`.

Summaries live in the session transcript lineage, not in a separate
thread/project memory store. They should not overwrite another thread's summary
unless the session-key mapping points at the wrong session or the mapping is lost.

## Persistence And Restart Behavior

`SessionStore` persists the active routing index in `sessions.json`. The SQLite
session database stores transcript messages. On a gateway restart:

- `sessions.json` reconstructs the active `session_key -> session_id` mapping.
- `state.db` reconstructs transcript history once a session ID is known.
- `_session_sources` is only an in-memory LRU cache and is not durable.
- `_agent_cache` is only an in-memory LRU cache and should not be required for
  transcript continuity.

If `sessions.json` is pruned, missing, stale, or not saved after compression, the
gateway cannot reconstruct the active Discord thread mapping from `state.db`
alone. It creates a new session ID for the same thread key and loads an empty
transcript. The old transcript remains in `state.db` but is no longer the active
session for that thread.

`SessionStore.prune_old_entries()` explicitly drops old `sessions.json` entries
while preserving transcripts in SQLite. That is functionally a context reset for
the next message in that thread.

## Concrete Likely Regression Causes

1. High confidence: loss or pruning of `sessions.json` mapping.
   Evidence: `SessionStore._save()` persists only `sessions.json`, while
   `load_transcript()` requires an already-known `session_id`. `prune_old_entries()`
   removes the mapping and leaves the transcript behind.

2. High confidence: session split after compression not persisted or interrupted.
   Evidence: gateway compression paths mutate `session_entry.session_id` and must
   call `session_store._save()`. Existing regression test
   `tests/gateway/test_compression_session_id_persistence.py` pins this.

3. Medium confidence: source construction drift where a Discord thread event loses
   `thread_id`.
   Evidence: normal Discord thread events set both `chat_id` and `thread_id`;
   the new pure test shows that omitting `thread_id` produces a different,
   per-user key.

4. Medium confidence: mismatch between adapter `config.extra` and gateway config.
   Evidence: `DiscordAdapter._text_batch_key()` reads `self.config.extra`, while
   `SessionStore._generate_session_key()` reads `GatewayConfig` attributes. If
   `thread_sessions_per_user` or `group_sessions_per_user` diverge, batching and
   active-session tracking can disagree with final session storage.

5. Medium confidence: project walling relies on provider/profile convention, not
   one central policy.
   Evidence: external memory receives scope inputs, but built-in memory is only
   profile-scoped and context files are cwd-scoped.

6. Medium confidence: compression drops project-critical state if summary quality
   is poor or old decisions are outside the protected tail.
   Evidence: `ContextCompressor` is explicitly lossy and summaries are structured
   but not schema-validated against project isolation invariants.

## Existing Tests And Gaps

Existing useful coverage:

- `tests/gateway/test_discord_thread_persistence.py`
  - covers Discord thread participation tracker persistence.
- `tests/gateway/test_session_dm_thread_seeding.py`
  - covers thread sessions starting empty and multiple threads remaining
    independent, including Discord.
- `tests/gateway/test_base_topic_sessions.py`
  - covers topic-aware active-session handling and reply metadata.
- `tests/gateway/test_agent_cache.py`
  - covers stale cached agents not being reused across session IDs and cache caps.
- `tests/gateway/test_compression_session_id_persistence.py`
  - covers persistence of post-compression session ID changes.
- `tests/gateway/test_telegram_topic_mode.py`
  - covers stronger restart/reconstruction semantics for Telegram topic bindings.
- `tests/agent/test_context_compressor_summary_continuity.py`,
  `tests/run_agent/test_compression_persistence.py`, and related compression tests
  cover general compression continuity.

New audit-only tests added:

- `tests/gateway/test_discord_thread_memory_scope.py`
  - same Discord thread returns the same key.
  - different Discord threads under the same parent channel do not collide.
  - Discord thread scope is shared across users by default.
  - `thread_sessions_per_user=True` isolates users.
  - missing `thread_id` splits scope from the normal thread key.

Coverage gaps:

- No end-to-end fake Discord adapter test proving `_handle_message()` produces the
  exact expected `SessionSource` for real thread, auto-thread, slash-in-thread,
  and `/thread` starter paths.
- No restart test proving a Discord thread reloads the same `session_id` from
  `sessions.json` and then pulls the expected transcript from a temp `state.db`.
- No test proving `state.db` alone cannot reconstruct Discord thread mapping, or
  that an operator warning is emitted when mapping is missing.
- No cross-check test ensuring adapter text batching uses the exact same config
  source and key as `SessionStore`.
- No central isolation test for memory providers using `gateway_session_key`.
- No project-wall test proving built-in profile memory is excluded from walled
  project threads, because current built-in memory is profile-global.
- No compression test requiring project decisions to survive summaries across
  repeated compactions in a Discord-thread-shaped session.

## Proposed Fix Sequence

1. Add a single helper for Discord thread `SessionSource` construction and use it
   for normal messages, slash-in-thread, auto-thread, and `/thread` starter flows.
2. Add adapter-level tests using fake Discord channel/thread objects to prove all
   Discord thread ingress paths produce identical scope fields.
3. Add a restart reconstruction test with temp `sessions.json` and temp `state.db`
   for Discord thread sessions.
4. Add a warning/diagnostic when a Discord thread has transcript candidates in
   `state.db` but no active `sessions.json` mapping.
5. Make adapter batching and gateway storage use the same resolved session-key
   function/config source.
6. Define the project walling contract explicitly:
   - thread-key isolation only,
   - profile isolation,
   - configured project/workspace isolation,
   - or memory-provider namespace isolation.
7. Add memory-provider conformance tests requiring provider scope keys to include
   `gateway_session_key` or an explicit project namespace.
8. Add compression regression tests with durable project decisions and repeated
   compaction.

## Proposed Regression Tests Before Behavior Changes

- `test_discord_message_thread_source_contains_thread_parent_guild_message_ids`
- `test_discord_slash_thread_source_matches_message_thread_scope`
- `test_discord_thread_starter_source_matches_existing_thread_scope`
- `test_discord_auto_thread_source_uses_created_thread_as_scope`
- `test_discord_thread_session_survives_gateway_restart_from_sessions_json`
- `test_discord_thread_mapping_loss_warns_and_does_not_silently_claim_context`
- `test_discord_text_batch_key_matches_session_store_key`
- `test_walled_project_memory_provider_namespace_does_not_cross_thread`
- `test_compression_preserves_project_decisions_in_thread_session`
- `test_compression_session_split_persists_before_next_gateway_turn`

## Adapter Source Test Follow-up

Fake Discord adapter tests now cover normal thread messages, slash commands in
threads, `/thread` starter dispatch, and auto-thread routing. Those paths build
the stable thread key
`agent:main:discord:thread:<thread_id>:<thread_id>` when `chat_id` and
`thread_id` both point at the Discord thread.

The batching tests also document a remaining config-source drift: Discord text
batching reads `group_sessions_per_user` and `thread_sessions_per_user` from
`PlatformConfig.extra`, while `SessionStore` reads them from `GatewayConfig`.
If those values are not bridged into the adapter extra config, text batching can
group chunks by a different key than the gateway session store uses.

## Memory Walling Follow-up

Regression coverage in `tests/agent/test_memory_walling_scope.py` now pins the
memory-provider walling contract for Discord-thread-shaped sessions. External
providers receive non-sensitive scope metadata including `gateway_session_key`,
`platform`, `chat_type`, `chat_id`, `thread_id`, and `agent_identity` during
initialization. Built-in memory write mirroring now also includes Discord scope
metadata so providers that mirror `MEMORY.md` / `USER.md` writes can keep those
writes in the same project-thread namespace as turn sync and prefetch.

The tests confirm a provider that namespaces by `gateway_session_key` can isolate
Project A and Project B threads under the same profile. They also document the
current leakage mode: a provider that ignores thread scope and namespaces only by
profile/agent identity will expose memories across otherwise separate Discord
project threads.

Built-in `MEMORY.md` and `USER.md` remain profile-scoped. Temp-dir tests confirm
two Discord project threads using the same Hermes profile receive the same
built-in memory snapshot. No project-wall config or built-in memory namespace was
added in this batch.

Built-in walling is now design-only in
`docs/design/builtin_memory_walling_design_20260530.md`. Strict xfail tests in
`tests/agent/test_memory_walling_scope.py` pin the intended future behavior:
walled projects should exclude profile-global built-in memory by default, and
Discord thread-scoped mode should resolve distinct built-in memory paths from
thread scope rather than sharing `<HERMES_HOME>/memories`.

Proposed safe sequence:

1. Introduce an explicit `MemoryScope` object shared by provider init, turn sync,
   prefetch, and built-in memory-write mirroring.
2. Add provider conformance tests requiring providers to either honor
   `gateway_session_key` or declare that they are profile/global only.
3. Design built-in memory walling separately, including operator-visible defaults
   and migration/non-migration behavior for existing `MEMORY.md` / `USER.md`.

## Operational Diagnostics Follow-up

`tools/inspect_discord_thread_context.py` provides a metadata-only operator
diagnostic for Discord thread routing. It accepts explicit local paths and does
not call Discord, restart the gateway, read memory files, or query transcript
content.

Example:

```bash
python3 tools/inspect_discord_thread_context.py \
  --thread-id <discord_thread_id> \
  --state-root <path-containing-state.db-and-sessions-dir> \
  --no-content
```

The report includes the expected Discord thread session key, whether
`sessions.json` maps it to a session, active transcript counts and last
timestamp metadata, orphan candidate counts, and whether the missing-mapping
diagnostic would fire. Tests in
`tests/tools/test_inspect_discord_thread_context.py` cover mapped, missing
mapping, missing path, read-only, and no-content behavior using temp paths only.

## Thread Inventory Follow-up

`tools/list_discord_thread_context_inventory.py` provides a read-only inventory
of mapped Discord thread sessions from an explicit state root. It reads
`sessions.json` and aggregate `state.db` metadata only, and never queries
transcript or memory content.

The inventory tests in `tests/tools/test_list_discord_thread_context_inventory.py`
cover mapped thread sessions with nonzero transcript counts, zero transcript
counts, available names, missing names, `--limit`, JSON output, missing paths,
and read-only behavior.

Live metadata validation found two mapped Discord thread sessions with zero
transcript rows:

- `1507598956752928820`:
  `Travisaggie04's server / #general / Family Hub -- Part 17`
- `1507081077196460185`:
  `Travisaggie04's server / #general / Family Hub Public App -- Part 4`

Those zero-row sessions have active `sessions.json` mappings but no persisted
transcript rows in `state.db`, so they look like empty mapped sessions rather
than exact orphan/missing-mapping recoveries.

## Operational Checks To Run From The VPS

Run these only on the VPS/operator side, not from this audit session:

- Check the gateway service status and task count without restart:
  `systemctl --user status hermes-gateway`
- Check bounded process count for Hermes/Codex-related processes:
  `ps -fu "$USER" | grep -E "hermes|codex|node|python" | grep -v grep`
- Inspect recent gateway logs for session mapping/compression warnings:
  `hermes logs --level warning --session recent`
- Confirm current gateway session index exists and has recent mtime without
  printing its contents.
- Confirm `state.db` exists and has recent mtime without querying or dumping user
  transcripts.
- For one known affected Discord project thread, compare only non-sensitive
  metadata:
  - expected Discord thread ID
  - active `session_key`
  - mapped `session_id`
  - transcript message count
  - last updated timestamp
- Verify Codex app-server process count remains bounded after several Discord
  turns.

## Risks And Non-goals

Risks:

- Changing session-key format directly would orphan existing thread mappings
  unless migrated carefully.
- Reading or modifying production `sessions.json`, `state.db`, memories, or
  provider stores could destroy evidence or alter Jenny's live context.
- Fixing only one observed ingress path can leave slash/auto-thread paths
  inconsistent.
- Adding runtime diagnostics must not spawn long-running processes or recreate
  the prior Codex app-server process leak.

Non-goals for this batch:

- No production gateway restart.
- No Discord API calls or outbound Discord messages.
- No memory migration, pruning, compaction, or rewrite.
- No behavior changes beyond adding pure tests and this report.
- No changes to memory-provider implementations.
