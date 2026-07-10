# Resolution decisions - 2026-07-10

## Semantic reconciliations

- `plugins/memory/mem0`: take-fork per spec. Removed upstream refactor shards/tests and documented follow-up in `mem0-resolution.md`.
- `agent/auxiliary_client.py`: reconciled provider `api_mode` behavior. First-class/custom/auto providers keep upstream `api_mode=None` contract for downstream transport detection; plugin providers may read `ProviderProfile.api_mode` to preserve the fork fix.
- `agent/chat_completion_helpers.py`: keep-both. Preserved fork relay/fallback route announcements and audit sink; added upstream unavailable-fallback skip and stale-breaker reset.
- `tools/delegate_tool.py`: kept fork `inherit_context`/boomerang behavior and verified `code_execution` remains explicitly exempt from subagent toolset stripping. Took upstream removal of model-controlled ACP transport fields.
- `gateway/run.py` and `gateway/slash_commands.py`: keep-both/interleave. Preserved fork restart-loop/recovery gates, compaction telemetry, branch thread behavior, and last-turn usage cards; used upstream async session-store calls where hunk-local and safe.
- `gateway/session.py`: keep-both. Preserved fork durable reasoning/model identity fields and answerable platform-message lookup; kept upstream `model_override` and `rewrite_transcript()` success return, clearing undo state only after successful rewrite.
- `hermes_state.py`: keep-both. Preserved fork gated `effective_last_active` denorm path and backfill; added upstream `MAX_FTS5_QUERY_CHARS`, handoff index, `search_query`, and `compact_rows`. Denorm fast path is bypassed when upstream-only search/compact options are requested.
- `cron/scheduler.py`: keep-both. Preserved fork per-job reasoning/fallback chain and loud fallback alert; added upstream dispatch claiming, credential-exfil guard, Telegram DM-topic probe, and deferred agent teardown.
- `tui_gateway/server.py`: reconciled fork client `source` attribution with upstream `platform_override` naming. `_make_agent` accepts both and reports disabled reasoning as `"none"`.
- `run_agent.py`: interleaved upstream intrinsic persistence markers with fork interrupt-close finish-reason re-persist. Row ids are also attached to persisted message dicts so the fork resume discriminator does not depend solely on long-lived `id()` sets.
- `agent/model_metadata.py`: keep-both. Preserved fork two-tier token breakdown and request composition telemetry; added upstream async context lookup, local-probe cache, and bounded tool-token cache.
- Desktop conflicts: took upstream Electron TypeScript migration/package script shape; preserved fork app-side remote media/project/composer behavior where conflicts were directly in app code.

## Review flags

- `git grep -nE '^(<<<<<<<|=======|>>>>>>>)'` still reports pre-existing decorative `=======` heading/divider lines in unrelated files (for example `tests/tools/test_mcp_oauth_metadata.py:10`). A stricter conflict-marker scan has no `<<<<<<<`/`>>>>>>>` lines and no conflict labels.
- Several conflicting test hunks were resolved to the fork contract; see `test-conflicts.md`.
- Discord inbound handler keeps the fork `_dispatch_incoming_message()` path to preserve restart backfill parity. Upstream's newer inline channel-id/fail-closed refinements should be ported into `_dispatch_incoming_message()` in a focused follow-up if the pytest/security gates require it.
