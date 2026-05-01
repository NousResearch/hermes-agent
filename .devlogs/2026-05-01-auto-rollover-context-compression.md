# 2026-05-01 — auto-rollover after repeated context compression

## Goal
Prevent long-running Hermes gateway sessions from degrading through repeated lossy context compression by rolling over to a fresh session after a bounded number of successful compressions.

## Worktree
`/work/.hermes-data/worktrees/hermes-agent-auto-rollover`

## Plan/spec
`.hermes/plans/2026-05-01-auto-rollover-after-compaction.md`

## No-ADR rationale
This is a tactical reliability guard in existing gateway session/reset behavior, not a new architecture or cross-cutting design. The plan/spec plus regression tests document the decision.

## Changes
- Added durable `compression_count` metadata to `gateway.session.SessionEntry` serialization and update flow.
- Added `GatewayConfig.max_context_compressions_before_reset`, default `3`, with `0` as disable.
- Bridged root `config.yaml` key `max_context_compressions_before_reset` into `GatewayConfig` loading.
- Added `compression_count` to `AIAgent.run_conversation()` result metadata.
- Persisted returned agent compression count after each gateway turn.
- Incremented compression count when gateway pre-agent hygiene compression rewrites a transcript.
- Added `GatewayRunner._rollover_session_if_compression_limit_reached()` and called it before building the next turn's session context.
- Auto-rollover cleans up the evicted cached agent before removing it from the cache so memory/tool/process resources do not leak.
- Auto-rollover clears the `is_fresh_reset` marker immediately so the next inbound message does not duplicate session-start handling.
- Added `GatewayRunner._sync_agent_compression_count()` so persisted gateway-side hygiene compression counts seed the cached/new agent compressor before the turn.
- Added regression tests for session persistence/reset, config loading, and gateway rollover helper behavior.

## Validation
RED observed before production changes:
- `python -m pytest tests/gateway/test_session.py::TestCompressionCount -q`
- `python -m pytest tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q`
- `python -m pytest tests/gateway/test_context_compression_rollover.py -q`

GREEN:
- `python -m pytest tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py::TestCompressionCount tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q` — 13 passed
- `python -m pytest tests/gateway/test_session.py tests/gateway/test_config.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_agent_cache.py -q` — 185 passed, 20 warnings
- `python -m py_compile gateway/session.py gateway/config.py gateway/run.py run_agent.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py tests/gateway/test_config.py` — passed

## Risks / follow-up
- The rollover occurs on the next inbound message, not mid-run, by design. This avoids dropping in-flight tool/process state.
- There is no user-facing notice yet for compression-count rollover. Existing auto-expiry reset notices remain separate.
- A future operator command like `/session status` could expose compression count and impending rollover.

## Session-quality notes
The work followed worktree/spec/TDD discipline. No commit was created per user preference.
