# Auto-rollover gateway sessions after repeated context compaction

## Problem
Long-running gateway sessions can survive for many turns and repeatedly compress their context. Each compression is lossy. After several rounds, the leading context becomes a summary-of-summary and starts degrading task fidelity. Operators need the gateway to start a fresh command/session lane automatically before degraded summaries become the dominant context.

## Decision
Add a bounded automatic rollover trigger based on successful compression count per gateway session.

Default policy:
- Track successful context compressions in `gateway.session.SessionEntry` as durable metadata.
- Increment the counter whenever gateway hygiene compression rewrites a transcript, and after a normal agent turn if the agent reports its context engine compressed during that turn.
- Before running the agent for a new inbound message, if the session has reached the configured compression limit, auto-reset it into a new session id and run the message there.
- Do not reset in the middle of an active agent run; wait until the next inbound message. This avoids dropping in-flight tool/process state.
- Default limit: 3 successful compressions. `0` disables the guard.

## No-ADR rationale
This is a tactical reliability guard inside existing gateway session hygiene/reset policy rather than a new architecture. The repo does not appear to have an ADR convention. The implementation plan plus tests document the decision sufficiently.

## Acceptance criteria
- `SessionEntry` persists `compression_count` through `to_dict`/`from_dict`.
- `SessionStore.update_session()` can update compression count atomically after a turn.
- Manual reset creates a fresh entry with compression count reset to 0.
- Gateway auto-rolls over before an inbound message when `compression_count >= max_context_compressions_before_reset`.
- Gateway increments compression count after successful preflight/hygiene compression.
- Gateway increments/sets compression count after `_run_agent()` returns a higher `compression_count` from the agent context engine.
- AIAgent result includes context engine `compression_count`.
- Tests observe RED before production changes and GREEN after.

## Likely files
- `gateway/config.py`
- `gateway/session.py`
- `gateway/run.py`
- `run_agent.py`
- `tests/gateway/test_session.py`
- `tests/gateway/test_session_hygiene.py`

## Test strategy
Targeted tests first:
1. Session persistence/update/reset coverage for `compression_count`.
2. Gateway handler test for auto-reset before agent run when threshold reached.
3. Gateway hygiene test that successful preflight compression increments the counter.
4. Existing gateway hygiene/session regression tests.

Validation commands:
- `python -m pytest tests/gateway/test_session.py::TestCompressionCount -q` (RED before production change, GREEN after)
- `python -m pytest tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q` (RED before production change, GREEN after)
- `python -m pytest tests/gateway/test_context_compression_rollover.py -q` (RED before production change, GREEN after)
- `python -m pytest tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py::TestCompressionCount tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q` — 13 passed
- `python -m pytest tests/gateway/test_session.py tests/gateway/test_config.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_agent_cache.py -q` — 185 passed, 20 warnings
- `python -m py_compile gateway/session.py gateway/config.py gateway/run.py run_agent.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py tests/gateway/test_config.py` — passed
