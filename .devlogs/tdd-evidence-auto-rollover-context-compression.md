# TDD evidence — auto-rollover after repeated context compression

## RED runs observed before production fixes
- `python -m pytest tests/gateway/test_session.py::TestCompressionCount -q` — failed before `SessionEntry.compression_count` / `SessionStore.update_session(compression_count=...)` existed.
- `python -m pytest tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q` — failed before `GatewayConfig.max_context_compressions_before_reset` and root YAML bridge existed.
- `python -m pytest tests/gateway/test_context_compression_rollover.py -q` — failed before `GatewayRunner._rollover_session_if_compression_limit_reached()` existed.
- `python -m pytest tests/gateway/test_context_compression_rollover.py::TestContextCompressionRollover::test_syncs_persisted_count_into_agent_before_turn -q` — failed before `GatewayRunner._sync_agent_compression_count()` existed.
- `python -m pytest tests/gateway/test_context_compression_rollover.py::TestContextCompressionRollover::test_rolls_session_when_compression_limit_reached -q` — failed before auto-rollover cleaned up the evicted cached agent.

## GREEN / regression validation
- `python -m pytest tests/gateway/test_context_compression_rollover.py -q` — 4 passed.
- `python -m pytest tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py::TestCompressionCount tests/gateway/test_config.py::TestGatewayConfigCompressionRollover -q` — 13 passed.
- `python -m pytest tests/gateway/test_session.py tests/gateway/test_config.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_agent_cache.py -q` — 185 passed, 20 warnings.
- `python -m py_compile gateway/session.py gateway/config.py gateway/run.py run_agent.py tests/gateway/test_context_compression_rollover.py tests/gateway/test_session.py tests/gateway/test_config.py` — passed.
- `git diff --check` — passed.

## Review gate
- Delegated review found three real issues: gateway wrapper omitted `compression_count`, auto-rollover left `is_fresh_reset=True` for the next turn, and auto-rollover evicted cached agents without cleaning up their resources.
- These findings were fixed and covered/validated by the final focused and regression commands above.
