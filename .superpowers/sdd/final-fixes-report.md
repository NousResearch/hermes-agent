# M6-03 Hermes final-review fixes

## Scope

- API session chat now resolves a deterministic parent to its active compression continuation for both sync and SSE execution.
- Effective execution uses the continuation transcript and immutable policy, and responses expose the effective session ID.
- Ordinary API forks remain independent and are not followed as compression continuations.
- Added direct execution-policy conflict and executor-thread whitelist cleanup regressions.
- Removed the unused session-message binding and documented `post_llm_call` suppression plus compression continuity.

## Tests added

- `test_create_session_enriches_execution_policy_once_on_conflict`
- `test_run_agent_read_only_generation_clears_dispatch_whitelist_on_worker` (normal and exception cases)
- `test_session_chat_parent_resolves_active_compression_tip`
- `test_session_chat_stream_parent_resolves_active_compression_tip`
- `test_session_chat_does_not_resolve_ordinary_api_fork`

## Verification

- `scripts/run_tests.sh tests/gateway/test_session_api.py tests/test_hermes_state.py -q` reached 340 passing `SessionDB` tests, then hit the 120-second command bound while the API test file remained running.
- `scripts/run_tests.sh tests/gateway/test_session_api.py -q` reproduced the 120-second bounded timeout.
- Canonical split `scripts/run_tests.sh -j 1 --file-timeout 90 tests/gateway/test_session_api.py -q`: bounded timeout at 90.1 seconds, 0 tests reported and 0 assertion failures before the runner killed the process tree.
- Canonical split across `test_api_server_toolset.py`, `test_read_only_generation_policy.py`, and `test_background_review_toolset_restriction.py`: 29 passed.
- Canonical Gmail adapter suite: 19 passed.
- Direct API pytest diagnostics identify the environment constraint: aiohttp loopback socket creation fails with `PermissionError: [Errno 1] Operation not permitted` inside the sandbox. The requested elevated loopback run was rejected by the approval service because its usage limit was reached.
- `.venv/bin/python -m pytest tests/gateway/test_session_api.py -q -k 'run_agent_read_only_generation'`: 3 passed.
- `.venv/bin/python -m pytest tests/test_hermes_state.py -q -k 'execution_policy_once'`: 1 passed.
- Python compilation and `git diff --check`: passed.
- Direct compression resolver smoke: passed and selected `compression-tip` with `read_only_generation` policy.

The API test-file timeout is the same previously documented harness behavior; the sandbox also blocks the aiohttp loopback sockets used by direct pytest. It is not claimed green.

## Self-review

- The resolver delegates to the existing compression-only `SessionDB.get_compression_tip()` marker instead of the broader resume resolver, preserving ordinary fork behavior.
- Policy fallback is non-mutating and only fills a missing child policy from the requested compression ancestor for the active execution.
- Sync and stream select the effective session before history loading and agent creation; both expose the effective ID.
- No service, config, secret, dependency, installed skill implementation, live API/state, push, deploy, or Gmail write was performed.
