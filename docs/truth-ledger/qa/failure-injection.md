# T14 — Failure injection, recovery, and security hardening

Date: 2026-07-19
Task: t_df29cdae
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Machine-readable artifact: `docs/truth-ledger/qa/failure-injection-results.json`

Verdict: PASS

## Scope

Re-ran T14 with faithful boundary injection at the exact write/replace/transaction boundaries called out in review notes. The prior claims that inferred PASS from adjacent fault classes were replaced with direct tests for ENOSPC at spool payload file fsync, permission-denied at pending atomic replace, interruption between tmp-write and pending replace, interruption after ledger append before index commit, and lifecycle close/recovery disappearance race.

## RED → GREEN repairs

1) T14-F1 — Dead-letter reason visibility in operator review output
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_commands.py -q`
  - Failure: `test_review_report_reads_nested_dead_letter_reason_from_flow`
  - Symptom: expected `schema_mismatch`, got `unknown`
- Repair:
  - `plugins/truth-ledger/commands.py` resolves reason from `flow.dead_letter_reason` fallback path.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_commands.py -q`
  - Result: 7 passed, 0 failed.

2) T14-F2 — Unbounded in-memory dedupe state in lifecycle hook
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py -q`
  - Failure: `test_post_llm_call_bounds_seen_dedupe_state`
  - Symptom: `_SEEN_ENVELOPES` grew to 3000 entries (expected bounded <=1024).
- Repair:
  - `plugins/truth-ledger/__init__.py` uses lock-protected bounded `OrderedDict` with FIFO eviction at 1024.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py -q`
  - Result: 14 passed, 0 failed.

3) T14-F3 — SQLite lock path raised exception instead of retry result
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_ledger.py -q`
  - Failure: `test_append_event_returns_retry_when_index_db_is_locked`
  - Symptom: `sqlite3.OperationalError: database is locked` escaped from append path.
- Repair:
  - `plugins/truth-ledger/ledger.py` catches lock/busy `sqlite3.OperationalError` around intent insert and indexed update paths, returning fail-open retry `{status: retry, reason: index_db_locked}`.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_ledger.py -q`
  - Result: 5 passed, 0 failed.

4) T14-F4 — Pending atomic replace interruption left temp artifacts and ambiguous queue state
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_spool.py -q -k "permission_denied_at_pending_atomic_replace or interruption_between_tmp_write_and_pending_replace"`
  - Failure: stale `.tmp-*.json` records remained in pending after permission/interruption fault injection.
- Repair:
  - `plugins/truth-ledger/spool.py::_write_private_json_atomic` now unlinks tmp files when `os.replace` fails, then re-raises.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_spool.py -q -k "permission_denied_at_pending_atomic_replace or interruption_between_tmp_write_and_pending_replace"`
  - Result: targeted boundary tests passed.

5) T14-F5 — Lifecycle close/recovery disappearance race was quarantined as corruption
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_spool.py -q -k record_disappearing_mid_recovery`
  - Failure: benign concurrent removal during `recover_stale_processing` produced dead-letter artifacts.
- Repair:
  - `plugins/truth-ledger/spool.py::recover_stale_processing` now treats missing source file during exception path as benign race and skips quarantine.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_spool.py -q -k record_disappearing_mid_recovery`
  - Result: targeted race test passed.

6) T14-F6 — Interruption after append before index commit could duplicate logical event
- RED:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_ledger.py -q -k interrupted_after_append_before_index_commit`
  - Failure: second append after forced update-lock wrote duplicate JSONL line for same `event_key`.
- Repair:
  - `plugins/truth-ledger/ledger.py` now detects `intent` rows, scans ledger file for already-appended `event_id`, and promotes the journal row to indexed (`recovered_from_intent`) instead of appending again.
- GREEN:
  - Command: `scripts/run_tests.sh tests/plugins/truth_ledger/test_ledger.py -q -k interrupted_after_append_before_index_commit`
  - Result: targeted boundary test passed with single logical event line.

## Fault matrix coverage

- disk full: pass
  - `test_post_llm_call_fail_open_on_enospc_at_payload_fsync_cleans_tmp_and_retries`
- permission denied at pending atomic replace: pass
  - `test_post_llm_call_is_fail_open_on_permission_denied_at_pending_atomic_replace`
- SQLite busy/locked: pass
  - `test_append_event_returns_retry_when_index_db_is_locked`
- interruption between tmp write and pending replace: pass
  - `test_post_llm_call_recovers_from_interruption_between_tmp_write_and_pending_replace`
- interruption after ledger append before index commit: pass
  - `test_append_event_is_idempotent_when_interrupted_after_append_before_index_commit`
- malformed model JSON: pass
  - `test_extract_malformed_parsed_payload_is_conservative_dead_letter`
- schema mismatch: pass
  - `test_extract_schema_mismatch_dead_letters_without_secret_leakage`
- provider timeout/5xx: pass
  - `test_extract_timeout_is_retry_with_jitter_delay`
  - `test_extract_http_5xx_is_retry_then_dead_letter_when_attempts_exhausted`
- duplicate callback: pass
  - `test_post_llm_call_enqueues_once_and_never_persists_conversation_history`
- partial final JSONL line: pass
  - `test_scan_quarantines_partial_tail`
  - `test_rebuild_current_view_reports_and_quarantines_invalid_source_lines`
- oversized turn/record: pass
  - `test_append_event_rejects_oversized_record`
  - `test_source_envelope_and_dead_letter_size_limits`
- credential-like strings: pass
  - `test_realistic_synthetic_credential_never_leaks_across_spool_or_extractor_surfaces`
  - `test_secret_value_is_rejected_and_redacted_in_reason`
- unknown speaker: pass
  - `test_unknown_speaker_blocks_user_scope`
- concurrent writers + lifecycle close/recovery race: pass
  - `test_multiprocess_append_is_non_corrupt_and_idempotent`
  - `test_recover_stale_processing_moves_back_to_pending`
  - `test_recover_stale_processing_tolerates_record_disappearing_mid_recovery`
  - `test_on_session_start_recovers_stale_processing`

## Verification runs

- `scripts/run_tests.sh tests/plugins/truth_ledger/test_commands.py tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_ledger.py tests/plugins/truth_ledger/test_spool.py tests/plugins/truth_ledger/test_projection.py tests/plugins/truth_ledger/test_extractor.py -q`
  - Result: 51 passed, 0 failed.
- `scripts/run_tests.sh tests/plugins/truth_ledger/test_concurrency.py tests/plugins/test_truth_ledger_admission_redaction.py tests/plugins/test_truth_ledger_reconciliation.py -q`
  - Result: 24 passed, 0 failed.
- `git diff --check`
  - Result: pass.

## Residual risks

- Intent recovery scans the ledger file linearly when it finds an `intent` row. This keeps idempotency/correctness across the crash window but may add latency on very large monthly files.
- Faithful boundary injection is deterministic via fault harness tests; no live process kill signals were sent to gateway/orchestrator processes.
