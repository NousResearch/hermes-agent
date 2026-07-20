# Q3 — Specification/compliance review (extraction/projection/integration)

Date: 2026-07-19
Task: t_377f52ff
Reviewer: automation-operator (fresh worker)
Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Commit under review: `48ee34c2bf448a480ef499bb8d24d184d0fcc198`

## Scope

Independent Q3 compliance review for T9-T12 against plan/Q1 contracts. No fixes in this task.

Checked domains:
- Structured extraction API surface and schema handling (T9)
- Projection/replay semantics and deterministic rebuild behavior (T10)
- Lifecycle eligibility/fail-open capture behavior and worker/subagent exclusion (T11)
- Operator safety for status/review/rebuild/retract/export and private API absence (T12)
- No plugin-managed writes to curated memory/GBrain in this scope

## Independent rerun evidence

### Canonical focused suite
Command:
- `scripts/run_tests.sh tests/plugins/truth_ledger/test_lifecycle_integration.py tests/plugins/truth_ledger/test_commands.py tests/plugins/truth_ledger/test_extractor.py tests/plugins/truth_ledger/test_projection.py tests/plugins/test_truth_ledger_reconciliation.py tests/plugins/test_truth_ledger_admission_redaction.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/agent/test_turn_finalizer_interrupt_alternation.py tests/agent/test_turn_finalizer_final_response_persistence.py tests/agent/test_turn_finalizer_cleanup_guard.py tests/plugins/truth_ledger/test_spool.py -q`

Result:
- 11 files discovered
- 82 passed, 0 failed

### Temp-profile replay/retract/rebuild/export probe
Command:
- `python /private/tmp/q3_probe.py`

Probe behavior:
- disposable `HERMES_HOME` profile
- plugin `register()` validated hook/command registration
- eligible `on_post_llm_call` invoked twice with same `(profile,session,turn)` and deduped to one pending spool record
- replay path validated (`claim_next -> retry_processing -> claim_next -> ack_processing`) with `attempt_count_on_reclaim=1`
- operator command surface exercised in dry-run and apply modes

Observed result (key assertions):
```json
{
  "registered_hooks": ["on_session_start", "post_llm_call"],
  "registered_commands": ["truth-ledger"],
  "spool": {
    "pending_after_dedupe": 1,
    "attempt_count_on_reclaim": 1,
    "pending_after_ack": 0,
    "processing_after_ack": 0
  },
  "commands": {
    "status_ok": true,
    "dry_retract_dry_run": true,
    "apply_retract_appended": true,
    "dry_rebuild_dry_run": true,
    "apply_rebuild_has_backup": true,
    "dry_export_dry_run": true,
    "apply_export_exists": true,
    "apply_export_mode": "0o600"
  },
  "safety": {
    "private_api_cli_ref_present": false
  }
}
```

### Direct static compliance checks
Command:
- `search_files(pattern='USER\.md|MEMORY\.md|GBrain|gbrain|memory\(|ctx\.memory|write_memory|memory_provider|conversation_history', path='plugins/truth-ledger', file_glob='*.py')`

Result:
- only redaction-related `conversation_history` sanitization references found
- no curated-memory (`USER.md`/`MEMORY.md`) or GBrain write surfaces in plugin code

## Compliance findings

### Summary decision
PASS.

No critical or important compliance gaps were reproduced in T9-T12 scope.

### Findings table

| ID | Severity | Component | Evidence location | Reproduction | Expected behavior | Actual behavior | Owner |
|---|---|---|---|---|---|---|---|
| Q3-F1 | info | Structured extraction API usage | `plugins/truth-ledger/extractor.py:155-265`; `tests/plugins/truth_ledger/test_extractor.py:111-260` | Run extractor tests including schema mismatch, timeout/5xx retry, override modes | Uses `ctx.llm.complete_structured` with schema contract; conservative dead-letter on mismatch; retries on transient failures | 8/8 extractor tests pass; behaviors match contract | truth-ledger plugin |
| Q3-F2 | info | Hook-path network isolation and eligibility gating | `plugins/truth-ledger/__init__.py:51-67,116-138`; `tests/plugins/truth_ledger/test_lifecycle_integration.py:60-99,101-160,243-269` | Rerun lifecycle tests and temp-profile enqueue probe | `post_llm_call` path is fail-open, excludes failed/interrupted/worker/subagent contexts, and does not perform extraction-network calls | 11/11 lifecycle tests pass; hook enqueue dedupe and fail-open behavior reproduced | truth-ledger plugin |
| Q3-F3 | info | Replay semantics for spool processing | `plugins/truth-ledger/spool.py:277-343`; `tests/plugins/truth_ledger/test_spool.py:91-107,144-154` + temp probe | claim/retry/reclaim/ack processing flow | retry increments attempt and allows deterministic reclaim without record loss | Probe shows reclaim attempt_count=1 and zero pending/processing after ack | truth-ledger plugin |
| Q3-F4 | info | Projection deterministic rebuild and corrupt-tail quarantine | `plugins/truth-ledger/projection.py:66-133`; `tests/plugins/truth_ledger/test_projection.py:153-227` | Run projection tests | derived view rebuild is deterministic, handles operation field + retract semantics, quarantines malformed tails | 7/7 projection tests pass; invalid tail quarantine contract holds | truth-ledger plugin |
| Q3-F5 | info | Operator safety surface (dry-run default + append-only retract + protected export) | `plugins/truth-ledger/commands.py:180-347,380-474`; `tests/plugins/truth_ledger/test_commands.py:69-153` + temp probe | status/retract/rebuild/export dry-run and apply mode | safe defaults, append-only retract, rebuild backup before replace, export local protected tarball | 6/6 command tests pass; temp probe confirms dry-run booleans and export mode `0o600` | truth-ledger plugin |
| Q3-F6 | info | Private API absence and memory/GBrain non-write boundary | `plugins/truth-ledger/__init__.py` and global search over `plugins/truth-ledger/*.py` | static scan for `_cli_ref`, USER/MEMORY/GBrain write surfaces | plugin does not depend on private CLI ref and does not mutate curated memory/GBrain | `_cli_ref` absent in plugin init; no memory/GBrain write references found | truth-ledger plugin |

## T9-T12 acceptance mapping

- T9 (structured API extraction): PASS
- T10 (projection/replay semantics): PASS
- T11 (eligibility + fail-open lifecycle integration): PASS
- T12 (operator safety + private API absence): PASS

## Residual risk notes (non-blocking)

1) Focused suite currently reports 82 tests (not 83 from earlier handoff snapshots); no failures observed, but future gate scripts should avoid hard-coding prior test-count snapshots.
2) This Q3 card validates plugin-local behavior and temp-profile command semantics; full runtime matrix items (interactive/gateway/headless/profile isolation matrix from plan Q3) should remain validated by dedicated runtime QA cards/gates.

## Final QA decision

Q3 specification/compliance review: PASS for extraction/projection/integration scope (T9-T12), with no reproduced critical/important gaps.
