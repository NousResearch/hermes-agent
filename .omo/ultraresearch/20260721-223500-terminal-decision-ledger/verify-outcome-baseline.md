# Verification — Outcome receipt root/session baseline

## Claim

Outcome receipt root/session tests are currently failing and must be fixed
before a new approval receipt can use the evidence database.

## Execution

The verification worker used the repository wrapper under Git Bash with the
Hermes development venv. The WSL shim did not have a distribution installed,
so it was explicitly excluded as an environment failure before test execution.

```text
scripts/run_tests.sh tests/agent/test_verification_evidence.py -q -k
  'test_reusable_outcome_listing_can_be_scoped_to_one_session'
1 passed

scripts/run_tests.sh tests/agent/test_verification_evidence.py -q -k
  'test_session_scoped_outcome_listing_fails_closed_without_workspace_root'
1 passed

scripts/run_tests.sh tests/agent/test_verification_evidence.py -q -k
  'test_other_session_edit_stales_outcome_receipt_for_learning_candidates or
   test_reverification_after_workspace_edit_keeps_older_receipt_stale or
   test_delayed_older_edit_cannot_restore_reusable_receipt or
   test_outcome_receipt_confirmation_requires_current_session_and_workspace'
4 passed
```

## Verdict

**REFUTED.** Current source passes the focused root/session and stale-evidence
tests. The earlier apparent failure was a WSL-shim environment failure, not a
product regression. No baseline repair is needed.

## EXPAND

none — the tested claim was refuted and all related focused variants passed.
