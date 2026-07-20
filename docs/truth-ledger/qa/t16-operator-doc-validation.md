# T16 operator doc validation evidence

Date: 2026-07-19
Workspace: /Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2

> Historical validation snapshot. The bounded-dedupe failure and export URI caveat
> below were subsequently remediated. `_SEEN_ENVELOPES` is now lock-protected and
> capped at 1,024 entries; remote export destinations are rejected. The automatic
> pending-queue drain remains deferred in the reduced MVP.

## Commands executed

1) Temp-profile plugin install/opt-in/disable flow
- Copied plugin to user-plugin path in temp profile:
  - `cp -R plugins/truth-ledger $TMP_HOME/plugins/truth-ledger`
- Enable:
  - `HERMES_HOME=$TMP_HOME /Users/hermes/.local/bin/hermes plugins enable truth-ledger`
- Disable:
  - `HERMES_HOME=$TMP_HOME /Users/hermes/.local/bin/hermes plugins disable truth-ledger`
- Result:
  - `ENABLE_RC=0`
  - `DISABLE_RC=0`

2) Headless command behavior probe (temp truth-ledger root)
- Seeded one assert event to `ledger/2026-07.jsonl`.
- Invoked `dispatch_headless()` for `status`, `rebuild --apply`, `retract --apply`, `export --apply`, and `export --apply --destination s3://bucket/export.tar.gz`.
- Output:

```json
{"export_is_tar": true, "export_mode": "0o600", "export_remote_ok": true, "export_remote_reason": null, "rebuild_dry_run": false, "retract_appended": true, "status_enabled": true}
```

Interpretation:
- Export artifacts are local tarballs with `0600` mode.
- `retract --apply` is append-only and succeeds for valid fact IDs.
- URI destination guard does not currently reject `s3://...` in this code path (documented as a caveat).

3) Focused test run
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_commands.py tests/plugins/truth_ledger/test_lifecycle_integration.py -q`
- Result summary:
  - `test_commands.py`: PASS (7/7)
  - `test_lifecycle_integration.py`: 11 pass, 1 fail
  - Failing test: `test_post_llm_call_bounds_seen_dedupe_state`
  - Failure detail: `_SEEN_ENVELOPES` length reached `3000` (> expected `1024`)

## Outcome for T16 docs
- Runbook authored at:
  - `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2/docs/truth-ledger/operator-privacy-recovery-rollout.md`
- Runbook reflects current code behavior, including known caveats:
  - pending queue drain not wired in lifecycle
  - unbounded `_SEEN_ENVELOPES`
  - export URI destination validation caveat
