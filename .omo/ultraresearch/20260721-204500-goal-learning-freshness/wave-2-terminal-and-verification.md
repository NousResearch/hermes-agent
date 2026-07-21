# Wave 2 — terminal proposal semantics and verification baseline

## Findings

- Only explicit integrity/freshness failures are safe terminal outcomes.
  Ordinary apply failures still include temporary store errors, capacity,
  validation and recoverable drift, so their existing release-and-retry
  semantics remain intact.
- New memory V2 records bind a canonical record digest and a target-bound
  revision.  Legacy, malformed, modified, or revision-mismatched records are
  claimed and completed without application.  There is no ungrounded TTL.
- The goal evidence suite is green when pytest uses a base temp directory
  outside the home Git ancestor:
  `python -m pytest --basetemp C:\\tmp\\hermes-verify-evidence-basetemp-20260721-204500 tests/agent/test_verification_evidence.py -q`
  produced `29 passed`.
- The requested provenance worker could not start because its selected model
  was at capacity.  Parent source review independently traced the ContextVar
  and implemented a bounded origin-preserving replay that accepts only
  `foreground` or `background_review`.

## Evidence

`tools/write_approval.py:296-370`,
`hermes_cli/write_approval_commands.py:108-151`,
`tools/skill_manager_tool.py:1271-1342`, and
`tools/skill_provenance.py`.

## EXPAND

- DEAD END: terminalizing every apply failure — retry is an existing contract for valid records and must remain available for recoverable failures.
- DEAD END: arbitrary proposal TTL — no product policy or configuration provides a supported duration.
- DEAD END: the 19 outcome-evidence failures as a feature regression — explicit outside-home `--basetemp` executes the full suite green.
- LEAD: full Skills V2 static review artifact and CAS — WHY: provenance preservation fixes a current guard bypass but does not make a live diff immutable — ANGLE: separate record-aware replay and action-specific locking design.
- LEAD: terminal decision receipt ledger — WHY: terminal handling is user-visible but not append-only audited yet — ANGLE: design raw-payload-free receipt retention independently of the mutation correctness patch.
