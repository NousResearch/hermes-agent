# Wave 2 — integration and safety digest

## Verified integration boundaries

- `GoalState` remains session-owned state in `state_meta`; the receipt ledger
  remains separate in `verification_evidence.db`.
- Existing wait resumption is at-least-once and CAS-backed. No new event queue,
  generic task database, synthetic turn, model tool, approval bypass, memory
  write, or trace upload is required for the outcome surface.
- Existing receipt confirmation already binds session and workspace and hides
  foreign receipt ids. The new display must retain those boundaries itself.

## Review findings and repairs

1. Added direct CLI and gateway command-dispatch tests after the initial review
   found transport parity evidence missing.
2. Restored the pre-existing gateway confirmation test's function boundary
   after a test insertion accidentally combined two independent cases.
3. Changed session-scoped receipt retrieval to fail closed when the supplied
   cwd has no canonical workspace root, with a dedicated regression test.
4. Kept the scope to a read-only control plane; generic task/event/trace work,
   terminal-write freshness, and new automatic learning are deferred.

## Worker EXPAND markers (closed)

- LEAD: CLI와 gateway의 `/goal outcomes|learning` 직접 dispatch 테스트 부재 — closed by direct transport tests.
- LEAD: shared worktree diff lacked execution proof — closed by the canonical
  focused suite recorded in `verify-goal-outcomes.md`.
