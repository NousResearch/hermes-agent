# STATUS — rebuilt-restart-fallback-depth

## Phase 01 — Independent ceiling for the rebuilt-message (fallback-hop) restart
Status: READY FOR BUILDER (not yet implemented in this worktree)

Gate verified in both directions by the controller before dispatch:
- Direction 1 (fails against nothing implemented): CONFIRMED — `GATE FAIL: new test
  test_rebuilt_restart_ceiling_scales_with_fallback_chain_depth not found`, 20/20 existing
  tests still pass unmodified.
- Direction 2 (passes against the real implementation): CONFIRMED — controller applied the
  phase's exact contract as a throwaway stub, ran the gate (`GATE PASS`, 21/21 passed
  including the new test), then reverted all three implementation files back to pristine
  `origin/main` before handing off. The builder is doing real, fresh work — the stub was
  verification-only and is not present in this tree.

Builder: implement phase-01 exactly as specified in
`phases/phase-01-independent-rebuilt-restart-ceiling.md`, run
`gates/phase-01.sh`, and report back here with the gate's real output.
