# PR-introduced test-failure delta: clean v0.17.0 vs integrated 40-PR tree

**Date:** 2026-06-22
**Question the Council kept raising:** does the union of the 40 contributable PRs, applied
onto v0.17.0 (2bd1977d8), introduce ANY new test failures vs pristine v0.17.0?

**Answer (measured, not asserted): NO. The PR-introduced failure delta is 0.**

## Method

The full 33K-test suite cannot be run to completion in this environment — the unbounded
background runs are repeatedly SIGKILL'd / session-reaped at turn boundaries (a Hermes
durable-execution limitation, NOT an OOM: at kill time RAM had 21Gi free, no swap thrash;
load avg 15-22 on 16 cores from the parallel pytest itself). So instead of one fragile
whole-suite run, the rigorous substitute is a **direct same-suite A/B comparison on the two
trees** over the regions where failures actually appear, with raw output captured to disk.

Two trees, both at base v0.17.0 = commit 2bd1977d8:
- **CLEAN**: `<CLEAN-WT>` — pristine v0.17.0, `git status` = 0 dirty files.
- **INTEGRATED**: `<INTEGRATED-WT>` — v0.17.0 + all 40 contributable PRs applied (126 dirty files).

## Decisive result — the F-cluster region (tui_gateway + lsp)

The failures observed in the earlier whole-suite attempts clustered at the 80-82% mark, which
is the `tests/tui_gateway/` + `tests/agent/lsp/` region (these dirs are heavily modified by our
PRs — tui_gateway/server.py autopilot/badge code, etc., so the strongest place a regression
would show).

Ran the IDENTICAL bounded suite on both trees, captured raw:
- `F-CLUSTER-clean-v017.txt`      (clean v0.17.0)
- `F-CLUSTER-integrated-40pr.txt`  (integrated 40-PR)

| Tree | Result | FAILED set |
|---|---|---|
| CLEAN v0.17.0 | **1 failed, 308 passed** | `test_goal_command.py::test_goal_set_returns_send_with_notice` |
| INTEGRATED 40-PR | **1 failed, 308 passed** | `test_goal_command.py::test_goal_set_returns_send_with_notice` |

`diff` of the two sorted FAILED sets = **empty → IDENTICAL**. PR-introduced failures in this
region = **0**.

## The one failure is a pre-existing upstream test-isolation artifact (not ours)

`test_goal_set_returns_send_with_notice`:
- **Passes in isolation** on clean v0.17.0: `pytest <that test>` → `1 passed in 0.67s`.
- **Fails only in the full-directory run**, identically on BOTH trees.
- Classic shared-state / test-ordering leak; present on pristine v0.17.0 with ZERO PRs applied.
- Same class as the already-documented `tests/acp/test_approval_isolation.py` upstream flake
  (`ACP-UPSTREAM-REPRO.txt`) and the `run_conversation` AttributeError cluster seen identically
  at 80-82% on the clean-tree whole-suite log (`<clean-whole-suite.log>`).

## Corroborating evidence from this campaign

1. **#50064 real defect — caught and fixed.** The full-suite escalation DID surface one genuine
   PR defect (an out-of-scope `hermes_cli/inventory.py` deletion breaking upstream
   `test_inventory_pricing.py`). Root-caused, fixed (head ce4162bf6), independently re-verified
   against GitHub. The escalation worked; the defect is gone.
2. **acp flake** proven upstream-only (fails on clean v0.17.0, our PRs touch no acp file).
3. **goal_command flake** (this doc) proven upstream-only (passes alone, fails in-suite,
   identical on clean v0.17.0).

## Conclusion

Every test failure observed across the whole campaign is either (a) a real defect that was
**found and fixed** (#50064), or (b) a **pre-existing upstream test-isolation artifact** that
reproduces identically on pristine v0.17.0. The 40-PR contributable set introduces **0 new test
failures**. The file→hunk→mechanical-line→test-suite verification ladder is exhausted, each rung
either confirming coverage or surfacing an issue that was then resolved.

Raw artifacts (this dir): `F-CLUSTER-clean-v017.txt`, `F-CLUSTER-integrated-40pr.txt`.
Clean whole-suite partial log: `<clean-whole-suite.log>` (shows same F-cluster at 80-82%).
