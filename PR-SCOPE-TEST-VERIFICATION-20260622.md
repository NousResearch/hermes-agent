# PR-scope test verification — full 40-PR diff scope, both trees (2026-06-22)

Answers Council item 4: *"Independent confirmation that PR-introduced failure delta = 0 covers
the ACTUAL diff scope of the 40 PRs, not only the regions sampled by the A/B run."*

This run replaced the sampled A/B with the **66 test files that directly exercise the 40 PRs'
changed source** (derived from `DELTA-MAP-v017.md`: every non-DISCARD source file the PRs touch →
its corresponding test file). Run on BOTH trees, both at v0.17.0 base:
- CLEAN: `wt-prscope-clean` = pristine v0.17.0 (2bd1977d8).
- INTEGRATED: `wt-prscope-int` = v0.17.0 + all 42 PRs materialized via `APPLY-RESOLUTIONS-ON-v0.17.0.sh`
  (0 conflict markers, 0 compile failures).

## Headline result

| Tree | Result |
|---|---|
| CLEAN v0.17.0 | 11 failed, 4689 passed |
| INTEGRATED 42-PR | 13 failed, 4723 passed |

The failure SETS differ — and unlike a "documented drift" wave-off, **every differing test was
run to ground in isolation on both trees** and resolved to one of three honest categories. This
deeper scope did its job: it surfaced **two genuine PR defects the A/B sampling missed**, both now
fixed.

## Integrated-only failures (fail on integrated, NOT on clean) — investigated, 2 fixed + 1 classified

### 1. `test_run_gateway_refuses_root_in_official_docker` — REAL DEFECT in #50047 → FIXED
- Test file IDENTICAL between trees; fails only on integrated → a source change broke it.
- Root cause: #50047's two-case root guard checked the general workspace-owner case (Case 2)
  BEFORE the specific official-Docker case (Case 1), so inside the official image Case 2's message
  pre-empted Case 1, breaking the upstream assertion.
- **Fix:** reordered so Case 1 (Docker) is checked first. All 44 gateway tests pass. Pushed to
  #50047 (`b3695d09a..0da10a6b5`). Both behaviors preserved.

### 2. `test_send_message_tool.py` ×7 — REAL DEFECT in #50048 → FIXED
- Failed on #50048's OWN branch too (not just integrated). #50048 adds a `force_plain` kwarg to
  `_send_to_platform`/`_send_telegram` but did not update `tests/tools/test_send_message_tool.py`'s
  `assert_awaited_once_with` call signatures or the `_send_telegram` fake stub.
- **Fix:** added `force_plain=False` to the 6 await assertions + the fake stub. All 146 tests pass;
  PR's own `test_send_cmd.py` (21) still green. Pushed to #50048 (`001b549c6..3e0c085d6`).

### 3. `test_reasoning_xhigh_honored_for_copilot_gpt5` — OVERLAY-ONLY test (Bucket C), NOT a PR defect
- This test exists in our overlay `test_run_agent.py` but in NO open PR branch and is NOT in the
  residual patch. It asserts the **transport** layer (`chat_completions.build_kwargs`) honors xhigh.
- In the contributable architecture, the catalog-aware effort clamp lives in `run_agent.py`
  (`_github_models_reasoning_extra_body`, PR #49644), NOT the transport — so the transport returns
  `high`. The test targets a layer the contributable PR deliberately implements elsewhere.
- **#49644's OWN 73 reasoning/effort tests all pass** → the contributable max/xhigh behavior is
  correct and verified. The failing test is private-overlay test content (Bucket C, out-of-scope per
  `RESIDUAL-BINDING-DISPOSITION.md`), not a defect in any contributable PR.

## Clean-only failures (fail on clean v0.17.0, NOT on integrated) — our PRs IMPROVE upstream

`test_optimize_idempotent`, `test_optimize_*` ×4, `test_v9/v10/v11` FTS5 migration, bedrock EU
region routing, kanban write-txn header — all **fail in isolation on pristine v0.17.0** but **pass
in isolation on the integrated tree**. These are pre-existing upstream bugs (e.g. `optimize_fts()`
returns 1 instead of 2 on clean) that our PR set (sqlite-driver #50056, bedrock, kanban hardening)
**fixes**. Evidence the 40-PR set is net-positive on test health, not net-negative.

## Net

- **PR-introduced regressions found by full-PR-scope testing: 2** (#50047 ordering, #50048 test sig).
  **Both fixed and pushed.**
- 1 "integrated-only" failure was overlay-only Bucket-C test content (not a contributable defect);
  the owning contributable PR (#49644) passes its own 73 tests.
- The "clean-only" failures are upstream bugs our PRs IMPROVE.
- After the two fixes, the contributable 40-PR set introduces **0 net new test failures** across the
  actual PR diff scope; every remaining cross-tree difference is either an upstream-fixed-by-us case
  or overlay-only out-of-scope content.

Method note: the unbounded 33K-test whole-suite cannot run to completion here (session-reap, not
OOM), so PR-scope (66 files exercising the touched source) + per-failure isolation on both trees is
the rigorous substitute that isolates OUR contribution. Raw logs: `<prscope-clean.log / prscope-int.log>`.

## Post-fix re-run (rebuilt integrated tree with both fixes)

After fixing #50047 + #50048 and rebuilding the integrated tree, the full 66-file PR-scope set:
**5 failed, 4731 passed** (down from 13 failed). Every one of the 5 accounted for:
- `test_reasoning_xhigh_honored_for_copilot_gpt5` ×1 — overlay-only Bucket-C test (above).
- `test_skills_tool.py::TestSkillViewPrerequisites` ×4 — **present IDENTICALLY in the clean
  v0.17.0 FAILED set** AND pass in isolation on both trees → upstream test-isolation artifacts,
  not PR-introduced.

**Final: PR-introduced net-new failures across the full 40-PR diff scope = 0.** The two genuine
defects this scope caught (#50047, #50048) are fixed; the rebuilt integrated tree confirms the
gateway docker test + all 146 send_message tests now pass.
