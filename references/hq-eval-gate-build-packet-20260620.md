# HQ Eval Gate Build Packet
Created: 2026-06-20
Source: latest completed Planner tick for this Build run; live worktree dirty scope verified read-only.
Status: preflight-only / deferred-code because worktree contains uncommitted tracked changes plus untracked eval-gate files already present.

## Goal
Prepare the next safe Build step for the existing harness/worktree without inserting code into the dirty tree now.

## Dirty-worktree findings
Tracked modified files:
- cron/scheduler.py
- gateway/run.py
- hermes_cli/config.py
- hermes_cli/plugins.py
- scripts/hq_harness_validator.py
- tests/gateway/test_pre_gateway_dispatch.py
- tests/scripts/test_hq_harness_validator.py

Untracked files:
- agent/eval_gate.py
- references/hq-interactive-coder-policy.md
- references/hq-runtime-budget-policy-v0.md
- scripts/hq_coder_depth_classifier.py
- tests/agent/test_eval_gate.py
- tests/cron/test_eval_gate_pre_dispatch.py

Classification: substantive-tracked
Current HEAD: 1318bf8ef
Branch: hq-harness-validator-v0

## This tick
- Completed: read-only preflight and build packet documentation only.
- Not completed: code/tests insertion, pytest reruns of heavy harness/dashboard suites, branch push, draft PR updates, Gateway/cron mutation.

## Next approvals needed
- `hq-eval-gate dry-run code insertion — 실행 승인`
- `hq-eval-gate branch push and draft PR update — 실행 승인`
- `hq-eval-gate live deployment into hermes_cli/ cron/ gateway — 별도 승인`

If approved, the follow-up Build tick should:
1. run focused syntax/tests for the eval-gate scope only,
2. add a concise commit on the existing branch,
3. rerun only that focused test subset,
4. then hand off to Review Dispatch.
