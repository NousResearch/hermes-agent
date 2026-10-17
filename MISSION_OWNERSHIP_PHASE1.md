# Mission ownership enforcement: Phase 1 candidate

## Scope and evidence

This candidate is based on installed Hermes `v0.22.2` (`d923a5eb48`) in a separate worktree. It does not change the running Hermes process, gateway, profile configuration, dispatch settings, mission board, or the canonical X checkout. The X checkout and the October 16 audit were inspected read-only. PR #59 is an open draft in `FTr3d/X` and records the intended cross-surface ownership model.

The candidate adds a durable mission record beside Kanban, mandatory turn admission for Fred, a mandatory execution fence for registry and inline tools, explicit progress/update/handoff/recovery/completion operations, a fast Telegram admission hook backed by the same store, and a guard against completing a mission's root Kanban card with a prose-only summary. Root missions are not Kanban tasks, so the dispatcher cannot claim them. Child specialist tasks remain separate and do not transfer root ownership.

## Focused verification

Run in this checkout with the project dependencies installed:

```sh
PYTHONPATH=. pytest -q --confcutdir=mission_ownership_tests mission_ownership_tests/test_mission_ownership.py
python -m compileall -q hermes_cli/mission_ownership.py agent/turn_context.py agent/tool_executor.py hermes_cli/kanban_db.py model_tools.py tools/mission_tools.py plugins/mission_admission
git diff --check
```

The focused suite currently contains 22 tests. It exercises duplicate admission, single-root ownership, cross-session no-takeover, correction and unrelated follow-up handling, atomic handoff and rollback, stale-owner fencing (including inline `delegate_task`), dispatcher task state, dead-worker recovery, and evidence requirements. Tests use temporary SQLite databases. No live gateway or browser UI was exercised, and no live service was started or restarted.

## Enforcement limits that remain

- A tool that has already started before a handoff can continue running. The fence checks authority immediately before each new tool dispatch; it does not drain or cancel an in-flight shell, browser action, subprocess, or code sandbox at transfer time. A strict no-overlap guarantee needs an execution lease/drain protocol and process cancellation integration.
- The completion receipt validates the current contract version, accepted-criteria list, code artifact hash, and a successful check receipt. The caller supplies the check command and exit status; Hermes does not independently rerun it, and the receipt is not a semantic review. It blocks the audited summary-only closure path but cannot prove the work is correct.
- Admission stores the full original request. Initial acceptance has a broad completeness criterion; the model or user must refine it with `mission_update` when the request needs measurable criteria. Correction classification is lexical and can need explicit `mission_update` for ambiguous follow-ups.
- The admission key derives from normalized objective text because the common turn-context interface does not expose a stable transport delivery ID. Repeated identical substantial wording is treated as the same mission, including across sessions; a new session is denied automatic takeover. Reusing identical wording after completion is not yet modeled as a new mission instance.
- Existing plugin hooks and in-process code are not a security boundary against arbitrary SQL writes or code that bypasses Hermes tool dispatch. This is application-level coordination enforcement, not OS isolation.
- Recovery checks native task-run state and refuses takeover while a worker is alive or still launching. The recovery tests simulate worker liveness; validate the deployed platform's PID/start-time behavior before relying on automatic recovery.
- The existing installed profile has a legacy Telegram-only `mission-admission` plugin and profile-local dispatch configuration. A safe rollout must replace/disable that legacy plugin version and verify the effective host-level dispatcher policy; a Fred-local `dispatch_in_gateway: false` value alone did not prove that the multiplexed gateway could not dispatch a Fred task.

## Rollout gates

Do not deploy this worktree directly. Before staging it:

1. Review the candidate against the installed Hermes version and pin the exact source commit.
2. Add versioned migration/backup behavior for an existing Kanban board and test it on a copied production database.
3. Integrate in-flight tool leases or explicitly accept the overlap limitation.
4. Define measurable acceptance criteria for the X source-capture request, then test mission completion against them.
5. Run the full Hermes test suite plus Telegram, CLI, desktop/web, dispatcher, restart, and recovery integration tests in an isolated profile/gateway.
6. Replace the old profile plugin with the candidate version, then verify effective gateway-level dispatch settings and all other profiles sharing that gateway.
7. Back up the board, roll out to a non-live test profile, observe duplicate admission/handoff/worker recovery, and only then schedule a separately approved live rollout.

## Rollback

For a test rollout, stop the candidate gateway, restore the profile's prior plugin directory/configuration and prior Hermes build, and restore the pre-migration copy of the Kanban DB if the schema migration was applied. The mission tables are additive and may be left in place only after confirming the old build tolerates unknown tables. Do not delete mission rows during rollback; preserve the board copy and event history for investigation. This candidate itself has not been deployed, so no runtime rollback has been performed.

## Expected overhead

No extra model call is introduced. Fred turns add one small SQLite admission/read path plus a compact mission summary in turn context; tool calls add one SQLite ownership read immediately before dispatch. Mission writes occur on admission, user steering, progress, handoff, recovery, and completion. The current context summary may grow with accumulated steering entries, so long-running missions should summarize older updates without discarding the original request/history. The local focused suite completes in roughly 2.5 seconds on this host; that is test runtime, not a production latency measurement. Production overhead remains unmeasured and should be sampled before rollout.

## Local candidate state

Branch: `codex/mission-ownership-phase1`

Worktree: this directory. The candidate is not committed or pushed yet. The configured Git remote is `NousResearch/hermes-agent`; no `FTr3d/hermes-agent` fork was found during the initial read-only repository lookup.
