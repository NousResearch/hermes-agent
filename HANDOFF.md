# Hermes Execution Router API — Handoff

## Current objective

T001–T005 are complete. T002–T005 exist only in the uncommitted working tree and have no commit/install/profile/runtime/LIVE effect.

## Completed T001 record

1. Accepted planning/project-control snapshot commit: `a6731b60ff2408379c31fed356557c9ab0b35533`.
2. Original scaffold ancestor: `90efb60b5c10329db2843bf461027c138d6dcca7`.
3. Exact upstream merge parent: `110736c0bc9fd249f1ce7f7ca5d353040f640be6`.
4. Repository is non-shallow and preserves complete reachable ancestry.
5. Exactly three documentary conflicts were resolved: `.gitignore`, `AGENTS.md`, `README.md`.
6. Upstream README is retained byte-for-byte; both applicable AGENTS rule sets are retained; ignore rules are unioned.
7. No upstream source/runtime/test/package semantics were changed by T001.

## Completed T002 record

1. Implemented the immutable contract/canonical serialization, non-dispatching resolver, exact registration/consent lifecycle, common SessionDB lifecycle/read projection and target-free fallback signal accepted by T002.
2. Fresh focused and relevant regression suite: `263 passed, 3 skipped in 24.03s`; all three expected skips are shell-only `.recover` cases requiring the optional `sqlite3` CLI.
3. All 12 changed production modules pass `py_compile`; `git diff --check` passes.
4. Fresh `project_validate`: `PASS`, `implementation_ready=true`, `errors=[]`, `warnings=[]`, validator SHA-256 `aef7ccd9e74cc6549dff327742b3f53dfc48b2c81d91d68e3bbe00792ce74f35`.
5. HEAD remains `0b3c4e19ed457772dc7c45d413a259d0c11f1a8e`; no commit was created.

## Completed T003 record

1. Integrated accepted C6/C7 `main_turn` behavior across Classic CLI, TUI/backend, Gateway and one-shot without changing T004/T005.
2. Completion evidence: 16/16 canonical tests; 95/95 focused T002 regression; 81/81 directly affected surface regressions; pagination 10/10 non-reproduced; `py_compile` 26 files; validator PASS; `git diff --check` PASS.
3. Final independent scope review `deleg_b4631bb6`: PASS; D1 closed; Defect 0, Material risk 0, Hardening 0, Ceremony 0.
4. Frozen 28-entry manifest passed strict 28/28 entry/exit comparison; SHA-256 `aa611c677b77122fb89b6158fa7e7ea41fc9bf1f90e66da3603c955f6cf1ddfb`.
5. No commit, push, install, profile, runtime/gateway process, publication, pilot or LIVE mutation was performed.

## Completed T004 record

1. Simplified C9 v1 is complete: one initial route decision per normalized native child; automatic replacement/re-entry remains deferred to a future version.
2. Owner clarification is preserved: active pass-through uses the shared native credential bundle and native batch-wide credential/build errors; there is no per-child recovery.
3. The four actual T004 test nodes are evidence, not a product ceiling. Consolidation: T004 focused `105 passed` (plus test-runner subtests), async delegation `34 passed, 1 macOS-only skipped`, T002 `95 passed`, T003 `16 passed`; total collected `250 passed, 1 skipped`.
4. `py_compile` PASS, `git diff --check` PASS and canonical `project_validate` PASS.
5. No-router/native errors/fallback are unchanged; active lifecycle is mandatory; decision precedes credentials/build; `stop/router_error` creates no executor; all active started attempts finish once; terminal placeholders bypass child finalization; C9 fallback is terminal with no replacement.
6. Independent broad review initially found D1/D2; both were corrected. Final missing-only review `deleg_42d8cec0` PASS: Defect 0, Material risk 0.
7. Unauthorized functional expansion: NONE. Unnecessary security expansion: NONE. No executable C8 mechanics, T005, public API/version/capability, new dispatcher, queue, retry or store was added.
8. Frozen manifest v3 `/tmp/hermes-execution-router-api-002-T004-review-v3.sha256` passed strict `43/43`; SHA-256 `7532a5bfef6fe03b17fa265230c50ba76f7c4afe7dcd7b429ae68ce310b083e4`.
9. No commit, install, profile, runtime or LIVE mutation was performed.

## Completed T005 record

1. Accepted C10 SHA-256 is `f714322813b498a778059d691b2ebb2d953aa297d4a10631d8ff39c3a926ba0a`.
2. T005 implements only the minimal existing-claim design: reservation reuses `claim_lock`/`claim_expires` in unchanged `ready`/`review`; accepted routing atomically opens the canonical run through existing identity/metadata/events; existing running recovery and retry/requeue remain authoritative.
3. No routing status, task routing/source/planned-attempt columns, routing index/dashboard mapping, reservation store, queue, scheduler, dispatcher, retry engine, restart-receipt orchestrator or phantom run was added.
4. Fresh executed evidence after final corrections: full T005 focused `24 passed`; T002 `95 passed`; T003 `16 passed`; T004 `105 passed` plus `2` test-runner subtests; directly affected isolated Kanban/CLI/agent-init suites `153 passed, 1 platform skip`. These are separate reported results; no aggregate is asserted.
5. Compile, `git diff --check` and canonical project validator passed.
6. Independent whole review `deleg_870400e6` found F1–F3; F1/F2 were closed by `deleg_a6cfd2ab`; F3 was corrected; missing-only review `deleg_9706b100` passed with Defect 0, Material risk 0 and no unauthorized or unnecessary expansion.
7. Final frozen manifest `/tmp/hermes-execution-router-api-002-T005-final-review-v3.sha256`: 49 entries, strict `49/49` PASS, SHA-256 `cb838bb6ea7ac1ea6b02fa77e5d3015f2f3930c904db67bff96c958343d2a25b`.
8. T005 remains uncommitted. No commit, push, install, profile, runtime/gateway, publication, pilot or LIVE mutation was performed.

## Resume from

STOP after formal T005 closure. T006 is the next gate and requires separate exact owner authorization. Do not start T006+ or add a future-version task.

## Accepted C9 product/architecture correction

1. C9 SHA-256: `8c46a9e0083edd9dd6b8591664e62d530f1c80d6e2692c49b75cde1080db2223`.
2. V1 `native_child` remains supported and resolves once before each initial normalized delegated-child launch.
3. A router-selected child requiring route/model-change fallback terminates visibly; the same `delegate_task` unit does not construct or resubmit a replacement.
4. A later normal parent delegate call is a new execution and routes normally. No-router/pass-through native in-agent fallback is unchanged.
5. Automatic native-child replacement/re-entry remains deferred future-version work requiring a new product/architecture task and separate authorization; C8 mechanics are not current authority.

## Verification boundary

T005 verification covers only the accepted minimal C10 `kanban_worker` existing-claim integration, lifecycle/renderer behavior, reservation/open/crash recovery, routed retry/requeue and native no-router/pass-through compatibility. It does not prove or authorize T006.

## Accepted C10 T005 correction

1. C10 SHA-256: `f714322813b498a778059d691b2ebb2d953aa297d4a10631d8ff39c3a926ba0a`; owner decision: «Минимальный T005: existing claim + defer в той же lane».
2. T005 keeps task status in exact `ready`/`review`, reuses existing `claim_lock`/`claim_expires`, opens the canonical run atomically after routing, and records lifecycle in existing `task_events`/run metadata.
3. `stop/router_error` creates no credentials/run/process/executor, defers under the existing claim, and does not feed worker failure accounting or circuit breaker. Real post-start worker failure retains existing retry/requeue/failure authority and routes fresh on the next ordinary attempt.
4. No routing status, mandatory task schema/index/dashboard mapping, reservation store, queue/scheduler/dispatcher, restart orchestrator, Kanban event subsystem or security expansion is authorized.
5. T003/T004 remain complete and unchanged. T005/S001–S007 are complete; this closure grants no T006 authority.


## Not authorized

T006–T008, future-version native-child replacement/re-entry, model policy, additional commits, push, publication, installation, profile/runtime changes, consumer work, pilot and LIVE. T006 requires separate exact owner authorization.
