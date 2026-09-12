# Hermes Execution Router API — Status

Status: PLANNING_ACCEPTED / T001_COMPLETE / T002_COMPLETE / T003_COMPLETE / T004_COMPLETE / T005_COMPLETE
Updated: 2026-09-12
Project-Version: 0.1.1
Active-Feature: 002-execution-router-public-api
Implementation-Authorization: T005_COMPLETED

## Proven

- Product spec accepted baseline SHA-256 before C9 was `af4d8b1b9916c529d2eb9ac9be5a73ac5f8f0e149c297f76e62454b9a8758eac`; C9 is now integrated into current canonical authority.
- Integrated architecture is ACCEPTED; accepted source SHA-256 `b6f800775e5835c99cd2bcb12f13f553e00b9553b0af9ce4b8345565d9022ce3`.
- Complete task map T001–T008 is ACCEPTED; accepted source SHA-256 `bd768594a024d281e94afbf2db87cb8ec350e98e1d5858533f0c3d7713f8a0e9`.
- Project charter is accepted; SHA-256 `85e439c38b3330316ca84a66df5fb57759af8e3dde6f53761cee9d8569c27f55`.
- Project validator passes with `version_state=ACCEPTED` and `implementation_ready=true`.
- Portfolio registry marks this project `adopted` with active feature `002-execution-router-public-api`.
- Owner explicitly authorized T001 including its two local commits.
- T001 preserved the accepted planning history and complete upstream Hermes ancestry in one local repository.
- Exact planning parent is `a6731b60ff2408379c31fed356557c9ab0b35533`; exact upstream parent is `110736c0bc9fd249f1ce7f7ca5d353040f640be6`.
- The topology merge resolved exactly `.gitignore`, `AGENTS.md` and `README.md`: upstream README was retained, both applicable AGENTS rule sets were preserved, and ignore rules were unioned.
- T002 policy-neutral host core, T003 `main_turn`, T004 `native_child` and T005 minimal C10 `kanban_worker` integration are complete in the uncommitted working tree.
- T003 integrates accepted architecture corrections C6 and C7 across Classic CLI, TUI/backend, Gateway and one-shot.
- T003 evidence: 16/16 canonical tests; 95/95 focused T002 regression; 81/81 directly affected surface regressions; pagination 10/10 non-reproduced; `py_compile` 26 files; validator PASS; `git diff --check` PASS.
- Final independent scope review `deleg_b4631bb6`: PASS; D1 closed; Defect 0, Material risk 0, Hardening 0, Ceremony 0.
- Frozen 28-entry source/test/control manifest passed strict 28/28 entry/exit comparison; SHA-256 `aa611c677b77122fb89b6158fa7e7ea41fc9bf1f90e66da3603c955f6cf1ddfb`.
- T003 remains uncommitted; no commit, push, install, profile, runtime/gateway process, publication, pilot or LIVE mutation was performed.
- Owner correction C9 is ACCEPTED; SHA-256 `8c46a9e0083edd9dd6b8591664e62d530f1c80d6e2692c49b75cde1080db2223`. C8 is `SUPERSEDED / NOT_EXECUTABLE`.
- T004 implements simplified C9 v1 only: one initial route decision per normalized native child; automatic replacement/re-entry is deferred to a future version.
- Owner clarification preserves the active pass-through shared native credential bundle and native batch-wide credential/build errors; no per-child recovery was added.
- The four actual T004 test nodes are evidence, not a product ceiling. Consolidation results: T004 focused `105 passed` (plus test-runner subtests), async delegation `34 passed, 1 macOS-only skipped`, T002 `95 passed`, T003 `16 passed`; total collected `250 passed, 1 skipped`.
- `py_compile` PASS, `git diff --check` PASS and canonical `project_validate` PASS.
- Verified behavior: no-router/native errors/fallback are unchanged; active lifecycle is mandatory; decision precedes credentials/build; `stop/router_error` creates no executor; every active started attempt finishes once; terminal placeholders bypass child finalization; C9 routed fallback is terminal with no replacement.
- Independent broad review initially found D1/D2; both were corrected. Final missing-only review `deleg_42d8cec0` PASS: Defect 0, Material risk 0.
- Unauthorized functional expansion: NONE. Unnecessary security expansion: NONE. No executable C8 mechanics, T005, public API/version/capability, new dispatcher, queue, retry or store was added.
- Frozen implementation manifest v3 `/tmp/hermes-execution-router-api-002-T004-review-v3.sha256` passed strict `43/43`; manifest SHA-256 `7532a5bfef6fe03b17fa265230c50ba76f7c4afe7dcd7b429ae68ce310b083e4`.
- No commit, install, profile, runtime or LIVE mutation was performed.
- Owner correction C10 is ACCEPTED; SHA-256 `f714322813b498a778059d691b2ebb2d953aa297d4a10631d8ff39c3a926ba0a`. T005 implements only existing claim plus bounded defer in the unchanged `ready`/`review` lane, atomically opens the canonical run after routing, and reuses existing events, running recovery and retry/requeue. No new routing status, schema, dispatcher, queue, scheduler, retry engine, reservation store or restart orchestrator was added.
- Fresh executed evidence after final corrections: full T005 focused `24 passed`; T002 `95 passed`; T003 `16 passed`; T004 `105 passed` plus `2` test-runner subtests; directly affected isolated Kanban/CLI/agent-init suites `153 passed, 1 platform skip`. These are separate reported results; no aggregate is asserted.
- Compile, `git diff --check` and canonical project validator passed.
- Independent whole review `deleg_870400e6` found F1–F3. F1/F2 were closed by `deleg_a6cfd2ab`; F3 was corrected, and missing-only review `deleg_9706b100` passed with Defect 0, Material risk 0 and no unauthorized or unnecessary expansion.
- Final frozen manifest `/tmp/hermes-execution-router-api-002-T005-final-review-v3.sha256` has 49 entries, SHA-256 `cb838bb6ea7ac1ea6b02fa77e5d3015f2f3930c904db67bff96c958343d2a25b`, and passed strict `49/49` verification before formal closure.
- T005 remains uncommitted; no commit, push, install, profile, runtime/gateway, publication, pilot or LIVE mutation was performed.


## Current state

T001–T005 are complete. STOP before T006. T006 remains a separate authorization gate.

## Not yet authorized or proven

- T006–T008 execution, commit, delivery, release, installation, consumer integration, pilot or LIVE.

## Boundaries

T005 closure covers only the accepted minimal C10 `kanban_worker` existing-claim integration and its frozen evidence. It does not authorize T006+, commit, future-version replacement/re-entry, runtime/profile, consumer, pilot or LIVE work.
