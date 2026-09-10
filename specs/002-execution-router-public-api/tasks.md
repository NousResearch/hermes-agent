# Development Task Map — Hermes Public `execution_router` Plugin API

Stage: TASK_MAP
Status: ACCEPTED
Coverage: COMPLETE
Review-State: APPROVED
Project-Version: 0.1.1
Architecture-Version: 1
Traceability-Schema: 1
Accepted-Whole-Task-Map-Source-SHA256: `bd768594a024d281e94afbf2db87cb8ec350e98e1d5858533f0c3d7713f8a0e9`
Whole-Task-Map-Acceptance: Telegram current session — owner selected «Принимаю полный task map по SHA-256»
Accepted-Architecture-Source-SHA256: `b6f800775e5835c99cd2bcb12f13f553e00b9553b0af9ce4b8345565d9022ce3`
Accepted-Architecture-Record-SHA256: `3d43b28dc773fe8efd1cd0d74246c76b87abdb5ddcbcfc0ec8cbad56beae2464`
Accepted-Task-Map-Blocks: block 1 `e88c8e79cf0e663b8ec64f4c5ea7217735ac07ef3c872537893c1b70750c1431`; block 2 `8de1bb4c3eb9363db820f5db5b0109298833485e1ced69e618b160a629933de0`; block 3 `91e7ee4d24cfd9be291c0cdf9c22a006a665217d5498089febe4833c6279cc80`
Block-Acceptance: Telegram current session — owner «Принимаю» for each exact displayed block SHA-256
Accepted-Task-Map-Corrections: `TM-C1` SHA-256 `8c4084b7eeb6fc1299e30d71fb144c1d6513632c099c581b987259ca99ae2396`
Correction-Acceptance: Telegram current session — owner «Принимаю» for exact displayed TM-C1 SHA-256
Implementation-Authorization: DISABLED

## Planning state

T001–T008 are integrated from three individually accepted blocks and correction TM-C1. Coverage is complete and the whole task map is accepted after canonical validation and independent semantic, technical, Ponytail and scope/security review. This acceptance does not permit task execution, source/database mutation, commit, remote, PR, merge, release, publication, installation, consumer change, pilot or LIVE; every task requires a separate exact authorization.

## Dependency graph

`T001 -> T002 -> {T003, T004, T005} -> T006 -> T007 -> T008`

- T003, T004 and T005 are independent after T002 and may execute only under their own exact authorization.
- T006 requires PASS completion of T002–T005 and includes separately stated local qualified-source commit authority only if T006 itself is explicitly authorized.
- T007 and T008 each require their additional external authorization/dependency conditions; task-map acceptance alone never opens them.

## T001 — Local source baseline with preserved upstream ancestry

- [ ] T001 [Sources: ARC-006] [adoption] Establish the single local implementation repository without changing Hermes behavior: first commit the accepted Feature 002 planning snapshot and aligned project metadata, then fetch exact upstream Hermes commit `110736c0bc9fd249f1ce7f7ca5d353040f640be6` without retaining a remote and create one explicit `--allow-unrelated-histories` merge whose parents preserve both project scaffold/planning history and full upstream ancestry. Resolve only the three proven add/add documentary conflicts: keep upstream `README.md` as the product README; retain both project execution rules and applicable upstream development rules in root `AGENTS.md`; union `.gitignore` without weakening either side. Record exact parent SHAs, resulting clean HEAD and absence of remotes, tags, submodules, nested repositories and source-semantic changes. Do not implement `execution_router`, push, publish, install or modify any runtime/profile.
  - [ ] S001 Dependency: none; execution requires separate explicit authorization of T001, including its two local commits.
  - [ ] S002 Acceptance: the first commit contains only the accepted planning/project-control snapshot; the second is the topology merge with exact upstream commit as a parent.
  - [ ] S003 Verification: before merge, verify the fetched object is exactly `110736c0bc9fd249f1ce7f7ca5d353040f640be6` and `git rev-parse --is-shallow-repository` returns `false`; after merge, require that exact upstream commit as a merge parent, both `git merge-base --is-ancestor 90efb60b5c10329db2843bf461027c138d6dcca7 HEAD` and `git merge-base --is-ancestor 110736c0bc9fd249f1ce7f7ca5d353040f640be6 HEAD` succeed, and `git fsck --full` plus complete reachable-parent traversal report no missing object. Record `git diff --name-status 110736c0bc9fd249f1ce7f7ca5d353040f640be6 HEAD` and machine-check every changed path against the exact allowlist derived from the accepted planning commit plus `.gitignore`, `AGENTS.md`, `README.md`; run `git diff --exit-code 110736c0bc9fd249f1ce7f7ca5d353040f640be6 HEAD -- <all upstream source/runtime/test/package paths outside that allowlist>`. Repository must be clean and have no retained remote/tag/submodule/nested `.git` authority.
  - [ ] S004 STOP: failure to obtain complete upstream ancestry, any conflict beyond the three named paths, or any ambiguous source delta stops T001 for reconciliation rather than guessing.

## T002 — Public contract, host resolver, registration, consent and common lifecycle projection

- [ ] T002 [Sources: ARC-001, ARC-002, ARC-003, ARC-004, ARC-006] After T001, implement the complete policy-neutral host core with focused tests: add immutable public contract and pure validation in `agent/execution_router.py`; add non-dispatching `resolve_execution_route(...)`, canonical request serialization/digest, bounded deadline/late-result rejection, recursion/idempotency and host capability discovery in `hermes_cli/execution_router_runtime.py`; add one generation-bound provider slot and `PluginContext.register_execution_router(provider)`; add the narrow `execution.routing` capability and exact five-field pending/consent lifecycle through existing `plugin_capabilities.py`, `plugins_cmd.py`, `plugins_loader.py` and existing config/ownership ledger; define common SessionDB route-lifecycle/idempotency schema and methods for main/native, the immutable route-event/read and renderer-isolation contract, and a common routed-attempt termination/restart signal at the existing fallback core while leaving surface re-entry/retry-budget ownership to T003–T005; Kanban consumes the same projection/signal contract but retains its transaction in T005. Do not add a model policy, credential resolver, executor, dispatcher, scheduler, retry engine, aggregate event store, consent database or new command family.
  - [ ] S001 Dependency: T001 complete; execution requires separate explicit T002 authorization and does not authorize a commit.
  - [ ] S002 Deliverable: frozen v1 request/candidate/pin/previous-attempt/decision/event/capability types, exact `route | pass_through | stop` union, contract `1.0`, three supported kinds, constants/limits and canonical UTF-8 JSON rules.
  - [ ] S003 Deliverable: generic capability grant alone produces no active router; first validated registration without exact tuple consent records only host-owned pending state; TTY `hermes plugins enable <id>` records exact consent; activation occurs only on subsequent normal load; `hermes plugins capabilities <id>` remains read-only.
  - [ ] S004 Acceptance: focused contract/runtime/plugin tests prove immutability, deterministic serialization, limit boundaries, candidate/pin/eligibility validation, exact tuple consent, non-interactive denial, stale/forged/superseded rejection, one-provider conflict, generation-safe unload, 250 ms host deadline, late-result discard, recursion rejection and idempotent application. Unsupported execution kind, incompatible contract version/schema, unknown required field or decision variant fails before provider callback, credential binding or executor creation.
  - [ ] S005 Acceptance: no-router returns native behavior without synthetic route events; active router exception/timeout/malformed/stale/conflicting result is visible fail-closed and creates no executor; resolver/read API exposes no dispatch, retry, replay, completion or event mutation authority. Per-kind host authorization and existing redaction run before instruction disclosure; only post-redaction/post-truncation UTF-8 bytes up to 16 KiB reach provider, their SHA-256 binds the exact delivered bytes, and raw text/system prompt/full history/credentials/tools/workspace remain absent.
  - [ ] S006 Verification: run the focused T002 modules only; every retained test maps to ARC-001/002/003/004/006 or a concrete consent/credential/text-disclosure/event-truth risk. Broader surface regressions remain in later tasks.
  - [ ] S007 STOP: any unresolved contract/consent/eligibility/text-disclosure/common-lifecycle defect, active callable before exact consent, non-deterministic serialization, unsupported preflight reaching callback/credentials/executor or no-router drift leaves T002 incomplete.

## T003 — `main_turn` integration

- [ ] T003 [Sources: ARC-001, ARC-002, ARC-003, ARC-004, ARC-005, ARC-006] After T002, integrate one semantic `main_turn` route across all accepted entry surfaces—Classic CLI, TUI/backend, Gateway and one-shot—using shared `prepare_main_turn_attempt` before credential binding and executor construction, surface-owned SessionDB lifecycle projection/notices, route-aware agent reuse/rebuild and existing surface re-entry for router-selected fallback. Preserve explicit pins, existing history/session authority, native pass-through/no-router behavior and existing retry budgets. Do not create a main-turn dispatcher, gateway, retry loop, credential owner or second session/event store.
  - [ ] S001 Dependency: T002 complete; execution requires separate explicit T003 authorization and does not authorize a commit.
  - [ ] S002 Classic CLI: `CLIChatTurnMixin.chat()` calls the resolver once per new user turn after credential-free normalization but before `_ensure_runtime_credentials()`, `_resolve_turn_agent_config(...)`, `_init_agent(...)` and `run_conversation(...)`.
  - [ ] S003 TUI/backend: `_prepare_turn_input` resolves before binding the session agent to turn state and before `_invoke_agent`; changed route uses the existing reconstruction path.
  - [ ] S004 Gateway: `TurnRunner.run_sync` separates credential-free candidate/pin projection from post-decision credential resolution before `_resolve_turn_agent`; one-shot resolves before provider/runtime binding and `AIAgent` construction.
  - [ ] S005 Acceptance: parameterized focused tests prove `route/pass_through/stop/router_error`, one resolver invocation per new attempt, explicit-pin priority, no credentials/executor for stop/error, exact-signature reuse, changed-signature rebuild, requested/accepted/actual events, bounded notices and permission-scoped readback on all four entry surfaces. Existing SessionDB authority persists resolution before executor construction under its existing failure policy; optional renderer failure neither changes decision nor removes authoritative event.
  - [ ] S006 Fallback: a router-selected failed attempt terminates without in-place client/model mutation; existing surface authority consumes the existing budget and creates fresh request/attempt IDs before re-entering T003. No-router or explicit pass-through preserves native fallback behavior.
  - [ ] S007 Compatibility: no-router characterization proves unchanged native kwargs, call ordering, credential/config/construction path, history behavior and absence of router events/provider callback.
  - [ ] S008 STOP: any main-turn entry surface lacking pre-credential ordering, SessionDB persistence order, renderer isolation, routed fallback re-entry or no-router characterization leaves T003 incomplete.

## T004 — `native_child` integration

- [ ] T004 [Sources: ARC-001, ARC-002, ARC-003, ARC-004, ARC-005, ARC-006] After T002, integrate one route decision per normalized native child in `tools/delegate_tool.py::delegate_task` before `_resolve_delegation_credentials(...)`, `_build_children(...)` and `_run_batch(...)`; preserve per-child explicit pins, independent batch outcomes, existing delegation construction/dispatch ownership, SessionDB lifecycle projection and existing delegated-work retry budget/re-entry. Do not route once per batch or create a child dispatcher, credential lease, executor, retry engine or second child-state store.
  - [ ] S001 Dependency: T002 complete; T003 is not required and T004 may be authorized only as its own task.
  - [ ] S002 Acceptance: focused tests prove exactly one resolver call per new child attempt, pre-credential/pre-build ordering, host-issued candidate selection, per-child pin preservation and independent `route/pass_through/stop/router_error` outcomes inside one batch.
  - [ ] S003 Acceptance: stopped/errored children create no child agent, credential resolution or executor work; valid siblings continue under existing batch semantics; notices/events remain correctly correlated and redacted. Existing SessionDB authority persists child resolution before executor construction under its existing failure policy; optional renderer failure neither changes decision nor removes authoritative event.
  - [ ] S004 Fallback: router-selected child failure closes the old attempt; existing delegation authority consumes the existing budget, issues new request/attempt IDs and invokes the router again. No-router/pass-through retains native in-agent fallback semantics.
  - [ ] S005 Compatibility: no-router characterization proves unchanged child kwargs, explicit pins, build/run ordering, batch behavior and absence of synthetic route state.
  - [ ] S006 STOP: any child path with credential/build before decision, batch-wide rather than per-child routing, missing SessionDB persistence/renderer isolation, in-place router-selected fallback or no-router drift leaves T004 incomplete.

## T005 — `kanban_worker` integration and crash recovery

- [ ] T005 [Sources: ARC-001, ARC-002, ARC-003, ARC-004, ARC-005, ARC-006] After T002, refactor the existing Kanban claim/open transaction into `ready|review -> routing` reservation followed by validated route/open, using only existing board DB/tasks/task_runs/task_events authority; add the minimum reservation owner/token/expiry/planned-attempt fields and indexes/constraints, stale recovery for both lanes, exact stop/error restoration and existing worker restart re-entry. Preserve the existing scheduler, dispatch loop, failure accounting, retry budgets and circuit breaker. Do not create a queue, scheduler, dispatcher, reservation table, router store or phantom `task_runs`.
  - [ ] S001 Dependency: T002 complete; T003/T004 are not required and T005 needs separate explicit authorization because it changes the existing Kanban schema/transactions.
  - [ ] S002 Reservation: one CAS claims eligible `ready` or `review` work into `routing`, records original lane, bounded expiry and host-issued planned attempt UUID, but creates no `task_runs`, process or executor.
  - [ ] S003 Route/open: accepted `route/pass_through` atomically creates the canonical run using the planned attempt identity, records accepted lifecycle state and moves to existing running ownership; before start, Kanban notice includes execution kind, requested/accepted route, decision, bounded reason and request/attempt IDs. `stop/router_error` emits bounded not-started state and restores the exact original lane under existing retry/backoff policy.
  - [ ] S004 Crash recovery: focused DB/dispatch tests cover crash before reservation commit, after reservation/before router result, after decision/before route/open commit and after open/before start receipt; stale `routing` recovery handles both ready/review without duplicate run or permanently undispatchable work.
  - [ ] S005 Fallback: router-selected worker failure closes the existing run/attempt; current Kanban retry/requeue authority consumes the existing budget and creates a new reservation/request/attempt before resolver re-entry. No-router/pass-through retains native worker fallback semantics.
  - [ ] S006 Acceptance: no run/process/credential binding precedes accepted route/open; no phantom or duplicate `task_runs`; events distinguish requested/accepted/actual; accepted route is atomically bound to route/open under existing persistence failure policy; optional renderer failure neither changes decision nor removes authoritative event; exact notice, stop/error/readback semantics and no-router control flow pass against real temporary Kanban DB transactions.
  - [ ] S007 STOP: any pre-route run/process, unrecovered reservation, duplicate/phantom run, missing notice/lifecycle persistence/renderer isolation, in-place router-selected fallback or no-router drift leaves T005 incomplete.

## T006 — Documentation, packaging and exact local candidate qualification

- [ ] T006 [Sources: ARC-001, ARC-002, ARC-003, ARC-004, ARC-005, ARC-006] After T002–T005, complete the generic API delivery candidate: add policy-neutral public contract/consent/discovery/event/fallback documentation and one deterministic credential-free reference fixture provider; expose the public module and required docs/fixture in wheel and sdist; run the accepted focused suites and one risk-shaped regression bundle; perform the architecture-required independent scope/security review and one consolidated absence scan. After all pre-commit checks pass and only under separately authorized T006 commit scope, create one local qualified-source commit, rerun final candidate qualification from that exact clean commit, create immutable qualification JSON plus sanitized raw logs, then create one evidence-only descendant commit containing those records and no source/package semantic change. Do not push, publish, install or modify `EXT-001`.
  - [ ] S001 Dependencies: T002, T003, T004 and T005 complete; T001 ancestry remains valid. Execution and the two exact local commits require explicit T006 authorization; earlier implementation authorization does not imply commit authority.
  - [ ] S002 Focused bundle: contract/runtime/consent/events; unsupported kind/version/schema preflight with zero callback/credentials/executor; authorization/redaction/16 KiB/digest-to-exact-delivered-text controls; renderer isolation and mandatory lifecycle completeness; all four main-turn entry surfaces; one decision per native child; Kanban notice/reservation/open/crash recovery; router-selected new-attempt fallback and native pass-through/no-router fallback.
  - [ ] S003 Regression bundle: only directly affected existing plugin loading/ledger/capabilities/CLI, strict provider selection, approvals, stream hooks, delegation, Kanban, gateway/CLI/TUI/one-shot, fallback and packaging modules. Add another existing suite only when a changed ownership file creates a named material-risk mechanism.
  - [ ] S004 Packaging: build wheel and sdist in an isolated local environment; verify public import, docs, fixture, version identity, contract discovery and exact supported/unsupported kinds from both archives. Fixture selects only host-issued `candidate_id` and performs no network/model/tool/dispatch work.
  - [ ] S005 Evidence order: build/test candidate; create qualified-source commit; rerun qualification at its exact clean HEAD; create `execution-router-candidate-qualification-v1.0-hermes-<candidate-version>-g<12-char-source-commit>.json` plus sanitized logs/hashes referencing that source commit and artifacts; create an evidence-only descendant commit; verify exact parent/source/evidence identities and clean HEAD.
  - [ ] S006 Qualification record: record exact commands/results, artifact paths/SHA-256, Python versions actually tested, separate PASS for three execution kinds, three no-router cases, consent, unsupported preflight, text disclosure, events/lifecycle/renderer isolation, fallback, regression and packaging; any required missing verdict is FAIL.
  - [ ] S007 Scope proof: run one bounded consolidated absence scan and record command/output in evidence for product policy/classifier/catalog/chains, consumer code, credentials, second dispatcher/gateway/scheduler/retry engine/queue/store/consent DB, new command family, extra execution kinds and broader capability. Independent scope/security review must PASS; no additional semantic/technical review ceremony is mandatory.
  - [ ] S008 Acceptance: every required verdict is PASS, release-blocking `known_failures` is empty, source and evidence commits are exact/clean, both scaffold and upstream base remain ancestors, artifacts bind to the source commit, evidence commit changes only approved evidence paths, and the absence scan/scope review pass.
  - [ ] S009 STOP: any missing kind/no-router/lifecycle/renderer/package/unsupported/text-disclosure proof, unresolved consent/pre-credential defect, dirty or unbound source/evidence, failed scope review/scan or artifact/hash mismatch leaves T006 incomplete; partial PASS is not a compatible API candidate.

## T007 — Separately authorized upstream delivery

- [ ] T007 [Sources: ARC-006] After T006 PASS, and only with separate explicit upstream-delivery authorization naming repository/account/branch scope, submit the exact evidence commit whose parent chain contains the exact qualified-source commit as one reviewable upstream changeset preserving upstream ancestry. Create only the authorized branch/remotes/topology needed for delivery, push only that exact evidence head, open or update one scoped PR/change request, link the committed qualification record/log hashes, and read back repository/base/head SHA, diff scope, CI and review state from the external authority. Mutable PR text is not evidence authority. Do not claim merge/release, rewrite unrelated history, include Feature 002 policy, publish packages, install Hermes or activate any plugin.
  - [ ] S001 Dependencies: T006 PASS and exact owner authorization for repository, account/fork, branch, remotes, push and PR; absence of credentials, target fork or permission blocks T007 rather than selecting another destination.
  - [ ] S002 Acceptance: external readback proves intended upstream repository, authorized fork/account, base branch, exact evidence-head SHA and scoped diff; parent chain contains exact qualified-source commit; PR links evidence committed at that head and contains no consumer policy, local profile/runtime artifact or secret.
  - [ ] S003 Verification: record PR/change-request URL/ID, base/head/source/evidence commits, remote refs, required CI checks and reviewer disposition from GitHub/upstream; local command success or mutable PR prose alone is insufficient.
  - [ ] S004 STOP: missing authorized destination/credentials, head/evidence mismatch, failed external readback, secret/out-of-scope diff or non-PASS required CI/review leaves T007 incomplete.
  - [ ] S005 Boundary: PR creation, CI PASS or reviewer approval does not mark T008 complete and does not authorize merge, release, installation, consumer integration, pilot or LIVE.

## T008 — Upstream merge/release and release-specific qualification readback

- [ ] T008 [Sources: ARC-006] After T007 and an actual published upstream release, reconcile the merged source and published wheel/sdist against T006 and create a separate immutable release-qualification record. If the owner is explicitly authorized to perform merge/release, those mutations require their own named approval; otherwise wait for maintainer action. Fetch/read artifacts and execute the accepted focused, three-kind, no-router, fallback, consent/event and packaging qualification in an ephemeral isolated environment that is not any working Hermes profile/runtime. Bind actual merged commit, release version, artifact hashes, commands/results and relation to the T006 candidate; mark the API project version released only after exact external product readback. Do not install the build into a real runtime or resume `EXT-001` automatically.
  - [ ] S001 Dependencies: T007 delivered, actual upstream merge, published release artifacts and any separately required merge/release authorization.
  - [ ] S002 Release qualification: run the release-specific bundle against exact released source/wheel/sdist in an ephemeral isolated environment; record separate PASS for `main_turn`, `native_child`, `kanban_worker`, all three no-router cases, unsupported preflight, text disclosure, consent, lifecycle/renderer isolation, fallback, regressions and package discovery.
  - [ ] S003 Provenance: carry forward a T006 result only when scoped source bytes and contract fixtures are hash-identical; release-specific version/package/discovery and every changed byte are rechecked. Squash/rebase/version or artifact changes without proven equivalence require affected/full rerun rather than identity claims.
  - [ ] S004 Evidence: create `execution-router-release-qualification-v1.0-hermes-<release-version>-g<12-char-merged-commit>.json` plus sanitized log/artifact hashes. Commit/publish this record only under authority available for T008; otherwise retain it locally and leave project RELEASED state open.
  - [ ] S005 Delivery record: release notes/readback state contract `1.0`, supported and unsupported kinds, exact consent requirement, fail-closed active-router errors, immutable-route/new-attempt fallback, event truth and no-router compatibility.
  - [ ] S006 Handoff: provide `EXT-001` immutable release coordinate, source/artifact hashes, exact `requires_hermes` floor, fixture hash, candidate/release qualification records, event/read schema, consent acquisition flow and three-kind matrix. `EXT-001` keeps T005 `DEFER` until its own separately authorized compatibility qualification passes.
  - [ ] S007 STOP: merge without a published exact release, release without matching artifacts, unproven candidate-to-release equivalence, failed isolated release qualification or missing per-kind/no-router evidence leaves T008 open. API release does not authorize real installation, consumer changes, pilot or LIVE.

## Ponytail task-map pass

### Retained tasks and necessity

- T001 is separate because one repository must preserve both accepted planning history and complete upstream ancestry before semantic source edits; its two local commits require explicit authority.
- T002 merges contract, validation, resolver, registry, consent and common lifecycle/read projection into one host-core admission boundary; splitting these would create artificial intermediate contracts.
- T003, T004 and T005 remain separate because the three required execution kinds have different entry points, state/transaction owners, fallback re-entry and independent compatibility verdicts. Four main-turn frontends remain one T003.
- T006 merges docs, fixture, packaging, focused/risk-shaped regression tests, consolidated absence scan, scope/security review, hashing and candidate qualification into one exact local release-candidate/evidence boundary.
- T007 is separate because remote/push/PR mutations require independent external authorization and external readback.
- T008 is separate because merge and published release are external product outcomes not proven by local PASS, commit, PR or CI.

### Removed work

No separate task is retained for individual enums/constants/validators, consent service/store, event database, frontend-specific main-turn work, fallback engine, Kanban migration/reservation/recovery fragments, exhaustive unrelated test matrices, local ceremony tags/releases, automatic installation, consumer implementation, pilot or LIVE.

Rejected additions include a second store, dispatcher, gateway, scheduler, retry engine, queue, plugin event-bus decision authority, new CLI command family, model policy/classifier/catalog/chains, broader capability, auxiliary/cron/`ctx.llm`/aggregation/provider-retry routing, and manifest-schema repair unless a proven accepted-path blocker requires a separately reviewed correction.

## Authorization and external boundaries

- Whole task-map acceptance authorizes no task execution.
- Each T001–T006 requires separate exact task authorization; wording inside one task does not authorize another.
- T001 explicitly includes only its two local topology/planning commits. T002–T005 explicitly do not authorize commits. T006 may create one qualified-source commit and one evidence-only descendant commit only when T006 itself is authorized and all pre-commit gates pass.
- T007 requires separate authorization naming upstream repository, account/fork and branch/remote/push/PR scope.
- T008 requires actual upstream merge/release and any separately necessary merge/release authority; otherwise it is read-only reconciliation.
- Installation of Hermes or any plugin, changes in `hermes-multimodel-routing` (`EXT-001`), return to its T005, pilot and LIVE are outside this API map and require separate project/operational authority.

## Whole-map acceptance criteria

The map may be accepted only if independent review confirms:

1. every ARC-001–ARC-006 has dependency-ordered implementation and qualification ownership;
2. every T001–T008 cites accepted architecture and has measurable deliverable/acceptance/STOP conditions;
3. T001 topology is feasible without nested repositories, retained remotes or source-semantic drift;
4. all three required execution kinds receive separate complete compatibility verdicts;
5. no-router, consent, event truth, fallback/new-attempt, packaging and exact-release evidence are covered;
6. Ponytail removal list does not omit accepted behavior;
7. no unapproved product, operational/control-plane or unnecessary security expansion is embedded;
8. implementation, commit, external delivery, release, installation, consumer, pilot and LIVE gates remain distinct.
