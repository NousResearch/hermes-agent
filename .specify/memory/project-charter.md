# Project Charter: Hermes Execution Router API

Project-ID: hermes-execution-router-api
Project-Class: full
Status: active-planning
Standard-Version: 2.0.0
Business-Outcome: Maintain a versioned, generic, documented and qualified public Hermes `execution_router` plugin API, with exact source/release identity and evidence, without owning consumer routing policy or runtime rollout.
Owner: internal
Sources-of-Truth: `.specify/memory/constitution.md`; `.specify/memory/project-charter.md`; accepted `specs/<feature>/spec.md`, `plan.md`, `tasks.md`; `status/STATUS.md`; `HANDOFF.md`; feature/project `evidence/`
Live-Targets: none
Protected-Systems: local project Git repository; upstream Hermes repository and release/package authorities; installed Hermes runtimes, profiles and gateways; consumer project `hermes-multimodel-routing`
Knowledge-Scope: Internal
Evidence-Requirements: exact accepted artifact SHA-256 values; exact source/base/release commits and versions; focused and affected-regression commands/results; wheel/sdist/fixture/evidence hashes; per-kind and no-router verdicts; external readback for remote mutations or release claims
Rollback-Policy: Use identity-bound Git reversal or upstream-native rollback for an explicitly authorized target; never rewrite accepted/released history, silently replace runtime artifacts or infer rollback success without exact readback.

## Outcome

This project maintains the generic host-owned `execution_router` contract and its minimum Hermes integration, documentation, policy-neutral fixture, qualification and upstream release evidence. A positive project outcome requires an exact published Hermes release containing the qualified API; planning, local source, commits, tests, PRs, CI or merge alone are supporting state, not the released product result.

## Scope

### In scope

- accepted versioned product, architecture and task-map artifacts;
- generic public immutable contract, host resolver, validation, consent/registration and lifecycle/read projection;
- exactly the execution kinds accepted by the active feature;
- thin integration into existing Hermes-owned execution surfaces;
- policy-neutral documentation/reference fixtures, packaging, local candidate qualification and release-specific qualification;
- local Git ancestry/version evidence and separately authorized upstream delivery/release reconciliation.

### Out of scope

- consumer model policy, classifier, model catalog/tiers/chains, localized product policy or consumer rollout;
- a second dispatcher, gateway, scheduler, retry engine, queue, task/event/consent store or control plane;
- credential ownership, permission/budget/completion authority or arbitrary provider/model selection outside host-issued candidates;
- installed Hermes/profile/gateway mutation, plugin installation/enablement, consumer project changes, pilot or LIVE unless separately authorized under their own exact authority;
- auxiliary/cron/`ctx.llm`/aggregation/provider-internal-retry routing unless a later accepted product version adds it.

## Ownership

- Owner: internal
- Operator: an explicitly authorized Hermes project agent operating through Project Standard and the accepted canonical task map
- Default profiles: none; this project does not own or automatically synchronize Hermes profiles
- Consumer boundary: `hermes-multimodel-routing` owns product-specific routing policy, model qualification, consumer compatibility, installation, pilot and LIVE decisions
- Host boundary: Hermes remains owner of credentials, eligibility, permissions, budgets, dispatch, retry budgets, lifecycle truth and completion

## Sources of truth

- Project governance: `.specify/memory/constitution.md` and `.specify/memory/project-charter.md`
- Versioned feature intent: accepted same-feature `spec.md`
- Versioned architecture: accepted same-feature `plan.md`
- Sole execution route: accepted same-feature `tasks.md`, mirrored but not replaced by Hermes todo
- Current state: `status/STATUS.md`
- Continuation: `HANDOFF.md`
- Verification: feature/project `evidence/`, exact local Git objects and authoritative external readback

No status, handoff, todo item, review statement, passing test, local commit, PR or CI result may override an accepted canonical artifact or substitute for release/product readback.

## Systems and side effects

- Local project repository: mutable only through exact project/task authorization; commits only when the task explicitly includes them.
- Upstream Hermes/GitHub and release/package authorities: read-only by default; branch, remote, push, PR, merge, tag, release or publication requires separately named authorization and post-write readback.
- Installed Hermes runtimes, profiles and gateways: outside this project's default mutation authority.
- Consumer project and plugin: separate registered-project authority; API release evidence permits compatibility work only after its own task authorization.
- Secrets: remain only in existing protected stores and never enter source, planning, logs, fixtures, evidence, Git or chat.
- LIVE targets: none.
- Rollback: establish exact baseline/identity before any authorized mutation, use the owning system's bounded reversal, and verify the exact target afterward. Unknown side-effect identity requires reconciliation before retry.

## Evidence contract

The strongest proof path is:

1. accepted canonical spec/plan/tasks hashes and exact project version;
2. exact upstream base and qualified source/evidence commit ancestry;
3. focused tests plus only affected risk-shaped regressions;
4. three separate execution-kind and no-router verdicts;
5. consent, pre-credential ordering, text disclosure, lifecycle truth, fallback/new-attempt, crash recovery and packaging evidence;
6. hashed wheel/sdist/fixture/qualification records and sanitized logs;
7. external repository/PR/release/package readback for every external claim;
8. release-specific isolated qualification before declaring the project version released.

Missing identity, missing per-kind proof, unresolved material defect, secret exposure, scope expansion or absent external readback is fail-closed and cannot be reported as DONE or RELEASED.

## Authorization boundary

Acceptance of this charter and the canonical task map does not authorize any task execution. Source changes, tests, local commits, remotes, push, PR, merge, release, publication, artifact execution, installation, consumer changes, pilot and LIVE remain separate exact gates.
