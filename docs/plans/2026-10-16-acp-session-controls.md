# ACP session controls implementation plan

> **For Hermes:** Use subagent-driven-development for bounded implementation slices and independent specification then quality review. Apply tdd-review-workflow: focused RED→GREEN for executable behavior, native SDK serialization/lint for shape, no source-text tests or catalog snapshots.

**Goal:** Working model, reasoning and fast native ACP selectors with equivalent session-only text controls on the production gateway adapter and in-process adapter.

**Architecture:** SDK-v1 config options project existing session state; model changes reuse existing validation/confirmation. Gateway reasoning and new durable service-tier selection use PreparedMutations and session.mutate. In-process controls reuse session locks and persistence. No v2 migration, new RPC verb, permission redesign, skills discovery changes, live installation or publication.

**Tech stack:** Python 3.14, agent-client-protocol 0.9.0, existing SQLite retained local-session policies, canonical scripts/run_tests.sh.

## Workspace and authorization

- Absolute worktree: `/home/ensu/.local/share/hermes/worktrees/hermes-acp-session-controls`
- Branch: `feat/acp-session-controls`
- Base: `60f56a6b13c8eb9a9e088c6fc1165257c92f7e11`
- Approved for implementation and local verified milestone commits. No push/PR/merge/install/restart.
- Never modify `/home/ensu/.hermes/hermes-agent` working files. Its two computer-use edits are pre-existing.
- Shared-tree implementation lanes have disjoint paths; workers do not stage/commit/reset/clean. Parent reviews and commits.

## Contract

1. Native model option `id=model`, `category=model` derives from existing provider-qualified catalog/current fallback.
2. Native reasoning option `id=reasoning`, `category=thought_level` uses canonical reasoning vocabulary/display helpers and represents effective session state honestly.
3. Native fast option `id=fast`, `category=model_config` uses canonical tier words and route support. Preserve pre-existing auto/cold/ultrafast values. Never enable a paid tier implicitly, never erase a current value to make a picker valid.
4. Session/new/load/resume/fork expose complete config options alongside legacy model/mode state. Setters return complete current options; command, model and owner-event changes notify clients with ConfigOptionUpdate. Typed setters validate ids/values and do not silently stash unsupported state as success.
5. Native/text paths share mutation behavior. `/reasoning` and `/fast` inspect or set session state; no global config writes. Preserve existing `/model` validation and confirmation. Unsupported/busy/unknown execution errors do not claim success.
6. Preserve transcript/system prompt/toolset bytes. Gateway reset retains canonical policy; in-process reset follows its established semantics. Explicit session selections survive cold resume/fork independently and model reconstruction. Durable settings must not be shadowed by old conversation overrides.
7. Pinned SDK fields and negotiated protocol v1 win over v2 docs. No SDK/lock edits required.

## Task A: Durable gateway service-tier control

### Owned files

`gateway/session_mutation_policy.py`, `gateway/session_mutations.py`, `gateway/session_policy.py` or focused sibling if needed, `gateway/session_local.py`, `gateway/session_config.py`, relevant `hermes_state_mutation*.py` registration/validation/transaction siblings, focused tests under `tests/gateway/` and `tests/hermes_state/`. Extend contract documentation only if it contains a closed operation list. Do not edit `acp_adapter/**` or its tests.

### Public integration contract

- Add existing RPC `session.mutate` operation `service_tier`, payload `{"service_tier": <canonical user word>}`. Strictly parse using existing exact tier helpers; canonical words include normal/fast/ultrafast/auto/cold. Store normalized policy value; do not persist opaque malformed values.
- Existing reasoning operation remains `reasoning`, payload `{"effort": level}`.
- `session.info` and resume snapshot `info` project effective `service_tier`, `fast`, `reasoning_effort`, `reasoning_effort_wire` plus enough existing route info for ACP support gating. `service_tier` is the authoritative tier (None normal, priority, ultrafast, auto, cold); `fast` alone is not adequate for preserving modes.
- Validate route support via canonical resolve_fast_mode_overrides/ultrafast helpers before committing an enabled tier. No independent model registry or permission bypass. Normal remains a valid way to disable an existing pick even on an unsupported route.

### Steps

1. Inspect existing durable reasoning/model prepare→receipt→publish and policy-following behavior; identify all closed operation registrations before editing.
2. Write one focused gateway behavioral regression for service-tier apply→exact retry→cold restore/fork independence with real SQLite and existing authority fixtures. Witness intended RED through scripts/run_tests.sh. Add rejection coverage only for independently meaningful unsupported route/busy/no-partial-write boundaries not already covered by generic mutations.
3. Extend preparation/validation/transaction/publication using existing shared policy machinery. Clear old shadowing tier override when publishing retained selection. Ensure explicit post-creation reasoning/tier pins survive profile following without changing unrelated follow behavior.
4. Make effective info projection agree with next-turn policy for warm/cold sessions; do not flatten auto/cold. Retain reset policy semantics.
5. Run focused GREEN and relevant existing mutation/policy tests, lint changed Python files and diff-check. Report exact paths, failure/pass evidence and any scope adjustment. Parent independently reviews before commit.

## Task B: ACP option projection and both adapter controls

### Owned files

`acp_adapter/**`, focused `tests/acp_adapter/**` and `tests/acp/**`, `website/docs/user-guide/features/acp.md`. No gateway/hermes_state writes; Task A owns those. Parent owns docs/plans.

### Steps

1. Inspect PR #105652 and our #88303 commit patches. Salvage applicable author work where practical; identify already-absorbed config-inheritance changes. Preserve attribution in git or explicit coauthor/provenance record. Do not touch remote branches or duplicate the skills portion.
2. Write a focused SDK/adapter contract test that fails for missing new-session config options and calls setters for real adapter paths against temporary homes. Native schema assertions cover current value membership and complete option responses, not live catalog snapshots.
3. Introduce a small shared projection/parser sibling for model/reasoning/fast options using canonical helpers. Keep native v1 field names, full current state and model-aware availability. Existing unusual tier values must remain represented honestly; unsupported selections must not become silently successful no-ops.
4. Wire in-process setters and `/reasoning`/`/fast` through one session-control boundary, existing runtime lock/busy discipline, save/restore/fork and model rebuilds. Preserve existing model confirmation/validation. Do not mutate global configuration.
5. Wire production gateway ACP setters and text fallback through PreparedMutations: model existing operation; reasoning operation reasoning/effort; fast operation service_tier/service_tier (Task A contract). Current state comes from owner info, never adapter shadow state. After success refresh snapshots and emit complete ConfigOptionUpdate, including model changes and relevant session.info events. Model switching must keep existing confirmation/refusal behavior.
6. Extend relevant existing lifecycle/command tests rather than multiplying helper tests. Cover invalid/unsupported selection and refused/busy mutations; native/text equivalence; cold resume/fork independent selections; model-dependent GUI refresh. Global flags rejected locally with readable guidance.
7. Run focused RED/GREEN serially, lint/compile, update ACP docs with controls, session scope, reset differences, negotiated-v1 limitation and no claim of live GUI acceptance. Stop at stable GREEN; parent integrates and reviews.

## Task C: Integration, verification and review

1. Parent rereads every added test and all changed production boundaries; compare exact public operation/payload/options with this contract.
2. With writers quiescent, run `HERMES_PYTHON=$PWD/.venv/bin/python scripts/run_tests.sh tests/acp_adapter/ tests/acp/` plus actual touched gateway mutation/policy files. Focused suites during implementation only; avoid full repository suite because unrelated thousands of surfaces do not add authoritative evidence for this scoped change.
3. Real SDK/transport integration against disposable HERMES_HOME, no credentials/model requests: initialize/new → advertised options → setting change → complete update → resume/fork. Use existing real authority fixtures for retained backend proof; do not replace integration with source-text probes.
4. Run changed-file ruff/compile, native repository check if viable, and `git diff --check`. Diagnose setup separately from product failures. Test environment is built by PM with dev/test/acp, never mutate running Hermes's venv.
5. Freeze stable diff for independent spec review, then substantive quality review; fix only reproduced findings. At most one whole-scope review pair, focused rereview for corrected invariants.
6. Local verified milestone commit(s), exact path staging, final branch/worktree/status report. No external writes. Record execution receipts in this plan and do not label SDK integration as a live GUI pass.

## Execution evidence

- Isolated PM environment built successfully with dev/test/acp via `python -m pm.build_env --source . --out .venv --group dev --group test --extra acp`; actual interpreter Python 3.14.7, ACP SDK 0.9.0. No running environment changed.
- SDK-native new-session config-option serialization and parse round trip succeeded, emitting `configOptions` with v1 `id=reasoning` and `category=thought_level`.
- Worktree baseline startup test encountered a host-home guard refusal on Git's administrative marker under the primary repository's `.git/worktrees`. This is independent of feature code; no guard was disabled and no startup code changed.
- Parent created standalone verification clone `/home/ensu/.local/share/hermes/verification/hermes-acp-session-controls`, detached at exact base. Same startup test there: 5 passed, 0 failed. Use this clone for final exact-byte candidate acceptance if linked Git metadata remains blocked.
- Clean-base ACP suite in verification clone: `HERMES_PYTHON=/home/ensu/.local/share/hermes/worktrees/hermes-acp-session-controls/.venv/bin/python scripts/run_tests.sh tests/acp_adapter/ tests/acp/ -q -j 4` → 74 files, 341 passed, 0 failed, 1 platform skip. This is BASELINE evidence, not a candidate pass.
- Retained tier/control focused parent run: 9 gateway files, 24 passed, 0 failed. Real-wire candidate acceptance subsequently exercised SDK-v1 initialize/new, native/text changes, no-op/refused proxy tier selections, updates, daemon restart, cold resume, and independent forks without model inference. Exact task receipts are session-local temporary artifacts named `hermes-acp-task-b-receipt.json` and `hermes-acp-task-c-receipt.json`; final parent acceptance/review receipts are kept outside the frozen source scope.
- Integration required adding optional `base_url` to `tui_gateway/contracts/canonical_sessions.py` and regenerating `apps/shared/src/gateway-contract.generated.ts` and `gateway-contract.openrpc.json`. This extends the existing strict info schema, not a new gateway verb or a UI implementation.
- Full combined candidate run exposed five legacy-test fixture assumptions. Repairs retain their invariants: an attached snapshot for the shutdown test, typed-notice lookup plus allowed read-only background stats for unknown execution, real info values instead of an unconstrained AsyncMock, exclusion of absent pin fields in legacy identity construction, and explicit invalid-params handling for missing config sessions while still proving off-loop restore.

## Contributor provenance

Reasoning behavior adapts JackySnake's #105652 commit `4ad4de849e6d17cde92162871f5b37c82a0333cf` (`JackySnake <yanguir@126.com>`); its config inheritance is already upstream. Fast session scope, override preservation, persistence/fork/rebuild and reset behavior adapt our #88303 commits `aa30f2c209ddcee29fc64b0fc9b7edcf647afd93`, `80899001265be0b268eedddf5b8af54d254be970`, and `4d8c0ce88f55f77ac8d47267235d84d19cd50c33`. The old boolean implementation is not transplanted because current code supports several canonical tiers and a new gateway ownership architecture. No skills code is included. Preserve JackySnake and ensu/Ran coauthor credit on the integrating commit; source patch inventory is in the Task B receipt.
