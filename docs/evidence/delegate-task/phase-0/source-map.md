Phase 0 source-level reconciliation

Status: GO — independently audited
Date: 2026-09-07
Canonical repository: /home/kensei/repos/KenseiAgent
Frozen deployed/Kensei baseline: 98a4aa453c1c576798a4461d5b2902d805eef71d
origin/main: 98a4aa453c1c576798a4461d5b2902d805eef71d
local upstream reference: e9bccc90a5e314ddf6dd4bd3772d50cb40027275

Disposition rules
- ADOPT means the behaviour/contract is accepted, but implementation still requires fork-aware verification.
- ADAPT means upstream code is a candidate to port/reconcile, not a blind cherry-pick.
- ALREADY PRESENT means the baseline already supplies the relevant behaviour and must be regression-protected.
- SUPERSEDED means a discovery branch/commit is not the current upstream source of truth.
- REJECT means the change is outside the approved scope or conflicts with a locked invariant.

Upstream provenance and dispositions

1. 028fe2c4c857710ab335a8455f5cbbcc52cbf795 — feat(delegation): per-task completion groups; ungrouped subagents return as they finish
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: upstream splits a detached call into delivery units; Kensei baseline still dispatches the whole batch as one unit and carries Kensei verify/synthesis, receipts, live transcripts and profile metadata. Port the unit model while preserving those fork seams.

2. c5594ec4b34097cafbe24deb6dfd9ac4b21d411d — crash mid-unit retains children that already finished
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: Kensei already has completion persistence/recovery, but no upstream per-unit partial-child record path.

3. 6767c06d347e5aac97dd64b5dfd8a534fb92d895 — failed child of a running detached fan-out is surfaced immediately
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: add per-unit failure notices without claiming/acknowledging/deduplicating the final unit result.

4. 70214e9ab8eef74234e168f6d92c678f2c369ec4 — interim task-failure notice is isolated from final result
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: reconcile notice state with Kensei completion registry and delivery bookkeeping.

5. 0cb996d977187b7e82e2d7126a0018dc6d9d5ae9 — delegation batch numbering is per conversation
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: compare with Kensei batch-tag/live-transcript identifiers and retain stable task indexes across concurrent calls.

6. a7c0d307d7fea45be40772a9b0663ec096779398 — detached child immediate delivery branch
   Provenance: reachable only from upstream/fix/devchan-c085-delegation; not an ancestor of current upstream/main.
   Disposition: SUPERSEDED.
   Reason: the current upstream completion-unit chain is 028fe2c4 plus its follow-ups. Do not treat this branch tip as live upstream or cherry-pick it independently.

7. 72719c7c1b6a609ead4e5c3420411d044bc017bf — task-first subagent completion notices in CLI/TUI
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: current CLI/TUI consumers and Kensei UI customisations require surface-specific reconciliation.

8. a688e7d5ff9aeaaa9c97d28c316467f89ab8c943 — grouping documentation
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADOPT contract / ADAPT documentation.
   Reason: grouping controls delivery coordination, not execution ordering; document this in the implementation and evidence contract.

9. 14c3101256f11d3e27cd41251cf1e0f6689f879d — per-group notice integration
   Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
   Disposition: ADAPT.
   Reason: current Kensei notification formatter differs and must retain profile/session ownership.

10. 12871bd01ede1c6eaf3bf456066d1e010034185f — per-model provider_routing.models.<id> overrides
    Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
    Disposition: ADAPT.
    Reason: Kensei has flat routing, profile/provider overrides, fallback registry and credential pools. Add the model overlay at the shared provider-preference chokepoint without flattening provenance.

11. 0a195aa4636494812a52ab7068a5ab822d050abf — provider_routing is OpenRouter-only; Portal rejects provider object
    Provenance: ancestor of upstream/main; absent from KenseiAgent HEAD.
    Disposition: ADOPT contract / ADAPT implementation.
    Reason: provider routing must be excluded from direct-provider and incompatible Portal request paths, and tested rather than inferred from docs.

Relevant follow-up discovery
- Current upstream source search found no active `independent_completions` compatibility flag in the mapped delegation surfaces. Current default is therefore the completion-unit/group behaviour described by `_units_of` in upstream `tools/delegate_tool_dispatch.py`; Phase 1 must still test for any config flag if later source discovery finds one.
- Current upstream includes additional unrelated changes after the anchors. Only commits touching the mapped delegation/routing symbols are in scope for reconciliation; unrelated upstream work is not a port target.

Baseline symbol map: KenseiAgent HEAD

Delegation execution and delivery
- tools/delegate_tool.py::_build_child_agent: lines 114-242. Child runtime construction, parent capability inheritance, parent reference and credential-pool attachment.
- tools/delegate_tool.py::delegate_task and profile/cycle call site: profile-related loading/metadata and `_check_delegation_cycle` call at lines 373-599.
- tools/delegate_tool_dispatch.py::_Batch: lines 26-63. Whole-call batch state plus Kensei verify/synthesis/profile fields.
- tools/delegate_tool_dispatch.py::_execute_and_aggregate: lines 139-191. Whole-batch join, Kensei primitives, finalisation and live transcript updates.
- tools/delegate_tool_dispatch.py::_dispatch_background: lines 324-371. Whole-batch detached dispatch; no unit partitioning.
- tools/delegate_tool_dispatch.py::_run_batch: lines 373-377.
- tools/async_delegation.py::recover_abandoned_delegations: lines 273-305.
- tools/async_delegation.py::active_count: lines 469-472.
- tools/async_delegation.py::active_task_count: lines 475-481.
- tools/async_delegation.py::_dispatch: lines 546-603. No upstream `slot_key`/`task_indexes` unit arguments in the baseline signature.
- tools/async_delegation.py::dispatch_async_delegation_batch: lines 634-661.
- tools/process_registry_notifications.py::_format_batch_delegation: lines 116-157.
- tools/process_registry_notifications.py::format_process_notification: lines 226-257.
- tools/process_registry.py: completion/process registry state and ownership consumers.

Profile, authority and safety seams to preserve
- gateway/run.py::_profile_runtime_scope and _async_profile_runtime_scope: lines 1650-1689.
- agent/secret_scope.py::set_secret_scope/reset_secret_scope and fail-closed credential resolution.
- tools/terminal_scope.py: terminal policy boundary.
- tools/delegate_tool_config.py: delegation provider/model overrides, runtime credentials, credential pools, fallback and authority helpers.
- tools/delegate_tool_child_run.py: sanitised summaries/errors, output-schema validation/retry, credential lease release and live transcript lifecycle.
- tools/delegate_tool_primitives.py: Kensei verification/synthesis.
- tools/delegate_tool_results.py: Kensei receipts/result handling.
- tools/delegate_tool_tasks.py: nested delegation output-shape contract.
- tools/delegate_tool_toolsets.py: inherited MCP/toolset preservation.
- tools/delegate_tool.py::_delegate_parent_ref and `_delegate_profile_name`: ancestry and profile identity.

Provider routing seams
- agent/chat_completion_helpers.py::_provider_preferences_for_agent: lines 380-389; flat routing only in baseline.
- hermes_constants.py::_canonical_model_variants: lines 892-926; existing spelling-tolerant variant machinery.
- hermes_constants.py::resolve_per_model_reasoning_effort: lines 929-941; reusable variant-resolution pattern.
- hermes_constants.py::resolve_per_model_provider_routing: missing from baseline; present upstream at lines 944-954.
- plugins/model-providers/openrouter/__init__.py: OpenRouter request body and endpoint-pin handling.
- agent/turn_recovery.py: fallback/routing error hints.
- gateway/run.py::_load_profile_secret_scope, provider runtime resolution and profile scope.

Upstream implementation map
- tools/delegate_tool_dispatch.py adds `_units_of`, `_dispatch_unit`, `_Batch.group`, `_Batch.unit_id`, per-unit joins, task indexes, one shared `slot_key`, partial child recording and inline fallback for later unit scheduling failure.
- tools/async_delegation.py adds `record_unit_child`, unit task indexes and slot-aware active capacity while retaining whole-call task indexes.
- tools/process_registry_notifications.py and CLI/TUI consumers add task-first/per-unit notices.
- agent/chat_completion_helpers.py overlays model-specific values over flat routing at `_provider_preferences_for_agent`.
- hermes_constants.py adds `resolve_per_model_provider_routing` using `_canonical_model_variants`.
- plugins/model-providers/openrouter/__init__.py applies routing only to the appropriate OpenRouter request path.

Kensei preservation matrix
- Profile dispatch and SOUL injection: preserve; Phase 1/2 RED coverage required.
- Depth-3 nested delegation: preserve; Phase 4 property/fuzz coverage required.
- Cycle guard and ancestry receipts: preserve; never let unit splitting bypass the guard.
- Output schemas: preserve one bounded validation retry and result shape.
- Verify/synthesis: preserve Kensei primitives; unit completion must not lose enriched fields.
- Completion registry/restart restoration: extend per-unit, not replace.
- Sanitised child context: preserve before dispatch and in persisted/replayed results.
- Live transcripts/task indexes: preserve whole-call indexes and correct per-unit transcript subsets.
- Parent capability ceiling: preserve; target profile cannot elevate tools/secrets.
- Profile-specific provider/model config: preserve and test against parent leakage.
- Custom provider profiles: preserve direct endpoint and explicit pin precedence.
- Credential pools: preserve provider-bound pool identity across fallback.
- Fallback registry: keep separate from OpenRouter sub-provider routing.
- Existing flat OpenRouter routing: preserve as the overlay base.

Phase 1 expected RED IDs
- tests/tools/test_delegate_profile.py profile construction/schema tests (7 baseline failures recorded).
- New unit/group completion tests for ungrouped independence, grouped coordination, task indexes, capacity and crash/notice semantics.
- New per-model routing tests for partial overlays, model variants, fallback/model changes, profile isolation and OpenRouter-only payloads.
