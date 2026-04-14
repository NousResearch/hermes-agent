# Unified Operating Doctrine (Canonical v1)

Date: 2026-03-20  
Status: active (canonical top-level doctrine)  
Scope: control-plane behavior, docs/templates/protocol layer in `/home/yeqiuqiu/clawd-architect`

## Canonical source
- PDF: `/home/yeqiuqiu/.openclaw/media/inbound/Unified_Operating_Doctrine_for_a_Technical_AI_Operator_Syste---a1a3ed9d-8fec-43a2-82ce-8ed7da87af8a.pdf`

## Source-of-truth bootstrap discipline (mandatory)
Before starting non-trivial roadmap/system slices, bootstrap through:
1. `reports/openclaw_system_source_of_truth_map_2026-03-20.md`
2. `docs/ops/source_of_truth_and_subagent_bootstrap_doctrine_v1.md`
3. `reports/openclaw_full_roadmap_2026-03-20.md`
4. `reports/openclaw_full_roadmap_execution_table_2026-03-20.md`

Document-status rule:
- Canonical docs above + this unified doctrine = decision authority.
- Audit/progress docs = support context only.
- `reports/system_master_roadmap_2026-03-13.md` and `reports/system_prioritized_roadmap_table_2026-03-13.md` = historical reference only.
- evaluation-era multipool/Kimi routing docs (`docs/ops/model_routing_multipool_doctrine_v1.md`, `reports/kimi_k25_integration_synthesis_openclaw_2026-03-21.md`) = historical/reference-only, never canonical routing authority.
- `reports/openclaw_current_model_routing_matrix_2026-03-26.md` = support-only operational snapshot; canonical B6 routing authority remains `docs/ops/model_routing_no_llm_matrix_v1.md` + `docs/ops/model_pool_policy_v1.json`.

## Core doctrine (authoritative)
1. Main session is the **control plane** (orchestrator), not the default executor.
2. All nontrivial work must run as a bounded work slice with explicit contract.
3. Progress = **fresh evidence + verification**. “Running” is never sufficient proof.
4. Deterministic automation remains deterministic-first (cron/watchdog/canary critical path).
5. Status must be symptom-first and degrade loudly when truth/freshness is stale.
6. Model routing defaults to `NO_LLM` or lightweight paths; heavy model is exception-only.
7. No mutation until verify-before-resume gate passes.
8. False-green continuity is a sev-1 class operating failure.
9. Continuity artifacts must be successor-readable, timestamped, and freshness-scored.
10. User-sent docs/PDFs default to **subagent-led intake/analysis/integration** (B3/B2 worker-lane execution); main lane stays orchestration + approval.
11. Main lane must classify each work item (`task_class`, `risk_tier`, `scope_shape`, `verification_class`) before dispatch, choose explicit worker topology (single vs parallel vs staged serial), and enforce fold-in target (`canonical_doctrine | roadmap_pair | queue_continuity | support_only`) before completion claims.
12. Meaningful execution transitions must be reported via deterministic trigger packets (subagent finished, slice/phase landed, worker failed/junk, queue blocked/unblocked/relaunched, executor idle->relaunched); action/correction comes before narration.

## Operating modes
- `BLOCKER_BURNDOWN` is default while readiness/blocker classes remain active.
- `THROUGHPUT` is allowed only after explicit gate pass (fresh truth + verification + dependency health).
- Mode-switch protocol is defined in: `docs/ops/session_mode_blocker_vs_throughput_runbook_v1.md`.

## Verify-before-resume gate
Before resuming mutation-heavy or roadmap work, run the resume gate checklist:
- `docs/ops/verify_before_resume_gate_checklist_v1.md`

Gate result semantics:
- `allowed`: proceed with scoped execution.
- `caution`: proceed only with explicit constraints and tighter review windows.
- `forbidden`: stop mutation; restore truth/freshness first.

## Subordinate doctrine modules (normative detail)
The files below remain active and are interpreted under this canonical doctrine:

### Control-plane & governance
- `docs/ops/phase1_control_plane_doctrine_v1.md`
- `docs/ops/continuity_queue_state_model_v1.md`
- `docs/ops/core_roadmap_queue_layer_doctrine_v1.md`
- `ops/openclaw/architecture/schemas/core_roadmap_queue_layer.schema.json`
- `docs/ops/swarm_operating_contract_runbook_v1.md`

### Mode discipline & orchestration
- `docs/ops/blocker_burndown_control_loop_v1.md`
- `docs/ops/subagent_slot_fill_protocol_v1.md`
- `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`
- `docs/ops/execution_meaningful_event_reporting_checklist_v1.md`
- `docs/ops/invalid_output_retry_relaunch_contract_v1.md`
- `docs/ops/execution_supervisor_terminal_error_disposition_contract_v1.md`
- `docs/ops/session_mode_blocker_vs_throughput_runbook_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`

**Autonomous Execution Loop Rules:**
- **Loop Detection:** Autonomous execution loops polling the core roadmap queue must implement loop-detection heuristics (e.g., detecting repeated claim attempts on the same blocked candidate without queue-layer fingerprint changes) to force a fail-closed replan instead of spinning.
- **Trace Compaction:** Long-running execution loops must implement trace compaction to preserve context window limits without losing transactional history required for the final queue-layer commit.
- **Bounded Queue Pressure:** Long-running/live loops must use bounded queues or equivalent backpressure windows. Freshness-oriented capture/render stages may degrade by explicit policy only when drop/skip counters are surfaced; transactional, mutating, or evidence-bearing work must fail closed, retry, or block rather than silently drop.
- **Stale-Task Recovery:** Queue/executor supervision must detect stale `PROCESSING`/`RUNNING` work, apply deterministic drain/recovery or explicit blocked posture, and surface recovery counters/health so dirty in-flight state cannot linger silently.

### Model routing & LLM boundaries
- `docs/ops/model_routing_no_llm_matrix_v1.md`

### Required contracts/templates
- `docs/ops/templates/worker_slice_spec.template.json`
- `docs/ops/schemas/evidence_closeout.schema.json`
- `docs/ops/doctrine_object_contract_v1.md`
- `docs/ops/schemas/doctrine_object.schema.json`
- `docs/ops/templates/doctrine_object.template.json`
- `docs/ops/promotion_protocol_contract_v1.md`
- `docs/ops/promotion_trace_manifest_contract_v1.md`
- `docs/ops/templates/promotion_candidate.template.json`
- `docs/ops/schemas/promotion_candidate.schema.json`
- `docs/ops/templates/promotion_trace_manifest.template.json`
- `docs/ops/schemas/promotion_trace_manifest.schema.json`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`
- `docs/ops/schemas/model_qualification_packet.schema.json`
- `docs/ops/templates/model_qualification_packet.template.json`
- `docs/ops/core_roadmap_dependency_unblock_policy_pack_v1.md`
- `docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json`
- `docs/ops/source_of_truth_map_guard_policy_contract_v1.md`
- `docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json`
- `docs/ops/templates/source_of_truth_map_guard_policy_pack.template.json`
- `docs/ops/lane_topology_authority_contract_v1.md`
- `docs/ops/schemas/lane_topology_authority_contract.schema.json`
- `docs/ops/templates/lane_topology_authority_contract.template.json`
- `docs/ops/schemas/mutation_attestation.schema.json`
- `docs/ops/templates/mutation_attestation.template.json`
- `docs/ops/schemas/lane_action_intent.schema.json`
- `docs/ops/templates/lane_action_intent.template.json`
- `docs/ops/session_topology_contract_v1.md`
- `docs/ops/schemas/session_topology_contract.schema.json`
- `docs/ops/templates/session_topology_contract.template.json`
- `docs/ops/session_topology_transport_contract_v1.md`
- `docs/ops/schemas/session_topology_transport_contract.schema.json`
- `docs/ops/templates/session_topology_transport_contract.template.json`
- `docs/ops/orchestrator_api_contract_v1.md`
- `docs/ops/schemas/orchestrator_snapshot_resolve.schema.json`
- `docs/ops/templates/orchestrator_snapshot_resolve.template.json`
- `docs/ops/schemas/orchestrator_plan.schema.json`
- `docs/ops/templates/orchestrator_plan.template.json`
- `docs/ops/schemas/orchestrator_run.schema.json`
- `docs/ops/templates/orchestrator_run.template.json`
- `docs/ops/schemas/orchestrator_event_stream.schema.json`
- `docs/ops/templates/orchestrator_event_stream.template.json`
- `docs/ops/schemas/orchestrator_replay_resync.schema.json`
- `docs/ops/templates/orchestrator_replay_resync.template.json`
- `docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json`
- `docs/ops/templates/orchestrator_contract_bridge_packet.template.json`
- `docs/ops/knowledge_review_approval_promotion_queue_v1.md` (canonical queue runtime contract)
- `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- `docs/ops/schemas/knowledge_promotion_queue_entry.schema.json` (canonical queue entry schema)
- `docs/ops/templates/knowledge_promotion_queue_entry.template.json` (canonical queue entry template)
- `docs/ops/schemas/shared_memory_object.schema.json`
- `docs/ops/templates/shared_memory_object.template.json`
- `docs/ops/schemas/shared_memory_conflict_record.schema.json`
- `docs/ops/templates/shared_memory_conflict_record.template.json`
- `docs/ops/schemas/shared_memory_demotion_record.schema.json`
- `docs/ops/templates/shared_memory_demotion_record.template.json`
- `docs/ops/schemas/research_implementation_queue_item.schema.json`
- `docs/ops/templates/research_implementation_queue_item.template.json`
- `docs/ops/markdown_conversion_quality_gate_v1.md`
- `docs/ops/schemas/markdown_conversion_gate_packet.schema.json`
- `docs/ops/templates/markdown_conversion_gate_packet.template.json`
- `docs/ops/source_material_classification_layer_v1.md`
- `docs/ops/schemas/source_material_classification_packet.schema.json`
- `docs/ops/templates/source_material_classification_packet.template.json`
- `docs/ops/production_knowledge_ingestion_layer_v1.md`
- `docs/ops/schemas/production_knowledge_ingestion_packet.schema.json`
- `docs/ops/templates/production_knowledge_ingestion_packet.template.json`
- `docs/ops/document_pdf_os_operating_model_v1.md`
- `docs/ops/document_intake_batch_integration_protocol_v1.md`
- `docs/ops/schemas/document_intake_batch_integration.schema.json`
- `docs/ops/templates/document_intake_batch_integration.template.json`
- `docs/ops/release_evidence_ladder_contract_v1.md`
- `docs/ops/schemas/release_evidence_bundle.schema.json`
- `docs/ops/templates/release_evidence_bundle.template.json`
- `docs/ops/compatibility_path_lifecycle_contract_v1.md`
- `docs/ops/knowledge_review_approval_promotion_queue_contract_v1.md` (legacy compatibility helper contract)
- `docs/ops/schemas/knowledge_review_queue_item.schema.json` (legacy queue item schema)
- `docs/ops/templates/knowledge_review_queue_item.template.json` (legacy queue item template)
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/schemas/cross_lane_bridge_object.schema.json`
- `docs/ops/templates/cross_lane_bridge_object.template.json`
- `docs/ops/source_material_classification_layer_v1.md`
- `docs/ops/schemas/source_material_classification.schema.json`
- `docs/ops/templates/source_material_classification.template.json`

### Resume/freshness gate
- `docs/ops/verify_before_resume_gate_checklist_v1.md`

## Implementation priority (today)
1. Enforce canonical bootstrap through the source-of-truth map + bootstrap doctrine before lane execution.
2. Keep this unified doctrine as top-level operating interpretation; keep phase-1 docs as subordinate modules (no duplicate rewrites).
3. Use resume/mode gate runbooks to avoid false-green and unsafe mode escalation.
4. For new document/PDF batches, require subagent-first intake closeout (synthesis note + promote now/later/reference + minimal canonical edit plan + passing `document_intake_batch_integration` packet) before roadmap mutations.
5. Use `docs/ops/model_routing_no_llm_matrix_v1.md` as the day-to-day orchestration decision matrix for worker-count, parallelization guardrails, validator trigger class, and fold-in destination discipline.
6. Apply meaningful-event reporting triggers/checklist on queue/executor supervision turns so critical transitions are never silent.

## Out of scope in this promotion
- Runtime code changes, scheduler rewiring, or watchdog logic mutation.
- Historical doctrine migration beyond minimal pointer updates.
