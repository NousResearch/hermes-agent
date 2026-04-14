# Model Routing Doctrine v1

Date: 2026-03-30  
Status: active (Program C / C1 canonical doctrine)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 1) Purpose

Convert model selection from conversational habit into an explicit operating contract.

This doctrine defines:
- deterministic-first routing behavior,
- task-class -> lane policy,
- cheap-worker vs premium-trust boundaries,
- validator and break-glass control paths,
- and how routing policy connects to pattern-harvest work (keep/adapt/avoid posture) without cargo-cult adoption.

---

## 2) Consolidated source posture (archive-mined + current)

This v1 consolidates signal from:

### Canonical operating surfaces (current authority)
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `docs/ops/session_topology_contract_v1.md`
- `docs/ops/model_pool_policy_v1.json`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`
- `reports/openclaw_system_source_of_truth_map_2026-03-20.md` (B6 mapping)

### Archive-mined model-routing/pattern-harvest inputs
- `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md` (C1 target)
- `reports/openclaw_missing_layers_system_roadmap_foldin_2026-03-30.md` (S3/S5 layer framing)
- `reports/efficiency_layer_doctrine_pack_2026-03-28.md` + `state/continuity/latest/xe102_task_class_to_model_family_matrix_2026-03-28.json` (family defaults/escalation)
- `reports/multimodel_roster_utilization_audit_2026-03-29.md`
- `reports/multimodel_roster_audit_fold_in_2026-03-29.md`
- `reports/openclaw_archive_mining_creative_systems_intelligence_2026-03-30.md` (keep/adapt/avoid donor pipeline posture)
- `reports/repo_work_durability_audit_and_reuse_index_2026-03-29.md`
- `reports/repo_cross_wave_full_synthesis_openclaw_2026-03-26.md`

### Current support posture snapshots (non-canonical by themselves)
- `reports/openclaw_current_model_routing_matrix_2026-03-26.md`
- `reports/model_lineup_benchmark_2026-03-21.txt`
- `reports/xe307_model_routing_guard_and_provider_normalized_telemetry_pack_closeout_2026-03-29.md`

---

## 3) Non-negotiable invariants

1. **Deterministic-first is mandatory.**
   - Scheduler, watchdog, continuity truth, lock/fencing, gate/pass-block authority remain `NO_LLM`.
2. **LLMs are never authority for control-plane truth verdicts.**
   - Models may summarize deterministic outputs only.
3. **Heavy Codex is premium mutation lane, not default thinking lane.**
4. **Escalation is evidence-gated and fail-closed.**
   - Missing escalation evidence blocks upward routing.
5. **Exact-model allowlists are policy-bound.**
   - Family intent does not silently widen canonical allowlists.
6. **Support snapshots are not canonical policy.**
   - Canonical authority remains this doctrine + B6 contracts.

---

## 4) Lane taxonomy (v1)

## 4.1 Pool 0 — `NO_LLM` (deterministic authority lane)
Use for:
- scheduler/cooldown/backoff/idempotency decisions,
- continuity/freshness/gate checks,
- watchdog and cron authority verdicts,
- lock/fencing and readiness/blocker transitions,
- schema/gate runner pass-block authority.

Rule: if deterministic path exists for authority decision, use it.

## 4.2 Pool 1/2/3 mapped into `SPARK` (cheap worker lane)
Use for bounded helper work where model assistance is justified and risk is limited:
- reading/triage/audit compression,
- bounded extraction/classification,
- support-only synthesis,
- low-risk schema/code-adjacent helper tasks.

Current canonical allowlist for `SPARK` lives in `model_pool_policy_v1.json`.
Family-level defaults (DeepSeek/Gemini/Kimi) are intent-level routing posture and must resolve to qualified exact model keys.

## 4.3 Pool 4 mapped into `HEAVY` (premium trust lane)
Use for:
- multi-file coupled mutation,
- high-risk refactoring,
- canonical contract/doctrine/spec edits,
- release/cutover logic,
- final implementation convergence.

Default `HEAVY` model key is governed by policy allowlist and rollout stage requirements.

## 4.4 Validator lane (overlay control lane)
Validator is an independent review overlay, not a separate route class.

`validator_required` when any of:
- risk tier `high|critical`,
- canonical doctrine/spec/schema/policy mutation,
- routing/authority/lease/fail-close behavior change,
- roadmap truth mutation.

`validator_plus_human` when policy/risk posture is loosened or external-impacting behavior changes.

## 4.5 Break-glass lane (emergency overlay)
Break-glass is a time-bounded emergency override path, never default routing.

Required controls:
- explicit incident id + reason,
- bounded TTL,
- auditable event trail,
- authority role check,
- mandatory post-incident rollback/review packet.

If audit logging is unavailable, break-glass must fail closed unless explicitly authorized by higher-order emergency policy.

---

## 5) Task-class -> lane mapping (v1)

| Task class / work profile | Default route | Family default posture | Escalate when | Hard bans |
|---|---|---|---|---|
| `watchdog`, scheduler/continuity/gate authority | `NO_LLM` | deterministic only | never for authority verdict | any LLM as authority |
| `reading`, `triage`, `audit_compression` | `SPARK` | DeepSeek-first helper posture | context scale/quality needs exceed helper quality gate | direct `HEAVY` without evidence |
| `research`, `planning` (support synthesis) | `SPARK` | Gemini-first posture | synthesis fails quality/citation gates or becomes mutation-bound | premium lane as first pass by default |
| `comparison` / repo-donor mining | `SPARK` | Kimi-first comparison posture | cross-surface consolidation requires canonical merge owner | treating comparison output as canonical truth directly |
| bounded schema/code support execution | `SPARK` | codex-spark/deepseek helper posture | coupling/risk rises or validator flags unresolved risk | broad HEAVY defaulting |
| `implementation` / coupled multi-file mutation | `HEAVY` | Codex premium lane | n/a (already premium lane) | cheap/helper lanes as mutators for critical paths |
| final canonical promotion/fold-in | `HEAVY` + validator overlay | Codex + validator | n/a | support-only lanes self-promoting canonical truth |

Notes:
- Model-family defaulting follows XE-102/XE-306 posture but exact-model routing must satisfy `model_pool_policy_v1.json`.
- If preferred family has no qualified model key for requested stage, router falls back deterministically by policy, or blocks.

---

## 6) Cheap worker / premium trust / validator / break-glass rules

## 6.1 Cheap worker rules (`SPARK`)
1. Default-down from `HEAVY` unless mutation/coupling risk justifies premium lane.
2. For support-only heavy escalation, require non-risk escalation signals + artifact refs.
3. Log misrouting signals (especially premium-on-synthesis incidents).

## 6.2 Premium trust rules (`HEAVY`)
1. Reserve for mutation, convergence, and canonical truth edits.
2. Do not use as first-pass for recurring support audits/triage.
3. Preserve test/verification obligation before closeout claims.

## 6.3 Validator rules
1. Independent validator is required for high-risk/canonical mutations.
2. Validator must be evidence-linked (not subjective vibes).
3. Promotion blocks on unresolved validator objections.

## 6.4 Break-glass rules
1. Break-glass does not bypass evidence recording.
2. Break-glass decisions are reversible and review-mandatory.
3. Repeated break-glass use is a policy-smell and must trigger routing-policy review.

---

## 7) Qualification and rollout gates

No exact model key is routing-authoritative without qualification + rollout governance:
- qualification checklist + thresholds from `model_qualification_rollout_gate_contract_v1.md`,
- lane authority and staged rollout (`SHADOW -> CANARY -> RING -> FULL`),
- kill-switch + rollback fields required for non-trivial rollout states.

Gate failures block promotion and may force fallback route class/model.

---

## 8) Pattern-harvest integration rules (C-layer coupling)

Routing policy must support donor/pattern harvesting without turning mined ideas into auto-truth.

For repo/donor mining outputs:
1. classify output as `keep | adapt | avoid | defer | reject` (decision state),
2. require evidence refs + lane mapping + risk notes,
3. keep support outputs as support until validator-gated promotion,
4. treat mined patterns as selective donors, never wholesale architecture adoption.

Operational routing implication:
- donor comparison/mining defaults to `SPARK` comparison/research lanes,
- canonical adoption decisions require `HEAVY` finalizer + validator overlay.

---

## 9) Telemetry + governance signals (required)

Minimum routing governance signals:
- task-class tagging coverage (`tagged` vs `missing`),
- route-class selection counts,
- family utilization,
- provider-normalized cost (not provider-blind zero-cost assumptions),
- escalation count + reason,
- fallback count + reason,
- misrouting incidents:
  - `codex_used_for_synthesis_only`,
  - unmapped task-class warnings,
  - control-path LLM authority violations.

Control-path LLM authority violation is Sev-1 policy drift.

---

## 10) Canonical vs support boundary (explicit)

Canonical routing authority:
- `docs/ops/model_routing_doctrine_v1.md` (this file)
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `docs/ops/session_topology_contract_v1.md`
- `docs/ops/model_pool_policy_v1.json`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`

Support-only routing snapshots/audits (informative, non-authoritative alone):
- `reports/openclaw_current_model_routing_matrix_2026-03-26.md`
- `reports/multimodel_roster_utilization_audit_2026-03-29.md`
- `reports/multimodel_roster_audit_fold_in_2026-03-29.md`

---

## 11) C1 completion statement

C1 is complete when:
- task-class to lane mapping is explicit,
- cheap-worker/premium-trust/validator/break-glass rules are explicit,
- routing selection is policy-driven and auditable,
- and model selection no longer depends on conversational memory.

This doctrine establishes that v1 baseline.
