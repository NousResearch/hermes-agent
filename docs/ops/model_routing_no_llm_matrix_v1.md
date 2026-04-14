# No-LLM / Model-Routing + Work-Orchestration Matrix (Phase 1)

Date: 2026-03-19  
Updated: 2026-04-09 (local Gemma support-route promotion)  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Operational routing and execution-shape checklist for deterministic-first, operator-usable day-to-day work.

This file is the canonical practical replacement for evaluation-era multipool routing ideas; historical multipool docs stay reference-only.

## A) No-LLM first checklist (run before every slice)
- [ ] A deterministic local/scripted path exists for this action.
- [ ] Action touches scheduler gating, idempotency, retries, locks, or side-effect authorization.
- [ ] Action requires strict coherence tuple checks (`world_anchor_id`, `policy_epoch_id`, `connector_snapshot_id`, `valid_until`, `build_generation_id`).
- [ ] Output is machine-validated contract data (JSON/schema/invariants), not narrative synthesis.
- [ ] Risk of uncited/hallucinated claims is unacceptable for this step.

If any checked item is true, default route is **NO_LLM** unless explicitly overridden with evidence.

---

## B) Route-class matrix (model-selection only)

| Work unit class | Default route | Escalate to SPARK when | Escalate to HEAVY when | Hard bans |
|---|---|---|---|---|
| Scheduler gates, cooldown/backoff, idempotency, lock/fencing, retry policy | **NO_LLM** | Never (analysis-only summaries permitted post-fact) | Never | LLM cannot be authority for dispatch/gating decisions |
| Continuity/coherence checks and freshness expiry (`valid_until`) | **NO_LLM** | Need concise operator summary of deterministic outputs | Never for final gate verdict | No green-state decision from LLM |
| Watchdog triage from structured logs/metrics | **NO_LLM** | Need short textual incident digest | Multi-surface synthesis across many artifacts | No mutation decisions purely from model narrative |
| Local repo scan, log compression, draft synthesis, shortlist generation from trusted evidence | **LOCAL_SUPPORT** | Need stronger coding reliability, broader tool behavior, or cloud-provider convergence | Any final-authority, multi-file risky, or canonical-truth mutation work | Local support models cannot be final authority or mutate canonical truth without stronger review |
| Bounded extraction/classification from trusted evidence pack | **SPARK** | N/A | Complex cross-document synthesis with strict citations | Must pass schema + pointer gates |
| Multi-file implementation reasoning and high-risk refactor planning | **SPARK** | N/A | If SPARK fails quality gates or cannot converge | LLM cannot skip verification/validator lane |
| Final copy/synthesis for operator brief | **SPARK** | N/A | If precision/citation depth is insufficient | No uncited factual claims |

Routes:
- `NO_LLM` = deterministic scripts/tools only
- `LOCAL_SUPPORT` = `ollama/gemma4:26b` (local support-only lane)
- `SPARK` = `openai-codex/gpt-5.3-codex-spark`
- `HEAVY` = `openai-codex/gpt-5.3-codex`

---

## C) Main-lane work classification (execution shape)

For every new work item, main lane must classify all four fields before dispatch:
1. `task_class`: `diagnose | implementation | docs_doctrine | research_synthesis | intake_integration | release_ops`
2. `risk_tier`: `low | medium | high | critical`
3. `scope_shape`: `single_surface | multi_surface_disjoint | multi_surface_coupled`
4. `verification_class`: `self_check | validator_required | validator_plus_human`

### Execution-shape decision table

| Classifier outcome | Worker topology | Parallel policy | Notes |
|---|---|---|---|
| `single_surface` + `low/medium` + clear deterministic verification | **1 worker slice** | none (serial) | Fastest default path |
| `multi_surface_disjoint` + independent deliverables + clear merge owner | **2-4 worker slices** | allowed | Each branch must own disjoint artifact targets |
| `multi_surface_coupled` or shared canonical files | **1 worker at a time** (or staged micro-batches) | forbidden until merge barrier | Avoid conflicting mutations and fake progress |
| `high/critical` risk regardless of shape | **1 executor + independent validator lane** | implementation serial, validation separate | No self-approval |
| broad intake/synthesis with lane impacts (PDF/doc/research batch) | **fan-out analyzers + 1 integrator** | allowed for analysis, serial for canonical edits | Integrator composes promote-now/later/reference |

Mode cap alignment:
- `BLOCKER_BURNDOWN`: max active workers = 2 (unless explicit exception with rationale).
- `THROUGHPUT`: max active workers = 4, only with non-overlapping write scopes.

---

## D) Parallelization hard rules
1. No parallel branches may mutate the same canonical file in the same batch.
2. Every parallel branch must declare:
   - owned paths,
   - expected artifact,
   - verification command,
   - kill condition.
3. Main lane sets a merge barrier before any roadmap/doctrine truth mutation.
4. If one branch fails a hard gate, affected coupled branches pause; do not force-merge partial truth.
5. Parallelism is for **independent evidence production**, not for racing conflicting doctrine edits.

---

## E) Validator review policy (operator-default)

`validator_required` when any condition is true:
- risk tier is `high` or `critical`;
- change modifies canonical doctrine/spec/schema/policy contracts;
- change alters roadmap status, maturity, priority, or sequencing claims;
- change affects routing/authority/lease/fail-close behavior;
- implementation slice touches safety gates, mutator ingress, or release rollback controls.

`validator_plus_human` when:
- external-impacting behavior/policy changes are proposed;
- security/risk posture is loosened;
- ambiguous evidence could change canonical direction.

`self_check` allowed only for low-risk, non-canonical support artifacts (notes, interim analysis, draft synthesis).

---

## F) Fold-in protocol (roadmap / queue / truth surfaces)

After slice completion, main lane must classify output destination:

1. **Canonical doctrine/spec truth** (normative behavior changed)
   - update relevant `docs/ops/*.md` contract/doctrine files,
   - ensure lane ownership remains mapped in `reports/openclaw_system_source_of_truth_map_2026-03-20.md`.

2. **Roadmap truth** (priority/maturity/order changed)
   - update both roadmap authorities together:
     - `reports/openclaw_full_roadmap_2026-03-20.md`
     - `reports/openclaw_full_roadmap_execution_table_2026-03-20.md`
   - reconcile lane mapping in source-of-truth map in same slice.

3. **Queue/continuity truth** (execution state changed, no roadmap shift)
   - emit/update queue + handoff evidence artifacts,
   - keep state/continuity surfaces fresh and successor-readable,
   - attach evidence refs in closeout packet.

4. **Support evidence only** (no authority change)
   - write/update `reports/*.md` synthesis,
   - do not silently mutate canonical docs.

Completion claim is valid only after required fold-in + verification evidence exists.

---

## G) LLM hard gates (must-pass)
From model strategy contract:
- JSON parse + schema validity: **>= 99.5%**
- Evidence-pointer resolution: **100%**
- No-evidence abstention F1: **>= 0.95**
- Evidence-present non-abstain recall: **>= 0.95**

Operational handling:
- Invalid JSON = deterministic failure.
- Unresolvable pointer = deterministic failure.
- Unsupported claim rate above 1–2% = non-promotable output.

---

## H) Minimal route + orchestration decision record (required in slice spec)
Record for each slice:
- `selected_route`: `NO_LLM | LOCAL_SUPPORT | SPARK | HEAVY`
- `reason`
- `escalation_trigger` (if not NO_LLM)
- `fallback_route`
- `task_class`
- `risk_tier`
- `scope_shape`
- `worker_topology`: `single | parallel_fanout | staged_serial`
- `verification_class`: `self_check | validator_required | validator_plus_human`
- `verification_plan`
- `fold_in_target`: `canonical_doctrine | roadmap_pair | queue_continuity | support_only`

Production enforcement note (2026-03-31):
- `ops/openclaw/continuity.sh session-route` now enforces worker-slice allocation fields by default (`scope_shape`, `verification_class`, `worker_topology`, `fold_in_target`).
- High/critical requests fail closed when `verification_class=self_check` or `worker_topology=parallel_fanout`.
- Coupled scope (`multi_surface_coupled`) fails closed with `parallel_fanout`.
- Temporary compatibility bypass is explicit only via `--legacy-allow-missing-worker-allocation-contract`.

---

## I) B6 work-profile defaults and support-boundary guard

Stable B6 route defaults (canonical):
- **A1-A3 state/gate/control work** (scheduler gating, continuity truth verdicts, lock/fencing, failover guardrails): route **NO_LLM**; model narratives may summarize outputs but cannot issue final gate decisions.
- **Local support-only scan/compress/draft/shortlist work on trusted evidence**: route **LOCAL_SUPPORT** when the task is bounded, non-authoritative, and cheap-local execution is preferred. Escalate to **SPARK** for coding reliability or broader provider behavior, to **HEAVY** for high-risk or final-authority work, and back to **NO_LLM** for deterministic gates.
- **Bounded extraction/classification and low-risk synthesis**: route **SPARK** by default; escalate only when citation depth or convergence requirements exceed SPARK quality gates.
- **High-risk multi-file implementation or coupled cross-surface synthesis** (including broad intake integrator synthesis): route **HEAVY**.

Canonical/supported boundary:
- Canonical routing authority = this matrix + `docs/ops/model_pool_policy_v1.json`.
- Provider- or environment-specific assignment snapshots belong in support artifacts (currently `reports/openclaw_current_model_routing_matrix_2026-03-26.md`).
- Support snapshots do **not** change allowlists, budgets, or rollout authority unless promoted through canonical policy/spec edits with validation evidence.

## J) 2026-03-29 roster-utilization fold-in guardrails

Immediate policy adjustments from actual-work audit:
- **Recurring watchdog/cron/canary/checkpoint/scheduler-governance jobs stay `NO_LLM` as authority.** Any model-written digest must be an explicit secondary downstream step and cannot share the authoritative verdict path.
- **Authority enforcement is operationalized via XE-304 pack artifacts**: `ops/openclaw/no_llm_watchdog_cron_authority_guard.sh`, `ops/openclaw/harden_no_llm_watchdog_cron_authority.sh`, and blocker-side-effect routing in `ops/openclaw/cron_protocol_outcome.sh`.
- **Bounded code/schema/support execution defaults to `SPARK` before `HEAVY`.** In the currently qualified helper pool, prefer `codex-spark` for bounded code-adjacent helper work and escalate only when coupling/risk/quality-gate evidence justifies it.
- **Support-only audits, convergence packs, ranking passes, and queue-shaping notes require escalation evidence before `HEAVY` is the first worker.** Heavy Codex remains the premium mutation/finalization lane, not the default thinking lane.
- **Execution-surface enforcement is explicit:** session-route requests mark support-bound helper work via `fold_in_target=support_only` (or `support_only=true`), and router escalation blocks heavy selection when non-risk signals + artifact refs are missing.
- **Subagent slot-fill reporting must include concrete model selection plus escalation evidence refs** so support-only heavy usage is queryable without transcript archaeology.
- **DeepSeek helper lane is now activated in bounded `SPARK` posture for `reading`/`triage`/`audit_compression` first-pass work with deterministic fallback to existing qualified helpers.** This activation is support-lane only: it does not widen watchdog/control authority (`NO_LLM`) or implementation mutation authority (Codex-heavy path remains evidence-gated).
- **Local Gemma (`ollama/gemma4:26b`) is canonically admitted only as `LOCAL_SUPPORT`.** It is approved for repo scanning, log compression, drafting, shortlist generation, and first-pass support synthesis, but it does not widen canonical authority, does not override `NO_LLM`, and does not replace Codex for final mutation or coupled implementation work.
- **Kimi remains comparison-specialist only.** Productive repo/comparison support use does not by itself promote Kimi into general planning or implementation authority.
- **Provider-normalized cost telemetry now uses explicit fallback normalization and unpriced-event accounting.** Routing governance must consume `state/continuity/model_rollout_cost/latest.json` (`provider_normalized_cost`) rather than treating zero-cost session-index rows as proof that non-OpenAI lanes are free.
