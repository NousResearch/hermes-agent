# Trading Journal Boundary and Risk Contract v1 (`XT-601`)

Date: 2026-03-28  
Status: active (canonical XT lane boundary foundation)  
Owner: Architect  
Scope: Trading journal subsystem lane (`XT-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XT-601` defines the fail-closed boundary and risk contract that must exist before any trading journal schema/runtime activation.

This slice canonicalizes:
1. non-core separation guarantees for trading-lane work,
2. risk classes and mandatory controls for journal artifacts,
3. refusal/escalation semantics for unsafe requests,
4. governance alignment with `XG-801`.

This slice is boundary-first only. It does **not** implement typed schema pack depth (`XT-602`) or runtime ingest/review surfaces (`XT-603`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XT-601..XT-603`)
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/true_expanded_out_of_core_scope_filter_rules_v1.md`
- `ops/openclaw/architecture/trading_terminal_design_language.v1.yaml`

### Canonical outputs for `XT-601`
- this boundary/risk contract,
- risk controls matrix artifact,
- scope compliance check artifact,
- closeout report and queue evidence refs.

Deferred to follow-on slices:
- `XT-602`: typed append-only journal schema/template pack.
- `XT-603`: ingest/review runtime with deterministic replay surfaces.

---

## 3) Scope boundary and non-core separation guarantees

## 3.1 In-boundary for `XT-601`
- Governance-first definition of trading-journal lane boundaries.
- Risk controls for append-only journal evidence lifecycle.
- Refusal/escalation semantics for unsafe or out-of-scope trading requests.
- Explicit non-core separation constraints to prevent C3 lane contamination of core roadmap execution.

## 3.2 Out-of-boundary for `XT-601`
- No broker/exchange execution or autonomous trading actions.
- No portfolio management automation or strategy optimization runtime.
- No production ingestion/review runtime mutation paths.
- No schema/runtime completion claims for `XT-602` or `XT-603`.

## 3.3 Non-core separation guarantees (mandatory)
1. Trading-lane work remains within XT/XG/XB/XU governed boundaries; no mutation of core A*/B*/C1/C2 execution contracts is allowed by XT artifacts.
2. Any cross-lane bridge is reference-first and must satisfy `controlled_cross_lane_bridge_contract_v1`.
3. Unknown classification against out-of-core scope rules defaults to quarantine/block.
4. Trading journal outputs are decision-support records only; they are never authority to execute financial actions.

---

## 4) Risk model and control obligations

Risk tiers:
- `TR0_DOCUMENTARY` — documentation-only boundary artifacts.
- `TR1_RECORD_INTEGRITY` — append-only journal record integrity and provenance.
- `TR2_DECISION_INFLUENCE` — user-impacting decision context from journal outputs.
- `TR3_HIGH_IMPACT_ACTION` — any request implying external financial mutation.

Alignment to `XG-801` classes:
- `TR0_DOCUMENTARY` -> `RG0_LOW`
- `TR1_RECORD_INTEGRITY` -> `RG1_MODERATE`
- `TR2_DECISION_INFLUENCE` -> `RG2_HIGH`
- `TR3_HIGH_IMPACT_ACTION` -> `RG3_CRITICAL`

Activation constraints:
- `TR0/TR1`: governance + boundary controls required.
- `TR2`: runtime activation blocked until `XG-802` and `XG-803` are complete and `XT-602` is DONE.
- `TR3`: always refuse; no autonomous external action is permitted in XT lane scope.

---

## 5) Minimum control set for XT lane

1. **Append-only integrity control**
   - Journal events must be immutable; corrections use explicit supersedes links.
2. **Provenance and evidence anchoring**
   - Every entry must carry source/evidence references and deterministic hashes (introduced formally in `XT-602`).
3. **Risk disclosure control**
   - Any synthesized output must declare it is non-advisory, decision-support only.
4. **Autonomy refusal control**
   - Requests to place/modify/cancel trades must fail-close with escalation.
5. **Governance dependency gate**
   - Runtime activation remains blocked unless queue dependencies are DONE.
6. **Boundary crossover control**
   - Cross-lane writes require declared bridge contracts and owner tuple parity.

---

## 6) Refusal and escalation semantics

Escalation levels:
1. `T1_BOUNDARY_REFUSAL` — request is outside journal scope (e.g., execute trades).
2. `T2_APPROVAL_OR_OWNER_GAP` — required governance owner/approval context missing.
3. `T3_GOVERNANCE_BLOCK` — dependency gates not satisfied (`XT-602`, `XG-802`, `XG-803`, `XB-402`, `XU-502` as applicable).
4. `T4_SAFETY_HIGH_IMPACT_BLOCK` — high-impact financial mutation request; refuse and redirect to human-owned workflow.

Fail-close defaults:
- Unknown risk class -> refuse (`T1`).
- Missing owner tuple/dependency mismatch -> block (`T3`).
- External financial mutation intent -> block (`T4`).

---

## 7) Validation entrypoints for `XT-601`

- `python -m json.tool state/continuity/latest/xt_601_trading_journal_risk_controls_matrix_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xt_601_trading_journal_scope_compliance_check_2026-03-28.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 8) Closeout condition for `XT-601`

`XT-601` is complete only when:
1. this boundary/risk contract is canonical,
2. risk controls matrix + scope compliance check artifacts are published,
3. source-of-truth map references this contract/artifacts under XT lane,
4. queue slice `XT-601` is transitioned to `DONE` with evidence refs,
5. no schema/runtime completion claims are made for `XT-602`/`XT-603`.
