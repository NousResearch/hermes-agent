# Personal Context Graph Schema Pack v1 (`XP-302`)

Date: 2026-03-29  
Status: active (canonical for `XP-302`)  
Owner: Architect  
Scope: Personal OS typed context-graph schema/template pack (`XP-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XP-302` lands the typed schema pack required before Personal OS runtime loop activation (`XP-303`).

This slice defines machine-validated contracts for:
1. planning objects (`goal`, `routine`, `constraint`, `event`, `commitment`),
2. reflective learning objects (`decision_record`, `after_action_review`, `lesson_card`, `pattern_card`),
3. provenance envelope compatibility with `XB-401`,
4. append-only update policy (`revision`, `supersedes_object_id`, `previous_object_hash`, `object_hash`).

This slice is schema-pack only. It does **not** activate runtime loop behavior.

---

## 2) Canonical inputs

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XP-302` authoritative queue contract)
- `docs/ops/personal_os_scope_boundary_contract_v1.md` (`XP-301` boundary + refusal/escalation + approval semantics)
- `docs/ops/downstream_capability_backend_contract_v1.md` (`XB-401` domain-event provenance envelope dependency)
- `docs/ops/c3_activation_governance_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/low_noise_interaction_policy_v1.md`

---

## 3) Schema-pack artifacts (normative)

### 3.1 Graph-object schema
- `docs/ops/schemas/personal_context_graph_object.schema.json`

Defines shared envelope for all Personal OS context graph objects:
- object identity + lifecycle (`object_id`, `object_type`, `status`),
- append-only evolution (`revision`, supersedes/hash parity),
- `XB-401`-compatible provenance envelope (`event_id`, `source_connector_id`, `risk_class`, `route_class`, `auth_tier`, `payload_sha256`),
- XP governance envelope (`risk_tier`, `approval_tier`, `escalation_level`, `advisory_only=true`).

### 3.2 Templates
- `docs/ops/templates/personal_context_goal.template.json`
- `docs/ops/templates/personal_context_routine.template.json`
- `docs/ops/templates/personal_context_constraint.template.json`
- `docs/ops/templates/personal_context_event.template.json`
- `docs/ops/templates/personal_context_commitment.template.json`
- `docs/ops/templates/personal_context_decision_record.template.json`
- `docs/ops/templates/personal_context_after_action_review.template.json`
- `docs/ops/templates/personal_context_lesson_card.template.json`
- `docs/ops/templates/personal_context_pattern_card.template.json`

### 3.3 Pack registry + fixtures
- `ops/openclaw/architecture/personal_context_graph_schema_pack.v1.json`
- `tests/fixtures/xp/personal_context_graph_objects_fixture_v1.json`

---

## 4) Mandatory contract semantics

1. **Planning + learning parity**
   - Schema pack must include all planning object classes and reflective-learning object classes.

2. **Provenance compatibility with XB lane**
   - Every object must carry `XB-401`-compatible provenance fields.
   - `route_class` remains advisory-only for this slice.

3. **Append-only evolution discipline**
   - `revision=1` records cannot supersede.
   - `revision>=2` records must provide both supersede id and previous hash parity.

4. **Decision-review linkage integrity**
   - `after_action_review` must reference an existing `decision_record`.
   - `lesson_card` and `pattern_card` must reference prior decision/review learning artifacts.

5. **XP-301 boundary compatibility**
   - Governance metadata must preserve XP refusal/escalation and approval semantics.

---

## 5) Validation entrypoints

- `pytest -q tests/test_xp_302_personal_context_graph_schema_pack.py`
- `bash ops/openclaw/architecture/validate_personal_context_graph_schema_pack.sh --json`
- `python -m json.tool docs/ops/schemas/personal_context_graph_object.schema.json`
- `python -m json.tool ops/openclaw/architecture/personal_context_graph_schema_pack.v1.json`
- `python -m json.tool tests/fixtures/xp/personal_context_graph_objects_fixture_v1.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 6) Closeout condition for XP-302

`XP-302` is complete only when:
1. schema/template pack is canonical and valid,
2. planning + decision/review/learning object classes are all present and validated,
3. provenance linkage examples demonstrate `XB-401` envelope compatibility,
4. fixture validation includes append-only evolution checks,
5. queue slice `XP-302` transitions to `DONE` with evidence refs,
6. no runtime completion claims are made for `XP-303`.
