# Personal OS Scope Boundary Contract v1 (`XP-301`)

Date: 2026-03-28  
Status: active (canonical XP lane boundary foundation)  
Owner: Architect  
Scope: Personal OS / life-assistant lane (`XP-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XP-301` defines the fail-closed boundary contract that must exist before any Personal OS runtime loop may activate.

This contract canonicalizes:
1. domain allow/deny boundaries,
2. risk-tier and refusal/escalation semantics,
3. approval semantics and autonomy envelope,
4. alignment with `XG-801` risk classes.

This slice is foundation-only. It does **not** implement schema-pack depth (`XP-302`) or runtime loops (`XP-303`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XP-301..XP-303`)
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/low_noise_interaction_policy_v1.md`

### Canonical outputs for `XP-301`
- this boundary contract,
- refusal/escalation matrix artifact,
- XG-801 alignment check artifact,
- boundary-violation fail-close test packet,
- approval semantics log artifact,
- closeout report and queue evidence refs.

Deferred to follow-on slices:
- `XP-302`: typed personal context graph schema/template pack.
- `XP-303`: runtime assistant loops and intervention cadence.

---

## 3) Scope boundary (allow / deny / escalate)

## 3.1 Explicitly allowed (in-boundary)
- Goal/routine/constraint capture and organization.
- Scheduling suggestions and reminder proposals (non-binding).
- Reflective journaling support and review prompts.
- Information synthesis for personal planning with provenance references.
- Escalation recommendations and operator-facing action cards.

## 3.2 Explicitly denied (out-of-boundary)
- Autonomous execution of high-impact external actions (financial, legal, medical, account/security mutations).
- Impersonation, hidden delegation, or bypass of user approvals.
- Claims of professional diagnosis/treatment/advice authority.
- Covert data exfiltration or cross-lane writes outside declared bridge contracts.
- Runtime activation when dependency gates or owner tuple integrity are unresolved.

## 3.3 Escalate-before-proceed classes
- Safety-sensitive self-harm/medical/legal/financial topics.
- Requests that imply irreversible consequences.
- Any action request with ambiguous user intent or ambiguous identity/authority.
- Any operation whose risk tier is unknown.

Fail-close default for unknown class: `REFUSE_AND_ESCALATE`.

---

## 4) Risk tiers and XG-801 alignment

Personal OS action tiers:
- `PX0_INFO`: low-risk informational support.
- `PX1_ASSIST`: bounded planning assistance with no external mutation.
- `PX2_HIGH_IMPACT`: consequential guidance or potential user-impacting automation.
- `PX3_SAFETY_CRITICAL`: safety-sensitive interactions.

Alignment to XG risk classes:
- `PX0_INFO` -> `RG0_LOW`
- `PX1_ASSIST` -> `RG1_MODERATE`
- `PX2_HIGH_IMPACT` -> `RG2_HIGH`
- `PX3_SAFETY_CRITICAL` -> `RG3_CRITICAL`

Activation ceilings inherited from `XG-801`:
- `PX0/PX1`: governance validation + bounded integration posture.
- `PX2`: blocked from runtime activation until `XG-802` and release-ladder obligations are satisfied.
- `PX3`: hard-blocked from runtime activation until `XG-802` + `XG-803` and domain-safety contracts are active.

---

## 5) Refusal and escalation contract

Escalation levels:
1. `E1_BOUNDARY_REFUSAL` - request violates lane boundary or asks for denied action.
2. `E2_APPROVAL_REQUIRED` - request is in-boundary but requires explicit approval tier not yet satisfied.
3. `E3_SAFETY_ESCALATION` - potential safety-sensitive user impact; provide safe alternatives and direct escalation guidance.
4. `E4_GOVERNANCE_BLOCK` - dependency, ownership, or release obligations unmet; no activation allowed.

Refusal policy:
- Unknown domain class -> `E1` fail-close.
- Unknown/absent approval context -> `E2` fail-close.
- Safety-critical ambiguity -> `E3` fail-close.
- Dependency/owner/release gap -> `E4` fail-close.

---

## 6) Approval semantics and autonomy envelope

Approval tiers:
- `AP0_NONE`: no approval required (informational responses only).
- `AP1_EXPLICIT_USER_CONFIRM`: clear per-action user confirmation in active session.
- `AP2_DUAL_CONFIRM`: explicit confirmation + verified provenance/trace artifact.
- `AP3_PROHIBITED_UNTIL_GOVERNANCE`: action class blocked until required XG/XP dependencies close.

Autonomy envelope:
1. Personal OS may suggest and structure work autonomously only inside `PX0/PX1`.
2. Personal OS may not execute external high-impact actions; it may only prepare proposals pending approvals.
3. Any request mapped to `PX2/PX3` requires refusal/escalation unless all relevant governance gates are satisfied.
4. All approvals must be append-only logged with timestamp, request class, decision, and rationale.

---

## 7) Runtime/UX lane reference obligations

This contract is normative for:
- UX/interaction throttling and intervention style under `docs/ops/low_noise_interaction_policy_v1.md`.
- Future Personal OS runtime loops (`XP-303`) and operator-facing UX slices (`XU-*`) handling approvals/escalations.

No Personal OS runtime lane may claim activation readiness unless this boundary contract is cited as an active dependency.

---

## 8) Validation entrypoints for `XP-301`

- `python -m json.tool state/continuity/latest/xp_301_personal_os_refusal_escalation_matrix_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xp_301_xg801_risk_class_alignment_check_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xp_301_boundary_violation_failclose_tests_2026-03-28.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 9) Closeout condition for `XP-301`

`XP-301` is complete only when:
1. this boundary contract is canonical,
2. refusal/escalation matrix + XG-801 alignment artifact + fail-close tests are published,
3. approval semantics log artifact exists,
4. source-of-truth map references this contract/artifacts under XP lane,
5. queue slice `XP-301` is `DONE` with bounded evidence refs,
6. no schema/runtime completion claims are made for `XP-302`/`XP-303`.
