# Source-of-Truth Map Guard Policy Contract v1

Date: 2026-04-02  
Status: active (bounded EX-04 policy-enforcement contract)  
Owner lanes: A5 / C2 / XR  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose

Promote the source-of-truth map anti-drift checker from a pure detector into a policy-enforced gate.

This contract is intentionally narrow:
- it applies only to `ops/openclaw/continuity/check_source_of_truth_map_regressions.py`,
- it codifies the approved machine-readable drift checks for that checker,
- it declares which drift classes are waivable vs non-waivable,
- and it forces fail-closed behavior when runtime findings drift from the policy pack.

## Canonical runtime surfaces

- Checker runtime:
  - `ops/openclaw/continuity/check_source_of_truth_map_regressions.py`
- Policy pack:
  - `state/continuity/latest/source_of_truth_map_guard_policy_v1.json`
- Waiver register:
  - `state/continuity/latest/anti_drift_waiver_register.json`
- Durable anti-drift artifact:
  - `state/continuity/latest/source_of_truth_map_drift_latest.json`

## Machine artifacts

- Contract payload:
  - `state/continuity/latest/source_of_truth_map_guard_policy_v1.json`
- Schema:
  - `docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json`
- Template:
  - `docs/ops/templates/source_of_truth_map_guard_policy_pack.template.json`

## Core invariants (normative)

1. **Every emitted deterministic drift finding must map to policy**
   - each checker-emitted `check_id` / `reason_id` pair must match an approved policy row.
   - unknown or mismatched policy rows force `BLOCK`.

2. **Severity and waiver posture are machine-declared**
   - each policy row must declare:
     - `category`
     - `severity`
     - `waivable`
     - `summary`

3. **Waivers may target only waivable policy rows**
   - waivers referencing non-waivable or unknown `reason_ids` / `check_ids` are invalid.
   - a valid waiver may suppress only the specific waivable policy rows it scopes.

4. **Policy failures are non-waivable**
   - missing/invalid policy pack or policy/runtime mismatch is never suppressible.

5. **Decision remains fail-closed**
   - any unwaived blocker-class drift finding keeps the checker decision at `BLOCK`.

## Policy categories (bounded v1)

Allowed `category` values in the v1 pack:
- `registry_integrity`
- `lane_contract_integrity`
- `historical_reference_integrity`
- `path_resolution_integrity`

## Operator-surface expectation

The durable drift artifact must publish:
- the policy pack path,
- policy validation status,
- applied/non-applied waiver summary,
- and per-failure policy metadata (`category`, `severity`, `waivable`, `summary`) when a failure exists.

## Out of scope

- generic anti-drift policy systems outside the source-of-truth map guard,
- CI workflow rewiring beyond this checker/runtime/contract slice,
- queue-map reconciliation beyond the current EX-04 bounded source-of-truth guard.
