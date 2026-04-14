# Anti-Drift Waiver Contract v1

Date: 2026-04-01  
Status: active (bounded EX-04 waiver contract)  
Owner lanes: A5 / C2 / XR  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose

Define a bounded, expiry-governed waiver mechanism for anti-drift checks.

This contract is intentionally narrow:
- it only applies to deterministic anti-drift guard checks,
- it is path/reason scoped,
- it is time-bounded,
- and it fails closed on malformed/unknown waiver state.

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

- Schema:
  - `docs/ops/schemas/anti_drift_waiver_register.schema.json`
- Template:
  - `docs/ops/templates/anti_drift_waiver_register.template.json`

## Core invariants (normative)

1. **Fail-closed parsing/validation**
   - malformed JSON, schema mismatch, invalid timestamp semantics, unknown reason/check ids, or duplicate waiver ids must block anti-drift decisions.

2. **Bounded temporal scope**
   - waivers require both `issued_at` and `expires_at`.
   - `expires_at` must be strictly later than `issued_at`.
   - expired waivers never suppress failures.

3. **Bounded logical scope**
   - waivers may target only explicit anti-drift `reason_ids` and/or `check_ids`.
   - targeted ids must exist in `state/continuity/latest/source_of_truth_map_guard_policy_v1.json` and be marked `waivable=true`.
   - optional `path_allowlist` further narrows applicability.

4. **No silent override on waiver failure**
   - waiver-register failures are non-waivable and force `BLOCK`.

5. **Evidence and accountability required**
   - each waiver requires owner, approver, justification, and at least one evidence reference.

## Register-level constraints

- `schema_version` must be `clawd.anti_drift_waiver_register.v1`.
- `checker` must be `source_of_truth_map_guard`.
- `max_active_waivers` is mandatory and bounded.
- all objects are strict (`additionalProperties: false`).

## Waiver status semantics

- `active`: eligible for application if non-expired and scope-matching.
- `revoked`: never applied.
- `expired`: never applied.

## Operational expectation

When anti-drift checker runs:
1. validate policy pack schema + semantic constraints,
2. validate waiver register schema and semantic constraints,
3. evaluate check failures,
4. apply only currently valid, scope-matching, policy-waivable waivers,
5. emit machine-readable decision + policy/waiver summary,
6. write durable drift artifact (fail closed on write failure).

## Out of scope

- generic/global waiver framework across unrelated lanes,
- policy exceptions for mutator boundary, release gate, or safety contracts outside anti-drift checker scope.
