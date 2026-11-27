---
id: 01-zero-option-empty
scenario: zero_option with no declared skills or access
profiles: [core, zero_option]
research_mode: offline
expected_promotion_max: NOT_COMPUTABLE
---

# Transcript: Zero Option — empty capabilities

## OPEN

- **Request:** First cash within 7 days; zero capital; no fictional resources.
- **Actor:** Solo operator.
- **Current state:** Skills and access **not declared**.
- **Desired state:** Executable micro-loop or honest `NOT_COMPUTABLE`.
- **Authority:** advisory_only; no spend/execution.
- **Artifact:** `ZeroOptionPacket` + run receipt.

## ALIGN

- Canonical sources: none supplied.
- Non-goals: invent credentials, audiences, or capital.
- Research: offline (`research.enabled=false`) — no live fetch.
- Gap: executable capabilities and access are missing and **decision-critical**.

## ASCEND

- Conversion path: ZERO_STATE → OPTIONALITY blocked at extract-capabilities.
- Detected: `NON_EXECUTION`, `FICTIONAL_RESOURCE` risk if we invent assets.
- No micro-loop can be built without declared skills/access.

Claims:

- Operator requested zero capital. — `OBSERVED` (request constraint)
- Operator has marketable skills. — `NOT_COMPUTABLE` (no list supplied; offline; would invent)
- First cash in 7 days is feasible. — `NOT_COMPUTABLE`

## CLEAR

- Gate **G**: do not invent resources under no-fiction constraints.
- High narrative pressure must not create fictional access.
- Promotion cannot leave `NOT_COMPUTABLE`.

## SEAL

```yaml
status: NOT_COMPUTABLE
profiles_loaded: [core, zero_option]
promotion_state: NOT_COMPUTABLE
authority: advisory_only
grants_execution: false
human_review_required: true
data_requests: []
open_blocking_requests: []
research_attempts: []
not_computable_fields: [capabilities, access, opportunities]
reason: "No executable capabilities or access supplied"
```

**Operator next step:** Declare skills and access, then re-run (see `examples/zero-option-with-skills.brief.yaml`).
