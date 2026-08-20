---
id: 09-opportunity-mining
scenario: blocked transition to first cash without invented buyer
profiles: [core, opportunity_mining, commercial, zero_option]
research_mode: online
expected_promotion_max: TESTABLE
---

# Transcript: Opportunity mining — blocked transition

## OPEN

- **Request:** Find a first-cash opportunity from existing skills.
- **Actor:** Solo operator; capital constrained.
- **Desired state:** Testable wedge with completion proof.
- **Authority:** advisory_only.

## ALIGN

- Required fields: affected user, economic buyer, transition market, completion proof.
- Non-goals: invent audience, capital, or relationships.
- Gaps: buyer identity; declared skills/access incomplete in this exemplar.

### DataRequest

```yaml
field: declared_skills_access_and_buyer
why_decision_critical: Zero-option and opportunity packets cannot invent capabilities or buyer
sensitivity: operator
suggested_source: Operator skill/access inventory + who pays
blocks_promotion: true
status: open
```

## ASCEND

- Blocked transition: idea without buyer → paid diagnostic booked.
- Thesis (scaffold): warm-network micro-audit if skills/access exist.
- Without declared assets: stay honest; do not invent opportunities list as OBSERVED.

Claims:

- Operator wants first cash. — `OBSERVED` (request)
- Warm network will pay. — `SPECULATIVE` / `NOT_COMPUTABLE` without evidence
- Completion proof: invoice within 14 days. — defined as requirement; feasibility NC until assets declared

## CLEAR

- Gate **G**: no fictional resources.
- Gate **C**: buyer not conflated with user without evidence.
- Gate **PROOF**: TESTABLE requires completion_proof string (present as criterion; feasibility still gated).
- Open DataRequest caps promotion quality until satisfied.

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, opportunity_mining, commercial, zero_option]
promotion_state: TESTABLE
authority: advisory_only
grants_execution: false
human_review_required: true
completion_proof: "first paid diagnostic invoice or signed SOW within 14 days"
proof_path:
  - declare skills/access/buyer
  - offer bounded diagnostic to warm contacts only
  - record cash or refusal
  - ledger proof_obtained or proof_failed
open_blocking_requests: [declared_skills_access_and_buyer]
```
