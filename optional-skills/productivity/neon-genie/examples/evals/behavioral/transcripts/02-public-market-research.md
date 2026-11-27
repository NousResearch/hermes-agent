---
id: behavioral-02-public-market-research
scenario: public market fact — must attempt research or record unavailability
profiles: [core, commercial, evidence_intelligence]
research_mode: online
expected_promotion_max: TESTABLE
---

# Behavioral: public market research attempt

## OPEN

- **Request:** Current public market price band for similar developer tools.
- **Authority:** advisory_only.
- **Sensitivity:** public (fetchable).

## ALIGN

- Gap is public and decision-useful → Gate **P** requires research attempt when tools exist.
- Plan: search public pricing pages / directories; cite or drop.

### Research loop

```text
GAP_DETECT → QUERY_PLAN → FETCH → CITE → LABEL
```

### Research attempt

```yaml
research_attempts:
  - query: "developer tools SaaS public pricing examples 2026"
    tool: host_web_search
    outcome: attempted
    note: "Record results with URL+date as OBSERVED; if tools unavailable, mark NOT_COMPUTABLE with attempt"
```

If host tools are **unavailable**, record:

```yaml
research_attempts:
  - query: "developer tools SaaS public pricing examples 2026"
    tool: none
    outcome: unavailable
    note: "Host research tools not present this run"
```

## ASCEND

Claims:

- Operator asked for public market pricing band. — `OBSERVED`
- At least one comparable public list price was retrieved. — `OBSERVED` if cited; else `NOT_COMPUTABLE` after attempt
- Our product should charge the median. — `SPECULATIVE` (inference; needs buyer fit)

## CLEAR

- Gate **P**: public gap + tools available without attempt → fail (avoided: attempt recorded).
- No fabricated OBSERVED from model prior alone.

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, commercial, evidence_intelligence]
promotion_state: MAPPED
authority: advisory_only
grants_execution: false
research_attempts:
  - query: "developer tools SaaS public pricing examples 2026"
    outcome: attempted
human_review_required: true
```
