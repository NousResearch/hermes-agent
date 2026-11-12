---
id: 08-evidence-intelligence
scenario: competitive facts with public find + private CRM gap
profiles: [core, evidence_intelligence, commercial]
research_mode: online
expected_promotion_max: TESTABLE
---

# Transcript: Evidence intelligence — find and request

## OPEN

- **Request:** Competitive landscape and whether we can price against peers.
- **Decision-critical:** public competitor positioning; private CRM pipeline truth.
- **Authority:** advisory_only; research may draft, not contact.

## ALIGN

- Source stack: operator files → workspace → live host tools → model prior as SPECULATIVE only.
- Gap-detect: public comps (findable); CRM conversion rates (private).

### Research plan (public)

1. Query competitor public sites / pricing pages when category known.
2. Normalize into evidence items with URL, retrieval time.
3. Label claims; never invent OBSERVED.

### DataRequest (private)

```yaml
field: crm_pipeline_conversion
why_decision_critical: Private conversion rates required for non-speculative pricing scenarios
sensitivity: private
suggested_source: Operator CRM export (no scraping)
blocks_promotion: true
status: open
```

## ASCEND

- Evidence items: empty until tools return rows; tooling_gaps recorded if search unavailable.
- Claim ledger separates person / company / model inference.
- Competitive scan scaffold only; firm market size remains NOT_COMPUTABLE without cites.

Claims:

- Live research is attempted when tools exist. — `OBSERVED` (protocol) / research_log
- Our win rate is 40%. — **forbidden** without CRM source → DataRequest + `NOT_COMPUTABLE`
- Public peer lists a price band. — `OBSERVED` only with URL cite; else `NOT_COMPUTABLE`

## CLEAR

- Gate **P**: public gap requires attempt when tools available.
- Gate **Q**: private CRM fact requires DataRequest (emitted).
- Gate **R**: no silent OBSERVED from model prior on private facts.
- Gate **B**: no forecast-as-fact market sizes.

## SEAL

```yaml
status: PROPOSED
profiles_loaded: [core, evidence_intelligence, commercial]
promotion_state: TESTABLE
authority: advisory_only
grants_execution: false
human_review_required: true
open_blocking_requests: [crm_pipeline_conversion]
research_attempts:
  - query: "public competitor pricing for declared category"
    outcome: tool_dependent
not_computable_fields: [private_conversion_rate, uncited_market_size]
completion_proof: "Cited public comps + satisfied CRM DataRequest or explicit waiver"
```
