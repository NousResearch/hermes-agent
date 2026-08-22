# Minimal Model Contract + Deterministic Enrichment Pattern

## Problem

An LLM is asked to produce a complex structured response with internal fields
(cost estimates, risk scores, outcome tables, predicate grammar, epistemic
verdicts). The model fails on successive fields, causing a cascade:

```
field error → add prompt guidance → model fails on next field → more prompt → ...
```

This is the **field-by-field prompt-tuning trap**.

## Solution

Split the response into two parts:

### Part A — Minimal Model Contract (7 fields max)

The model produces only the fields it can reliably generate:

```json
{
  "hypothesis_id": "h-1",
  "proposal_type": "DISCRIMINATING_TEST",
  "operation_id": "source.search",
  "target_id": "fixture-root",
  "parameters": {"pattern": "needle"},
  "rationale": "search for the marker in the source tree",
  "falsification_question": "Is the marker present?"
}
```

Every other field is REJECTED as unknown.

### Part B — Deterministic Enricher

A pure function derives every complex field from trusted registries:

```
MinimalProposal → enricher.enrich() → FullProposal
```

Derivation sources:

| Derived field | Source |
|---------------|--------|
| `proposal_id` | Mission ID + counter |
| `estimated_cost` | Operation definition + phase + proposal type |
| `estimated_risk` | Operation `risk_class` |
| `expected_information_gain` | Phase + proposal type ONLY |
| `outcome_table` | Operation `evidence_type` templates |
| `reopen_condition_reference` | `null` (OBS/DISCRIM) or from ledger (REOPEN) |
| `semantic_fingerprint` | Content hash of operation/target/params/outcomes |

## Invariants

```
MODEL_GENERATED_PREDICATE_COUNT = 0
MODEL_GENERATED_COST_FIELDS = 0
MODEL_GENERATED_RISK_FIELDS = 0
MODEL_GENERATED_EPISTEMIC_FIELDS = 0
MODEL_GENERATED_OUTCOME_TABLES = 0
```

## Validation Pipeline

```
1. MinimalResponse.from_mapping()          → reject unknown fields
2. Validate correlation (request_id, mission_id, phase, count limits)
3. Validate against registries (operation, target, compatibility)
4. DeterministicEnricher.enrich()          → full WorkerProposal
5. Semantic fingerprint + dedup
6. Normalize (sort by fingerprint)
```

## When to Use

- Any component where an LLM is asked to produce structured data with
  fields that depend on internal system state (registries, ledgers, budgets)
- The LLM should produce WHAT to do, not HOW to evaluate it
- The system should be the authority on cost, risk, evidence interpretation
