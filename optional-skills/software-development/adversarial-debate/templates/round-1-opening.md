# Adversarial Debate — Round 1: Opening Statement

You are **{{AGENT}}** participating in a structured adversarial debate.

## Topic

{{TOPIC}}

## Format

{{FORMAT}}

## Instructions

Construct your opening statement with the following strict JSON schema. Output ONLY valid JSON — no markdown, no commentary.

```json
{
  "agent": "{{AGENT}}",
  "round": 1,
  "position": "<FOR | AGAINST | CONDITIONAL>",
  "position_changed": false,
  "claims": [
    {
      "id": "{{AGENT}}-claim-1",
      "statement": "Short, falsifiable claim",
      "evidence": "What evidence supports this claim",
      "evidence_strength": "direct|inferred|speculative",
      "confidence": 75
    }
  ],
  "minority_report": null,
  "resolution": null
}
```

### Claim Rules

1. Each claim must be **falsifiable** — capable of being proven wrong by evidence.
2. At most 5 claims per agent per round.
3. `evidence_strength`:
   - `direct` = you have explicit data, code, docs, or test results
   - `inferred` = derived from existing evidence but not directly cited
   - `speculative` = logical reasoning without supporting data
4. `confidence` = integer 0–100 representing how sure you are.
5. Claims are indexed sequentially as `{{AGENT}}-claim-1` through `{{AGENT}}-claim-N` for cross-referencing in Round 2.

### Context

{{CONTEXT}}
