# Adversarial Debate — Round 2: Cross-Examination

You are **{{AGENT}}**. The full transcript from Round 1 is included below.

## Instructions

Read every claim from every other agent. For each claim you disagree with or find unsupported:

1. **Rebutt** — cite the claim ID and explain why it is wrong, incomplete, or speculative.
2. **Concede** — acknowledge claims by other agents that are well-supported.
3. **Update your confidence** — may increase, decrease, or stay the same.
4. **Optionally change position** — if evidence compels it, set `position_changed: true`.

Output ONLY valid JSON following this schema:

```json
{
  "agent": "{{AGENT}}",
  "round": 2,
  "position": "<FOR | AGAINST | CONDITIONAL>",
  "position_changed": true,
  "claims": [
    {
      "id": "{{AGENT}}-claim-1",
      "statement": "Updated claim (keep original or refine)",
      "evidence": "Same or updated evidence",
      "evidence_strength": "direct|inferred|speculative",
      "confidence": 80,
      "rebuttal_status": "unrebutted|conceded|defended",
      "rebuttal_of": null,
      "rebuttal_notes": null
    },
    {
      "id": "{{AGENT}}-claim-2",
      "statement": "A new or refined claim",
      "evidence": "Evidence",
      "evidence_strength": "direct",
      "confidence": 90,
      "rebuttal_status": "unrebutted",
      "rebuttal_of": "clio-claim-3",
      "rebuttal_notes": "Clio's claim ignores the latency overhead of ..."
    }
  ],
  "minority_report": null,
  "resolution": null
}
```

### Rebuttal Rules

- `rebuttal_of` = the claim ID you are rebutting (or `null` for original claims).
- `rebuttal_status`:
  - `unrebutted` — this is your own claim that no one has challenged
  - `conceded` — you accept another agent's rebuttal and lower your confidence
  - `defended` — you maintain your position despite the rebuttal
- Rebuttals MUST target valid claim IDs from Round 1 (format: `<agent>-claim-<N>`).

### Context

{{CONTEXT}}

---

## Prior Rounds Transcript

```json
{{TRANSCRIPT}}
```
