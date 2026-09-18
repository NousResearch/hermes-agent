# Adversarial Debate — Round 3: Convergence

You are **{{AGENT}}**. This is the final round. The DQI after Round 2 was below threshold — there are still unresolved disagreements.

## Instructions

Review all unresolved tensions (claims that were defended in Round 2). You MUST end with one of:

- **accept** — you agree with the consensus path forward
- **conditional_accept** — you accept if a specific condition is met
- **minority_report** — you maintain your dissenting position with a written rationale

Output ONLY valid JSON:

```json
{
  "agent": "{{AGENT}}",
  "round": 3,
  "position": "<FOR | AGAINST | CONDITIONAL>",
  "position_changed": true,
  "claims": [
    {
      "id": "{{AGENT}}-claim-1",
      "statement": "Refined final position",
      "evidence": "Summary of best evidence",
      "evidence_strength": "direct|inferred|speculative",
      "confidence": 85,
      "rebuttal_status": "unrebutted|conceded|defended",
      "rebuttal_of": null,
      "rebuttal_notes": null
    }
  ],
  "minority_report": "<only if resolution=minority_report>",
  "resolution": "accept|conditional_accept|minority_report"
}
```

### Context

{{CONTEXT}}

---

## Prior Rounds Transcript

```json
{{TRANSCRIPT}}
```
