EVIDENCE-LABELED INVESTIGATION for:

$ARGUMENTS

For EVERY finding in this investigation, you MUST label it with one of three evidence levels. Do not present an inference as a fact. Do not present speculation as confirmed.

EVIDENCE LEVELS:
- **CONFIRMED** — You have exact file:line evidence from grep or file read. You can point to the specific code. This is a fact.
- **INFERRED** — You have indirect evidence (e.g., a pattern match, a type signature, an absence of something). The logic is sound but you haven't mechanically verified it. Say what the inference is based on.
- **UNVERIFIED** — You suspect this but have no evidence yet. Say what you would need to check to verify it.

RULES:
1. Every finding gets a label. No exceptions.
2. A finding labeled CONFIRMED must include the exact file path and line number.
3. A finding labeled INFERRED must state what evidence supports it AND what would upgrade it to CONFIRMED.
4. A finding labeled UNVERIFIED must state what grep/query/test would verify it.
5. Do NOT upgrade INFERRED to CONFIRMED without running the verification.
6. Do NOT drop UNVERIFIED items — they go in the "open questions" list.
7. If you catch yourself writing "likely" or "probably" — that's INFERRED, label it.
8. If you catch yourself writing "appears to" or "seems like" — that's UNVERIFIED, label it.

OUTPUT FORMAT:
For each finding:
[CONFIRMED|INFERRED|UNVERIFIED] Finding description
  Evidence: {file:line reference or reasoning}
  To verify: {what would confirm/upgrade this} (omit for CONFIRMED)

At the end, summarize:
- CONFIRMED findings: {N}
- INFERRED findings: {N}
- UNVERIFIED findings: {N}
- Overall confidence: {HIGH if >80% confirmed, MEDIUM if >50%, LOW if <50%}
