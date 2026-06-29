# Confidence Gate

## Principle

When generating tests, scaffolding fixtures, classifying risk, or proposing any non-trivial test artifact, emit a confidence assessment before writing code. If confidence is below the threshold, stop and ask the user instead of generating plausible-looking output built on guesses.

## Rationale

The failure mode of LLM-generated tests is rarely "refused to try" — it is "generated something plausible that passes locally and breaks silently in CI." Hallucinated selectors, invented endpoint paths, fabricated risk scores, and reverse-engineered schemas all produce code that looks correct and tests nothing real. A confidence gate makes that failure mode loud by forcing the agent to declare its evidence and its unknowns before any artifact is committed.

## Required output shape

Every non-trivial test artifact proposal must include:

```
Confidence: <1-10>
Rationale: <one or two sentences citing concrete evidence from the repo or contract>
Unknowns: <bulleted list of things the agent does not know>
```

The Rationale must cite a file path, a contract document, an existing pattern, or a captured observation. Vague rationale ("based on standard patterns", "looks similar to other tests") is not evidence and forces the score down.

## Threshold rule

- **Confidence ≥ 7** — proceed with generation.
- **Confidence 5–6** — proceed but surface the assumptions to the user in the output so they can correct mid-flight.
- **Confidence < 5** — STOP. Do not generate. Ask the user to resolve the most-blocking Unknown first.

## When to apply

Apply the gate when generating or proposing:

- **Selectors and page objects.** Must have explored the live application via `playwright-cli` or read existing page object patterns. Confidence < 5 if neither.
- **Endpoint paths and request shapes.** Must have read the OpenAPI / Swagger contract or existing endpoint enums. Confidence < 5 if the endpoint is being invented.
- **Risk classification (test-design, NFR).** Must cite probability and impact evidence. Confidence < 5 if scoring is vibes-based.
- **Fixture composition.** Must understand existing `mergeTests` patterns and fixture boundaries in the repo. Confidence < 5 if composing blindly.
- **Schema authoring (Zod, Ajv, JSON Schema).** Must have a documented contract source (OpenAPI, JSON schema, existing schema file). Confidence < 5 if reverse-engineering from a single sample response.
- **Data factories.** Must understand the production data shape and constraints. Confidence < 5 if guessing field validity rules.

## When NOT to apply

- Mechanical refactors with clear scope (rename a variable, add a tag, update an import).
- Reading or summarizing existing artifacts.
- Producing reports from already-gathered data.
- Trivial test additions that copy an existing pattern exactly.

The gate exists to prevent fabrication, not to bureaucratize obvious work.

## Anti-patterns

❌ **Vanity scores.** `Confidence: 9` with no Rationale, or Rationale that does not cite evidence. Score the evidence, not the optimism.

❌ **Listing then ignoring Unknowns.** Listing unknowns and then proceeding anyway when Confidence is below threshold. If the gate is below threshold, the only valid next action is to ask the user.

❌ **Asking generically.** Asking "should I proceed?" instead of resolving the most-blocking Unknown with a concrete one-sentence question.

❌ **Inflating to clear the bar.** Adjusting Confidence upward to avoid the stop rule. If the evidence is weak, the score is weak; resolve the evidence, not the number.

## Patterns that work

✅ **Cite the source.** "Confidence: 8 — Rationale: read `src/openapi/users.yaml` line 142-167 and existing schema at `tests/api/users.schema.ts`."

✅ **One concrete Unknown.** When below threshold, ask one specific question: "Is `POST /users/{id}/role` documented anywhere? I can't find it in the OpenAPI spec and there are no existing tests for it."

✅ **Promote evidence.** When the user answers the Unknown, the Rationale gets stronger and Confidence rises legitimately. The gate is a feedback loop, not a checkpoint.

## Related fragments

- `test-quality.md` — Definition of Done for tests; the gate protects DoD compliance.
- `risk-governance.md` — risk scoring discipline that informs Rationale for risk-related gates.
- `probability-impact.md` — scoring scales used in risk-related Rationale.
- `selector-resilience.md` — selector confidence specifically.
- `playwright-cli.md` — the sanctioned exploration tool that promotes selector Confidence.
