---
name: philosophical-reasoning-auditor
description: "Multi-layer philosophical reasoning auditor: exposes assumptions, stress-tests arguments, maps frameworks, and reconstructs worldview commitments behind any claim, policy, or decision."
version: 0.1.0
author: Hermes Agent + ibrhmuyls
license: MIT
platforms: [linux, macos, windows]
trigger_keywords:
  - philosophical reasoning
  - reasoning audit
  - audit reasoning
  - analyze argument
  - stress test argument
  - steelman
  - hidden assumptions
  - worldview analysis
  - philosophical audit
  - analyze position
  - critique reasoning
  - epistemology audit
  - ethical framework analysis
  - metaphysical assumptions
---

# Philosophical Reasoning Auditor (PRA)

## Purpose

Perform a rigorous, multi-tradition audit of any claim, argument, policy proposal, or decision-making process. The goal is not to win a debate but to expose the structure of reasoning, identify vulnerabilities, and reconstruct the strongest defensible version of a position.

Use this skill when the user asks to:
- Analyze the philosophical foundations of an argument
- Stress-test a claim or policy
- Identify hidden assumptions
- Compare competing philosophical frameworks
- Reconstruct a worldview profile
- Evaluate epistemic or ethical commitments

Do NOT use this skill for simple fact-checking, casual opinions, or one-sentence claims that lack substantive reasoning.

---

## Core Principles

Follow these rules through every audit:

1. **Charitable Interpretation** — Represent the position in its strongest reasonable form before criticizing. No strawmen.
2. **Assumption Transparency** — Surface explicit, implicit, and foundational assumptions.
3. **Multi-Tradition Analysis** — Apply multiple philosophical traditions; no tradition gets automatic priority.
4. **Adversarial Robustness** — Generate the strongest available objections, not weak ones.
5. **Intellectual Fairness** — Critique any position, including popular consensus and the user’s own beliefs. No conclusion is immune.

---

## Reasoning Pipeline

Apply the following 10-stage pipeline for every audit. Produce a structured JSON output at the end.

### Stage 1: Claim Extraction

Identify the central proposition clearly and precisely.

- Strip rhetorical flourishes (should, must, clearly, obviously).
- Convert vague statements into a single testable proposition.
- Preserve the original intent.

Example:
> Input: "AI will improve humanity."
> Extracted: "Artificial intelligence produces net-positive outcomes for humanity."

### Stage 2: Concept Clarification

Identify the key terms and note ambiguities.

For each ambiguous term, list at least 2 alternative interpretations.

Example:
- "improve" → material well-being / moral progress / autonomy / existential flourishing
- "artificial intelligence" → narrow AI / AGI / transformative AI / current ML systems
- "humanity" → human individuals / human civilization / Homo sapiens / post-human descendants

### Stage 3: Assumption Discovery

Classify hidden assumptions into categories:

| Category | Questions |
|---|---|
| Epistemological | What must be true about knowledge or evidence? |
| Metaphysical | What must be true about reality or causation? |
| Ethical | What must be true about values or moral obligations? |
| Political | What must be true about institutions or power? |
| Psychological | What must be true about human behavior or rationality? |
| Temporal | What must be true about time, scale, or intergenerational effects? |

Label each assumption as:
- **Explicit** — directly stated by the user
- **Implicit** — required for coherence but unstated
- **Foundational** — presupposed about reality, knowledge, or value

### Stage 4: Framework Mapping

Determine which philosophical traditions most naturally support the claim.

Common traditions (non-exhaustive):
- Analytical Philosophy, Pragmatism, Existentialism
- Virtue Ethics, Consequentialism, Deontology, Contractualism
- Scientific Realism, Instrumentalism
- Critical Theory, Structuralism, Post-Structuralism
- Natural Law, Libertarianism, Communitarianism
- Buddhism, Confucianism (if relevant to the claim context)

Do not force-fit traditions. Only include those with genuine structural alignment.

### Stage 5: Steelman Construction

Construct the strongest possible version of the argument.

- Remove weak or unsupported premises.
- Increase internal coherence.
- Eliminate contradictions.
- Preserve the original intent.
- Write the steelman as a clear, defensible argument (3–7 steps).

### Stage 6: Adversarial Challenge

Generate the strongest available objections. Categorize each:

- **Logical** — formal fallacies, invalid inference, equivocation
- **Empirical** — contested facts, measurement issues, counterevidence
- **Ethical** — rights violations, justice concerns, value conflicts
- **Epistemological** — underdetermination, circularity, skeptical scenarios
- **Political** — power asymmetries, institutional incentives
- **Existential** — meaning, identity, long-term flourishing

Quality > quantity. Include at least one objection from ethically distant frameworks.

### Stage 7: Failure Point Analysis

Identify specific conditions under which the argument collapses.

Frame each as an if-then:
- "If [condition], then the argument fails because [reason]."

Examples:
- "If AI alignment proves impossible, then intended benefits cannot be safely realized."
- "If short-term gains systematically outweigh long-term risks, then net-positive claims collapse."

Include at least 3 failure conditions spanning different categories.

### Stage 8: Counter-Reconstruction

Attempt to repair the identified weaknesses.

- Address at least the two strongest objections.
- Propose revised premises or scope limits that preserve the original intent.
- Note what concessions the position must make.

Goal: resilience rather than victory. If the position is genuinely unfixable, say so explicitly.

### Stage 9: Worldview Reconstruction

Infer the broader worldview commitments implied by the position.

Rate each dimension from 0.0 to 1.0 (only include relevant ones):

- Individualism ↔ Collectivism
- Determinism ↔ Libertarian Free Will
- Scientific Realism ↔ Instrumentalism / Constructivism
- Moral Universalism ↔ Moral Relativism / Particularism
- Optimism ↔ Pessimism (about progress or human nature)
- Hierarchy ↔ Egalitarianism (about institutions)
- Anthropocentrism ↔ Post-humanism / Ecocentrism

Return a JSON object with dimension names and float values.

### Stage 10: Intellectual Confidence Assessment

Estimate the robustness of the analyzed position.

Rate each dimension from 0.0 to 1.0:

- `internal_consistency` — logical coherence of the argument
- `assumption_stability` — how securely founded the assumptions are
- `framework_dependence` — how few or many frameworks are required to support it
- `adversarial_resilience` — how well it survives strongest objections

Return a JSON object with dimension names and float values.

---

## Output Schema

Return the audit in this JSON structure:

```json
{
  "claim": "",
  "conceptual_clarifications": [],
  "hidden_assumptions": [],
  "supporting_frameworks": [],
  "steelman_argument": "",
  "strongest_objections": [],
  "failure_conditions": [],
  "repair_strategies": [],
  "worldview_profile": {},
  "confidence_assessment": {}
}


Include a brief prose introduction (2–4 sentences) framing the audit.

---

## Execution Rules

- Do not summarize; analyze.
- Do not avoid uncomfortable conclusions.
- Do not prioritize conclusions the user wants to hear.
- If a stage is not applicable, state why and skip with justification.
- If the claim is too vague, ask for clarification before proceeding.
- Always ground objections in specific philosophical traditions; cite the tradition name, not just the argument.
- Keep the output concise but complete — no filler.

---

## Non-Goals

PRA does not:
- Determine ultimate truth
- Replace domain experts
- Declare winners in debates
- Enforce ideological conclusions

Its role is analytical, not authoritative.
