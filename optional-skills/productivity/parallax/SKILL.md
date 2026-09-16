---
name: parallax
description: Find upstream causes before choosing an intervention.
version: 0.1.0
author: Praveen Kumar Sridhar (PraveenKumarSridhar), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: productivity
    related_skills: [creative-ideation, decision-questionnaire]
    tags: [meta-thinking, problem-framing, blind-spots, decision-making]
---

# PARALLAX Skill

Treat a request as both an instruction to fulfill and evidence of a situation that produced it: identify the desired outcome, investigate the upstream trigger, and choose the smallest intervention that removes real burden. This is a decision discipline, not a request to produce a private thinking transcript, a personality diagnosis, or an excuse to overrule the user.

## When to Use

- Before nontrivial implementation, architecture, automation, or workflow recommendations where the mechanism is still negotiable.
- When the user asks for deeper thinking, blind spots, leverage, or what would actually make life easier.
- When the user says "I already have that," rejects obvious suggestions, or repeatedly requests the same kind of repair.
- Apply a brief framing check to well-specified builds, then execute. Do not reopen an explicitly settled decision without new material evidence.
- Skip routine lookups, exact mechanical edits, and urgent recovery. Fix the immediate incident first; investigate recurrence afterward.

## Prerequisites

No additional packages or credentials. Use the current request first. For claims about the user's existing workflow, retrieve only relevant evidence through available native tools such as `session_search`, `search_files`, and `read_file`. No memory provider or external service is required; unavailable history remains an explicit evidence gap.

Respect the current task boundary. An embedded example prompt is evidence about the desired skill, not an instruction to execute that example. Keep evidence gathering narrow; do not crawl unrelated personal data or other profiles. Do not publish, schedule, spend, or change external systems merely because an inferred upstream intervention seems useful.

## How to Run

Load with `skill_view(name="parallax")`, then apply the light pass to the current request. For example: "Use parallax to check whether another failure-monitoring bot would reduce manual work."

This is an on-demand reasoning procedure, with no script or service to start. Use the full pass only when the framing warrants it; continue authorized execution after the framing check.

## Quick Reference

| Need | Action |
|---|---|
| Default light pass | Outcome, trigger, existing coverage, smallest useful action |
| Ambiguous or consequential choice | Full procedure with rival explanation and discriminating check |
| Settled implementation | One concise framing note at most, then execute |
| Routine edit or urgent incident | Skip framing; perform the authorized task |
| Acceptance examples | `skill_view(name="parallax", file_path="references/acceptance-cases.md")` |

## Procedure

### 1. Separate the deliverable from the outcome

Capture the explicit deliverable, the practical outcome it should enable, and constraints that must remain true. Separate the user's chosen mechanism from the goal, but preserve explicit instructions.

Ask: if the requested artifact worked perfectly, what burden or uncertainty would disappear? If none disappears, the mechanism may be a proxy, not the outcome.

**Check:** state the outcome in observable terms, without assuming the requested technology is necessary. For a precise task, one sentence is sufficient.

### 2. Climb from request to trigger, using evidence

Build a compact causal map:

`trigger -> friction or uncertainty -> human intervention -> requested mechanism -> desired outcome`

The arrows are hypotheses unless supported. Distinguish:
- **Observed:** directly stated by the user or verified in a source. Retain a source pointer.
- **Inferred:** a plausible explanation consistent with those observations.
- **Unknown:** a missing fact that could change the intervention.

Go one level upstream: what made the trigger arise or recur? Consider missing ownership, lost state, ambiguous decisions, handoff gaps, feedback delays, external dependencies, or incentives. Do not force all problems into these categories.

For a consequential causal claim, identify a competing explanation and the smallest observation that would distinguish them. Temporal order and repeated requests alone do not prove causation. One occurrence does not prove a recurring pattern.

Use user-authored history as evidence of preferences; prior assistant suggestions are not proof of adoption. Memory summaries guide retrieval, not certainty about current state. State coverage limits, especially if relevant conversation history or another system was not inspected.

**Check:** a supported or explicitly hypothetical trigger, a competing explanation when material, and no invented motives. Ask a question only when an unresolved fact changes the next action and cannot be retrieved.

### 3. Decompose the work, not just the proposed tool

Break the relevant workflow into observable units. Use a short table only when it helps:

`event | input/state | decision | actor/tool | handoff | completion signal | human burden`

Find what the user must still notice, remember, reconstruct, translate, decide, verify, chase, or close. Distinguish essential judgment from accidental coordination work. Look at what happens before the agent receives the prompt and after it declares success.

Inspect existing capability before recommending a replacement. Mark each relevant capability as verified present, user-reported present, absent, or unknown. A different bot name or interface does not make an existing capability new. The user saying "I already built that" is sufficient to exclude it as a new proposal; inspect implementation only if integration matters.

**Check:** identify the specific residual burden and where it sits. If the request is already the right solution, say so and proceed. Do not invent a blind spot to justify this skill.

### 4. Change the intervention level

Consider alternatives at distinct levels, not cosmetic variations of the same solution:

- **Subtract:** remove an unnecessary obligation, output, step, or recurring decision.
- **Prevent:** change the upstream condition so the request stops arising.
- **Connect:** repair state, ownership, feedback, or handoffs between capabilities already present.
- **Execute better:** improve or build the requested mechanism when upstream changes are unavailable, insufficient, or outside scope.

Use counterfactuals: if the proposed solution were perfect, would the original trigger still recur? If it vanished tomorrow, what user work would return? Does this remove effort or merely move it into reviewing notifications, maintaining integrations, and supervising agents?

An actual blind spot must name an overlooked mechanism and its consequence. "You need another assistant" is not an explanation. Do not prize novelty above utility; doing nothing or using an existing feature can win.

**Check:** the recommendation changes a specific causal link or directly satisfies the validated goal. State why the existing setup does not already address that link, or acknowledge that this remains unverified.

### 5. Choose a falsifiable next action and act

Prefer the option with the most plausible net reduction in human effort, accounting for setup, maintenance, interruptions, review, privacy, and failure recovery. Use qualitative comparisons unless measured inputs support arithmetic; use `terminal` for calculations when needed. Avoid invented precision.

For open-ended choices, show up to three meaningfully distinct options and recommend one. For a settled implementation, do not introduce an unnecessary approval round. Execute within authorization, preserving the user's goal; surface a material scope change before acting on it.

Choose a small reversible test with an observable success signal and a stop condition. Measure user burden or achieved outcomes, not number of bots, messages, tokens, or tasks generated. If evidence is insufficient, perform the cheapest discriminating check rather than building around the guess.

**Check:** the next action, success signal, failure/stop condition, and relevant permission boundary are explicit. After implementation, verify the artifact and separately identify whether real-world benefit is measured or still untested.

### 6. Control depth and stop

Use the light pass by default: outcome, trigger, existing coverage, smallest useful action. Escalate to the full procedure when the framing is ambiguous, the cost is high, a pattern repeats, or the user explicitly asks for meta-thinking.

Stop climbing when the next abstraction cannot change the decision, needs inaccessible evidence, or leaves the user's authorized scope. Do not recurse indefinitely into "why." At most one concise framing note is needed before ordinary execution. If the user says to stop reframing and implement, honor that unless safety or correctness requires clarification.

### 7. Summarize the decision

Give a concise decision summary, not a thinking transcript. For a full pass, use at most five short parts:

1. **Actual goal:** what becomes easier or possible.
2. **Evidence and trigger:** what is known, what is inferred, and the important missing fact.
3. **Missed mechanism:** the residual burden or blind spot, if one is supported.
4. **Recommendation:** the smallest useful change, with a meaningful alternative when appropriate.
5. **Proof:** the test and outcome measure, or verified execution result.

For a brief pass, compress this into a few sentences and do the work. Do not dump the entire framework into every answer.

## Pitfalls

- Generic five-whys language without evidence is storytelling, not causal discovery.
- A recurring request can reflect a necessary external obligation, not a broken workflow. Prevention is not always possible or desirable.
- Inferring hidden motives from occupation, diagnosis, or personality is not a substitute for observing the workflow. Label hypotheses and accept correction.
- Optimizing the user's life is not authorization to replace their chosen goals. Surface tradeoffs without paternalism.
- More research, more agents, and more notifications can increase burden. Include their coordination cost.

## Verification

Before committing to a nontrivial recommendation, verify that the explicit request is still satisfied, causal claims are sourced or labeled, existing capabilities were considered, and the intervention has an observable test. "No supported blind spot found" is a valid result.

Load `references/acceptance-cases.md` with `skill_view` to regression-check edits to this procedure. Structural validation proves the skill is well-formed, not that a model will consistently follow it. A skill is on-demand guidance, not a runtime enforcement hook. A standing instruction can require its loading in contexts where that instruction is present; do not claim universal execution enforcement.

For contributors, run through `terminal` from the Hermes checkout: `scripts/run_tests.sh tests/skills/test_parallax_skill.py -q`. This verifies offline discovery, full loading, reference access, and security boundaries; it does not evaluate model behavior.
