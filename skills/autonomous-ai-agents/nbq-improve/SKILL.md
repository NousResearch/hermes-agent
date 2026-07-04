---
name: nbq-improve
description: >
  Use to run one improvement iteration on the next-best-questions skill: review its learnings
  journal, research opportunities, pre-register a cost-aware experiment, build off-default, evaluate
  Δresult AND Δcost, and journal the verdict either way. Triggers: 'improve nbq', 'run the nbq loop',
  'next-best-questions iteration'. There are NO scripts; this protocol is agent-executed.
version: 1.0.1
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [self-improvement, experiment-protocol, pre-registration, cost-aware, meta]
    related_skills: [next-best-questions]
---

# NBQ Improve — standing improvement protocol

Run exactly one evidence-led improvement iteration on `next-best-questions`, then leave the next
iteration ready to start. This v1 is docs-only and agent-executed; it has no scripts.

1. **REVIEW**
   - Read the target skill's `README.md`, especially its key-learnings journal and §5,
     "Methodology learnings."
   - Read the target skill's `references/design-decisions.md`, including its closed-experiment list.
   - Read this skill's `references/backlog.md`. Although the target is `next-best-questions`, the
     living backlog physically lives in THIS skill at `nbq-improve/references/backlog.md`, not under
     `next-best-questions/`.
   - Drift-check every relevant documentation claim about versions, flags, and counts against the
     actual code and configuration before relying on it.
   - Never re-open a closed experiment without a NEW hypothesis that states specifically why the old
     verdict no longer applies.

2. **RESEARCH**
   - Run a bounded sweep of published literature and the repository's agentic-workflow ecosystem in
     which `next-best-questions` is embedded: `relentless-solve`'s clarify step, the `investigator`
     skill's routing, and `task-decomposer`.
   - Give every candidate a hypothesis, its expected mechanism, and the CHEAPEST falsifying test
     capable of killing it.
   - Update the backlog and rank candidates by expected value ÷ evaluation cost.

3. **PLAN, TEST-FIRST**
   - Select no more than the two highest-EV backlog items.
   - Before any build, complete `references/preregistration-template.md` for each item: define the
     smoke test, targeted tests, gate arms and n, primary metric, and mechanical ADOPT rule.
   - Pre-register the efficiency budget too: expected Δtokens and Δwall per run, plus a ceiling past
     which even a result win is no-adopt or "adopt with the knob off by default."

4. **BUILD OFF-DEFAULT**
   - Send implementation through `codex-worker`; never implement in the main planning loop.
   - Preserve the absent-key convention so existing harness configurations remain byte-identical
     when the new knob is untouched.
   - Add a selector and rollback flag. Add unit tests, including an explicit inert-by-default pin
     test.

5. **EVALUATE**
   - Run smoke → targeted → gate in that order.
   - Measure Δresult AND Δcost; capture time, tokens, and calls printed by the harness for each arm.
   - Apply `references/verdict-rubric.md` mechanically. Borderline results are no-adopt: by the #28
     precedent, a directionally right result that is not a broad win stays off.

6. **JOURNAL**
   - Update the target `README.md` key-learnings journal for either outcome, win or negative result.
     Record the full verdict in the target's `references/design-decisions.md`, then close or re-scope
     the item in this skill's `references/backlog.md`.
   - Treat negative results as products and document them with the same rigor as wins.
   - Follow the commit-message contract defined fully in `references/verdict-rubric.md`: every
     iteration commit states, verbatim and specifically, ATTEMPTED / WHY / RESULT. For a verdict,
     include gate numbers, Δresult, Δcost, and verdict. For a pre-gate commit, RESULT may say
     "build stage, gate pending," but a separate verdict commit must follow after the gate.
   - Make the commit log readable as the experiment ledger on its own, without external narration.

7. **LOOP**
   - Return to step 1.
   - Exit the session only after verdicts are landed, the journal is written, and the backlog is
     refreshed with ranked candidates for the next iteration.

## House rules

- The scoring formula is frozen; do not touch it casually.
- New features ship off-by-default behind selectors.
- Pre-register gates before running them and apply them mechanically, never through post-hoc
  rationalization.
- The OBJECTIVE outcome harness gates anything touching elicitation or generation; proxy harnesses
  are not sufficient for those changes.
- Implementation is always via `codex-worker`; workers never commit.
- The main loop reviews, gates, journals, and ships; it does not write the experiment code itself.
