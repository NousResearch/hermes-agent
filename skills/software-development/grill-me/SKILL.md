---
name: grill-me
description: "Adversarial plan interview before implementation."
version: 1.0.0
author: "Rafael Zendron (rafaumeu)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, adversarial, interview, decision-tree, pre-implementation, review, alignment]
    related_skills: [plan, requesting-code-review, subagent-driven-development, test-driven-development]
---

# Grill-Me Skill

Stress-tests a plan through structured adversarial questioning before any
code is written. One question per turn, each with a recommendation, resolving
the full decision tree until the plan is watertight.

## When to Use

- User says "grill me", "interview my plan", "stress test this idea"
- Before complex work: auth flows, schema changes, migrations, payments
- A plan has unresolved decisions or seems vague
- Before `subagent-driven-development` decomposition

Do NOT use for existing code (use `requesting-code-review`) or simple one-off
tasks.

## Prerequisites

None. The skill works on any plan or raw idea.

## How to Run

The agent loads the skill and enters interview mode. No special setup needed.

## Quick Reference

| Rule | Detail |
|------|--------|
| One question per turn | Never fire a list |
| Recommendation included | State your recommendation before waiting |
| Explore codebase first | Use `search_files`, `read_file`, `terminal` |
| No code during grill | Alignment only — code after explicit green light |

## Procedure

### Phase 1 — Understanding (2-4 questions)

Establish the real goal and boundaries.

- What is the ACTUAL objective?
- What is explicitly IN and OUT of scope?
- What are the constraints? (time, tech, team, budget)
- Who are the users?

### Phase 2 — Technical Decisions (4-8 questions)

For each architectural decision:

- "Why this approach and not X?"
- "What happens if Y fails?"
- "What's the worst case?"
- "How would you roll back?"

Cross-reference with the existing codebase using `search_files` and
`read_file`. If the project already has a pattern, call it out.

### Phase 3 — Edge Cases (2-4 questions)

- "What happens if the user does Z?"
- "What if dependency X goes down?"
- "What if volume is 100x expected?"
- "What security implications does this have?"

### Phase 4 — Synthesis

When the decision tree is resolved:

1. Summarize ALL decisions in bullet points
2. List anything left open
3. List what is explicitly OUT of scope
4. Ask: "Aligned? Should I start implementing, or adjust anything?"

## Pitfalls

1. **Asking all questions at once.** One question, one answer, always.
2. **Skipping the codebase.** Find the answer in code using Hermes tools instead of asking the user.
3. **Accepting "I don't know" as final.** Suggest options, explain trade-offs, make a recommendation.
4. **Writing code during the grill.** Alignment only — resist the urge.
5. **Being too agreeable.** Your job is to find problems. If everything looks fine, look harder.
6. **Not adapting to the user's language.** Interview in whatever language the user speaks.

## Verification

- [ ] Asked exactly one question per turn
- [ ] Provided a recommendation with each question
- [ ] Explored the codebase when relevant (used `search_files` / `read_file`)
- [ ] Covered all four phases before synthesizing
- [ ] Produced a clear summary of all decisions
- [ ] Confirmed user alignment before stopping
- [ ] Suggested next skill (`plan`, `subagent-driven-development`, or `requesting-code-review`)
