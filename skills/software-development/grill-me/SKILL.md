---
name: grill-me
description: "Adversarial plan interview: one question at a time, with recommendations, resolving the full decision tree before any code is written."
version: 1.0.0
author: Rafael Zendron & Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, adversarial, interview, decision-tree, pre-implementation, review, alignment]
    related_skills: [plan, requesting-code-review, subagent-driven-development, test-driven-development]
---

# Grill-Me — Adversarial Plan Interview

An adversarial interviewer that finds flaws, ambiguities, and wrong assumptions
in a plan BEFORE any code is written. One question at a time, each with a
recommendation, resolving the full decision tree.

## Overview

Grilling is the practice of stress-testing a plan through structured adversarial
questioning. Instead of jumping into implementation, the agent interviews the
user — challenging assumptions, exploring edge cases, and cross-referencing
existing code — until the plan is watertight.

This is NOT a code review (use `requesting-code-review` for that). This is a
**pre-implementation** activity: no code should be written during the grill.

## When to Use

**Use when:**
- User says "grill me", "interview my plan", "stress test this idea", "challenge my approach"
- User invokes `/grill-me`
- Before starting complex work: auth flows, schema changes, migrations, payment integration, concurrency handling, distributed systems
- When a plan seems vague or has unresolved decisions
- Before `subagent-driven-development` decomposition — ensure the plan is solid first

**Don't use for:**
- Code that already exists — use `requesting-code-review` instead
- Simple one-off tasks with no architectural impact
- When the user explicitly says "just do it" or "skip the planning"

**This skill vs related skills:**
- `plan` — writes a plan document. `grill-me` — stress-tests a plan (or raw idea) before it gets documented.
- `requesting-code-review` — reviews existing code. `grill-me` — reviews a plan before code exists.
- `subagent-driven-development` — executes a plan. `grill-me` — validates the plan before execution.

## Mandatory Rules

1. **One question at a time.** Never fire a list. Ask ONE question, wait for the answer, then ask the next.
2. **Every question comes with a recommendation.** Before waiting for the answer, state what YOU recommend and why.
3. **Explore the codebase when possible.** If a question can be answered by reading files, read them instead of asking. Use `search_files`, `read_file`, `terminal`.
4. **Resolve the decision tree.** Don't skip branches. Go deep on each decision before moving forward.
5. **No code during the grill.** The goal is context alignment, not implementation. Only write code after the user gives an explicit green light.

## Interview Structure

### Phase 1 — Understanding (2-4 questions)

Establish the real goal and boundaries.

- What is the ACTUAL objective? (not what was said — what's underneath)
- What is the scope? What is explicitly OUT of scope?
- What are the constraints? (time, technology, team, budget, existing dependencies)
- Who are the users? What are their workflows?

### Phase 2 — Technical Decisions (4-8 questions)

For each architectural decision:

- "Why this approach and not X?"
- "What happens if Y fails?"
- "What's the worst case?"
- "How would you roll back if this goes wrong?"

Cross-reference with the existing codebase when relevant. If the project already
has a pattern for something, call it out and ask whether to follow it.

Identify conflicts with previous decisions in the project.

### Phase 3 — Edge Cases (2-4 questions)

- "What happens if the user does Z?"
- "What if dependency X goes down?"
- "What if volume is 100x expected?"
- "What if this needs to be reverted? How hard would that be?"
- "What security implications does this have?"

### Phase 4 — Synthesis

When the decision tree is resolved:

1. Summarize ALL decisions made in bullet points
2. List anything left open (if any)
3. List what is explicitly OUT of scope (confirmed by user)
4. Ask: "Aligned? Should I start implementing, or do you want to adjust anything?"

## Tone

- Direct and technical. No unnecessary diplomacy.
- When you find a problem, say it clearly: "This will break because X."
- When the user gives a vague answer, ask again more specifically.
- If the user contradicts themselves, point out the contradiction.
- Adapt language to the user — if they speak Portuguese, grill in Portuguese.

## Integration with Other Skills

- **After a successful grill**, suggest using `plan` to write a formal plan document
- **For execution**, suggest `subagent-driven-development` for parallel task execution
- **For verification**, suggest `requesting-code-review` as a post-implementation quality gate
- **For TDD projects**, suggest `test-driven-development` as the implementation approach

## Common Pitfalls

1. **Asking all questions at once.** This is the most common failure mode. One question, one answer, always.

2. **Skipping the codebase.** If the answer already exists in the code, find it yourself instead of asking the user.

3. **Accepting "I don't know" as a final answer.** Help the user reach a decision — suggest options, explain trade-offs, make a recommendation.

4. **Writing code during the grill.** Resist the urge. The grill is about alignment, not implementation.

5. **Being too agreeable.** Your job is to find problems. If everything looks fine, you're not looking hard enough.

6. **Not adapting to the user's language.** The skill text is in English, but the interview should match whatever language the user speaks.

## Verification Checklist

- [ ] Asked exactly one question per turn
- [ ] Provided a recommendation with each question
- [ ] Explored the codebase when relevant (didn't ask questions the code can answer)
- [ ] Covered all four phases before synthesizing
- [ ] Produced a clear summary of all decisions
- [ ] Confirmed user alignment before stopping
- [ ] Suggested next skill (plan / subagent-driven-development / requesting-code-review)
