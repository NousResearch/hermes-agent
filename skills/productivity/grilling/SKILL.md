---
name: grilling
description: "Use when the user wants a relentless one-question-at-a-time interview to stress-test a plan or design before implementation, or uses grill/grilling trigger phrases."
version: 1.0.0
author: Matt Pocock + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [planning, review, critique, design, interview]
    homepage: https://www.aihero.dev/skills/grilling
    upstream_skill: https://github.com/mattpocock/skills/tree/main/skills/productivity/grilling
    related_skills: [grill-me, grill-with-docs, plan]
---

# Grilling

## Overview

`grilling` is a relentless interview for stress-testing a plan or design before anything gets built. The goal is shared understanding: walk the plan as a tree of decisions, resolve dependencies between decisions in order, and make implicit assumptions explicit.

The interview asks exactly one question at a time, waits for the user's answer, and includes the agent's recommended answer with each question so the user can react to a concrete proposal rather than start from a blank page.

This skill adapts Matt Pocock's `mattpocock/skills` v1.1 grilling primitive for Hermes.

## When to Use

Use this skill when:

- The user explicitly asks to be grilled, challenged, pressure-tested, or interviewed about a plan.
- The user invokes `/grilling` or `/grill-me`-style language.
- A plan or design feels plausible but has unresolved decisions, hidden assumptions, or unclear tradeoffs.
- You need to harden a design before writing a spec, tickets, or code.

Do not use this skill when:

- The user wants immediate implementation and the plan is already specific enough to execute.
- The user asks for a broad critique/report instead of an interactive interview.
- You can answer the question directly from available facts without needing user decisions.

## The Interview Loop

Repeat this loop until the user confirms shared understanding:

1. **Map the next decision.** Identify the most upstream unresolved decision whose answer changes downstream choices.
2. **Separate facts from decisions.**
   - **Facts** are things you can find by reading code, docs, tickets, prior context, or other available sources.
   - **Decisions** are things the user needs to choose: scope, tradeoffs, constraints, priorities, acceptable risk, product intent.
3. **Research facts before asking.** If a fact can be found with available tools, look it up instead of asking the user.
4. **Ask one decision question.** Ask exactly one question, in dependency order.
5. **Recommend an answer.** Include your recommended answer and a short reason.
6. **Wait.** Do not ask the next question, write the spec, create tickets, or implement until the user answers.

## Question Format

Use this format for each turn:

```markdown
Question: <one decision question>

My recommendation: <specific recommended answer>

Why: <one or two concise reasons>
```

Keep the question singular. If you are tempted to ask multiple questions, choose the parent decision first and defer dependent branches until after the user answers.

## Completion Gate

When the decision tree appears resolved, summarize the shared understanding and ask for explicit confirmation:

```markdown
I think we have shared understanding now:
- <settled decision>
- <settled decision>
- <remaining assumption, if any>

Do you agree, or should I keep grilling?
```

Do not enact the plan until the user confirms shared understanding.

## Common Pitfalls

1. **Question dumps.** A list of questions destroys the tree structure. Ask one question and wait.
2. **Self-grilling.** Do not explore and answer decisions by yourself. Research facts yourself; put decisions to the user.
3. **Premature implementation.** The interview is not complete until the user confirms shared understanding.
4. **Vague recommendations.** The user should react to a concrete proposal. Give a real recommended answer, not "it depends."
5. **Skipping dependency order.** Ask parent decisions before child decisions so answers can prune the tree.

## Verification Checklist

- [ ] Exactly one decision question was asked in the latest turn.
- [ ] Any answerable facts were researched instead of pushed to the user.
- [ ] The question includes a concrete recommended answer.
- [ ] No implementation/spec/ticket work starts before explicit confirmation.
- [ ] The final shared-understanding summary is confirmed by the user.
