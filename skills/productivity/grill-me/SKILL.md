---
name: grill-me
description: "Use when the user explicitly asks /grill-me or wants a stateless relentless interview that stress-tests a plan or design without writing docs."
version: 1.0.0
author: Matt Pocock + Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [planning, review, critique, design, interview]
    homepage: https://www.aihero.dev/skills/grill-me
    upstream_skill: https://github.com/mattpocock/skills/tree/main/skills/productivity/grill-me
    related_skills: [grilling]
---

# Grill Me

## Overview

`grill-me` is the stateless front door to the `grilling` interview. It stress-tests a plan or design by walking the decision tree one branch at a time until the user and agent reach shared understanding.

Use it when the user wants the soft spots in a plan forced into the open, but does not want durable artifacts such as ADRs, glossary entries, specs, or tickets created yet.

This skill adapts Matt Pocock's `mattpocock/skills` v1.1 `/grill-me` wrapper for Hermes.

## When to Use

Use this skill only when:

- The user explicitly says `/grill-me`, "grill me", "pressure-test this", or similar.
- The user wants an interactive pre-build interrogation of a plan or design.
- The desired output is sharpened shared understanding in the conversation itself.

Do not use this skill when:

- The user wants docs written while decisions are made — use a stateful docs/ADR workflow instead.
- The user asks for a one-shot critique, review, or audit rather than an interview.
- The user asks you to implement immediately.

## Behavior

Run a `grilling` session:

1. Walk the plan as a decision tree.
2. Resolve parent decisions before child decisions.
3. Research factual questions with tools when possible.
4. Put actual decisions to the user.
5. Ask exactly one question at a time.
6. Include your recommended answer with each question.
7. Wait for the user's response before continuing.
8. Do not enact the plan until the user confirms shared understanding.

## Stateless Contract

`grill-me` writes nothing by default. Do not create ADRs, glossary entries, specs, tickets, issues, or files unless the user explicitly asks to turn the grilling output into artifacts after the interview.

The only expected artifact is the conversation's sharpened understanding.

## First Turn Template

```markdown
Question: <the most upstream unresolved decision>

My recommendation: <specific recommended answer>

Why: <one or two concise reasons>
```

## Verification Checklist

- [ ] The user explicitly asked for grilling or pressure-testing.
- [ ] No files or durable artifacts were created.
- [ ] The latest turn contains one question, not a questionnaire.
- [ ] The question includes a concrete recommended answer.
- [ ] Implementation is blocked until the user confirms shared understanding.
