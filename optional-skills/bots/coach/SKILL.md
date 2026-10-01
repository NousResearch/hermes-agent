---
name: coach
description: Frame a life or career decision and pick a next step.
version: 0.1.0
author: Bot Cabinet, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [coaching, decisions, career, planning]
    category: bots
    related_skills: []
---

# Coach Workflow Skill

Help one person state where they are, where they want to go, and what limits them, then compare realistic paths and choose a small, reversible next step. Adapted from the Bot Cabinet Coach starter (MIT, github.com/Dgardenhire/bot-cabinet). The workflow asks and reflects; it does not decide for the person.

## When to Use

- The user is weighing a life or career decision and wants to think it through.
- The user wants a plan sized to the time, energy, and money they actually have.
- Do not use for medical, psychological, legal, or financial advice, for crisis support, or to contact anyone or change a schedule.

## Prerequisites

The first useful task needs only the conversation. Notes the user supplies can be read with `read_file`. No accounts or tools are required. If the user describes risk of harm, stop the coaching flow and point them to local emergency or professional support.

## How to Run

1. Copy `templates/decision-frame.md` into the owning profile workspace with a collision-safe name for each decision; never overwrite an earlier frame.
2. Try `references/sample-input.md` for an offline synthetic trial, or start from the user's own situation.
3. Check the frame against `references/expected-output-rubric.md`.

## Quick Reference

| Item | Meaning |
|---|---|
| Idea | something the person might do |
| Plan | an idea with a first step and a time |
| Commitment | something the person has confirmed they will do |
| Constraint | time, energy, money, or an obligation that limits options |
| Test | a small, reversible step that produces information |

A check-in routine is a recommendation only. Do not automatically create a scheduled job; scheduling requires explicit activation, a schedule, and a delivery destination chosen by the user.

## Procedure

### 1. State the situation

Capture the current situation, the desired direction, obligations, and constraints in the person's own words. Ask one consequential question at a time. **Done when** the situation and the direction each fit in two sentences the person agrees with.

### 2. Frame the decision

Write the decision as one question with a time horizon, and separate ideas, plans, and confirmed commitments. **Done when** the decision is one answerable question and nothing is labelled a commitment unless the person confirmed it.

### 3. Compare options

Lay out two or three realistic options with their tradeoffs against the stated constraints. Do not push toward a choice. **Done when** each option has at least one cost and one benefit tied to the person's constraints.

### 4. Size a next test

Propose one small, reversible step for the most promising option that fits the time and energy available and produces information. **Done when** the test has a size, a time, and a clear thing to learn.

### 5. Reflect back and stop

Present the decision frame with open questions and assumptions. The person decides what, if anything, becomes a commitment. **Done when** the person has the frame, and the rubric passes or remaining gaps are named.

## Pitfalls

- Do not diagnose, prescribe, or replace qualified professional advice.
- Do not treat an idea as a commitment, or pressure the person toward one option.
- Do not contact anyone or change a calendar.
- Offline sample mode uses an invented person and decision and makes no claim about a real situation.

## Verification

- [ ] The decision is one question with a horizon.
- [ ] Ideas, plans, and commitments are kept distinct.
- [ ] Options show tradeoffs against stated constraints.
- [ ] The next step is small, reversible, and sized.
- [ ] No advice outside coaching was given, and no earlier frame was overwritten.
