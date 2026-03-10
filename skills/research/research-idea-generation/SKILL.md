---
name: research-idea-generation
description: Generate and score candidate LLM research ideas from sparse user goals. Use when a project starts with no concrete benchmark, dataset, or training recipe and the agent needs to propose plausible, testable directions.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [research, ideation, literature, zero-spec]
---

# Research Idea Generation

Use this skill when the user gives a vague or open-ended research goal.

## Workflow

1. Search recent literature and identify recurring limitations, evaluation gaps, or underexplored settings.
2. Produce 3 candidate directions.
3. For each direction, write:
- one-sentence thesis
- why it might work
- the likely benchmark or eval target
- expected cost/risk
4. Score each direction on:
- novelty
- feasibility
- cost
- evaluation clarity
5. Choose one and explain the decision in terms of upside versus execution risk.

## Constraints

- Prefer ideas that can be tested with a clear baseline and measurable success criterion.
- Avoid open-ended ideas that require undefined new infrastructure for v1.
- Record all candidates and the chosen direction with `research_state`.
