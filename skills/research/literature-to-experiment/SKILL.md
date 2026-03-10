---
name: literature-to-experiment
description: Convert literature findings into concrete LLM experiment plans. Use when the agent has papers, blog posts, or benchmark notes and needs to turn them into hypotheses, training plans, dataset choices, and evaluation criteria.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [research, literature, experiment-design, planning]
---

# Literature To Experiment

## Workflow

1. Extract the core claim or gap from each relevant source.
2. Map the claim to a testable hypothesis.
3. Decide the minimum viable experiment:
- baseline
- intervention
- dataset
- evaluation suite
- expected win condition
4. Write a stop condition before any training job starts.
5. Save the result as an experiment plan with `research_state(action="record_experiment_plan", ...)`.

## Heuristics

- Prefer the smallest experiment that can falsify the hypothesis.
- If a paper uses complex infrastructure, define the closest tractable approximation first.
- If no benchmark is obvious, choose one that directly measures the stated claim instead of a broad aggregate score.
