---
name: autonomous-llm-research
description: Run durable end-to-end LLM post-training research loops from zero-spec ideas to finished reports. Use when the goal is autonomous literature review, hypothesis selection, experiment planning, Tinker training, evaluation, checkpointing, and resumable long-running research work.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [research, llm, post-training, autonomy, tinker, evaluation]
---

# Autonomous LLM Research

Use this skill as the default operating procedure for Hermes Research Agent.

## Core rules

1. Detect the request type first:
- `zero_spec`: the user gives a broad or vague goal. Start with literature and idea generation.
- `partial_spec`: the user gives some constraints. Fill the missing dataset, eval, and training details.
- `full_spec`: the user gives an executable recipe. Execute it unless there is a contradiction or an approval gate.

2. Be state-first:
- Before substantive work, create or resume a project with `research_loop` or `research_state`.
- Persist ideas, literature notes, hypotheses, experiment plans, results, checkpoints, and reports under `.hermes-research/`.
- Treat chat history as ephemeral. Treat on-disk state as canonical.

3. For `zero_spec`, the sequence is mandatory:
- Search literature and identify concrete gaps.
- Generate 3 candidate ideas.
- Score them on novelty, feasibility, cost, and evaluation clarity.
- Choose one and record why.
- Write a hypothesis, success metric, stop condition, and first experiment plan before training.

4. Never launch training without:
- a written hypothesis
- a written success metric
- a written stop condition
- a persisted experiment plan

5. Use the management layer after each major step:
- Use `research_manager(action="triage_literature", ...)` to rank papers and identify gap candidates.
- Use `research_manager(action="assess_dataset", ...)` before trusting newly generated or curated datasets.
- Use `research_manager(action="rank_runs", ...)` after evaluations to detect regressions versus baseline.
- Use `research_manager(action="plan_next_step", ...)` before deciding the next experiment.
- Use `research_manager(action="write_research_memo", ...)` whenever a loop chunk meaningfully changes the project state.

6. Long-running work must be resumable:
- Use `research_loop(action="checkpoint_loop", ...)` before context or time limits are likely.
- Use `research_loop(action="schedule_continuation", ...)` or rely on the checkpoint helper to resume in a fresh session.
- When Tinker runs are active, use `research_loop(action="monitor_run", ...)` so the project can return after hours of work.

## When to load other skills

- Load `research-idea-generation` when the project starts from little or no specification.
- Load `literature-to-experiment` when turning papers or blog posts into concrete hypotheses and experiment plans.
- Load `tinker` when preparing or launching post-training runs.
- Load `eval-and-ablation` after any training run completes or when planning comparisons.
- Load `research-reporting` when writing iteration reports, summaries, or final deliverables.

## Default output expectations

Every completed loop or chunk should leave behind:
- updated project state in `.hermes-research/`
- experiment and run metadata
- at least one written report or memo
- an inbox item when the work should be reviewed later
