# AI Model Routing Policy for GitHub Copilot

This document governs how Hermes Agent selects a GitHub Copilot model for software-development tasks. It is written for both human reviewers and autonomous agents.

## Scope and Core Principles

Use this policy only when delegating work to GitHub Copilot. Top-level planning, architecture, task decomposition, and agent orchestration remain the responsibility of ChatGPT Plus Codex by default. GitHub Copilot is the preferred implementation agent. Use ChatGPT Plus Codex for implementation only when Copilot quota is insufficient or Copilot is otherwise unavailable.

The primary rule is:

> Use the cheapest model that has a high probability of completing the task correctly on the first attempt.

Two constraints qualify that rule:

> Do not sacrifice first-pass correctness merely to minimize the multiplier.

> Clear specification -> cheaper model. Complex or uncertain implementation -> stronger coding model. Same-model failure -> escalate, preferably with model diversity.

Optimize for expected total task cost, not the multiplier of one request. Expected cost includes premium requests, retries, debugging time, context reconstruction, user intervention, orchestration overhead, and the risk of incorrect code.

## Responsibilities

- **ChatGPT Plus Codex:** planning, architecture, decomposition, repository-level strategy, acceptance criteria, and orchestration.
- **Hermes Agent:** task classification, prompt refinement, Copilot model selection, execution control, independent verification, escalation, and concise routing records.
- **GitHub Copilot:** bounded implementation and tests after the task is sufficiently specified.
- **Human reviewer:** major design decisions when requirements remain ambiguous and the final merge decision.

## Model Table

This table is the replaceable data layer for routing. The policy below refers to routing roles rather than assuming these model names and prices are permanent.

Last verified from the authenticated GitHub Copilot CLI `/model` screen: **2026-09-10**.

| Model | Premium request multiplier | Routing role | Default status |
|---|---:|---|---|
| MAI-Code-1.1-Flash | 0.25x | Mechanical | Preferred for trivial work |
| GPT-5 mini | 0.33x | Normal coding | Default implementation model |
| Claude Haiku 4.5 | 0.33x | Normal coding / diversity | Fallback only |
| MAI-Code-1-Flash | 0.33x | Mechanical | Avoid while 1.1 is available |
| GPT-5.3-Codex | 6x | Complex coding | Preferred high-quality coding model |
| GPT-5.4 | 6x | Deep reasoning | Preferred for reasoning, not routine coding |
| GPT-5.4 mini | 6x | Unassigned | Normally avoid |
| Claude Sonnet 4.6 | 9x | Second opinion | Escape hatch after a strong-model failure |
| Gemini 3.5 Flash | 14x | Unassigned | Normally avoid |

### Updating the Model Table

When GitHub changes models, capabilities, or multipliers:

1. Read the authenticated account's current `/model` screen. Do not rely on an old public list or infer account entitlement.
2. Update this table without rewriting the routing rules unless model capabilities or available roles also changed.
3. Reclassify each changed model using coding capability, reasoning capability, agentic repository exploration, first-pass correctness, latency, premium request multiplier, and model-diversity value.
4. Assign each eligible model to one of these roles: Mechanical, Normal coding, Complex coding, Deep reasoning, or Second opinion.
5. Record the verification date. Never derive subscription usage or remaining quota from token counts.

## Decision Procedure

### Step 1: Improve the Specification First

Before buying a stronger model, determine whether Codex or Hermes can reduce task uncertainty by defining:

- the problem and expected behavior;
- relevant files and existing patterns;
- constraints and non-goals;
- acceptance criteria;
- test, lint, and type-check commands.

After refinement, classify the task again. If it has become a clear, bounded implementation, prefer the Normal coding role rather than paying for a stronger model to compensate for a weak prompt.

### Step 2: Estimate First-Pass Risk

Consider:

- number of files and subsystems involved;
- amount of repository exploration required;
- completeness of the specification;
- dependency and control-flow complexity;
- uncertainty of the root cause;
- concurrency, state, migration, integration, security, or production risk;
- cost of a bad change or missed deadline;
- whether another model has already failed.

Do not manufacture a numeric success probability without evidence. Use these signals to choose the lowest-cost role with a high first-pass success likelihood.

### Step 3: Route by Tier

#### Tier 0 — Mechanical

Current preferred model: **MAI-Code-1.1-Flash (0.25x)**.

Use when the task is highly explicit, local, low-risk, and mostly pattern-following:

- dataclasses, type hints, renames, formatting, or boilerplate;
- simple CLI arguments or configuration changes;
- adding one function by following an established pattern;
- a simple SQL edit;
- a small unit test;
- moving notebook code into a module;
- a local change with explicit before/after behavior.

The task must require little repository context or independent reasoning. Do not retry this role repeatedly after an incorrect result.

#### Tier 1 — Normal Coding

Current preferred model: **GPT-5 mini (0.33x)**.

This is the default Copilot implementation route when Codex or Hermes has reduced task entropy. Use for:

- a clear feature with one or a few files;
- ordinary Python, TypeScript, or SQL work;
- API integration with a known endpoint and interface;
- unit tests, functions, and classes;
- a bug fix with a substantially known root cause;
- medium refactoring or extension of an existing pattern;
- an implementation task with clear architecture, constraints, and acceptance criteria.

Do not select an expensive model merely because the original request was broad if decomposition has made the implementation straightforward.

#### Tier 2 — Complex Coding

Current preferred model: **GPT-5.3-Codex (6x)**.

Route here directly when any of these materially affect success:

- a multi-file or repository-level feature;
- incomplete specification that the implementation agent must resolve;
- substantial codebase exploration;
- complex dependencies or control flow;
- debugging with an unknown root cause;
- architectural refactoring;
- concurrency, async behavior, or state management;
- database migrations or complex API integration;
- production-critical or otherwise high-risk logic;
- complex test failures;
- coordinated implementation, tests, and integration changes;
- a need to understand existing abstractions before editing safely;
- one failed attempt from a Mechanical or Normal coding model;
- first-pass correctness is materially more important than request cost.

If the dominant deliverable is working code, choose Complex coding before Deep reasoning.

#### Tier 3 — Deep Reasoning

Current preferred model: **GPT-5.4 (6x)**.

Use when the dominant task is reasoning rather than code production:

- architecture or system design;
- complex trade-off analysis;
- algorithmic reasoning;
- evaluating several designs before choosing an implementation;
- unusually difficult root-cause analysis.

Codex remains the preferred top-level planning system. Use this Copilot tier only when planning must occur inside Copilot or Copilot-specific context makes it useful.

#### Tier 4 — Second Opinion

Current preferred model: **Claude Sonnet 4.6 (9x)**.

Use as a deliberate model-diversity escape hatch, not as the daily default:

- GPT-5.3-Codex produced an inadequate result;
- the current debugging direction may be wrong;
- an independent second opinion is valuable;
- review indicates a fundamental flaw in the current solution;
- difficult reasoning or debugging remains blocked after one strong-model attempt.

Prefer `GPT-5.3-Codex -> Claude Sonnet 4.6` over repeated GPT-5.3-Codex attempts that are likely to follow the same failed reasoning path.

## Decision Tree

```text
START
  |
  +-- Mainly planning, architecture, or orchestration?
  |     +-- Yes -> Prefer ChatGPT Plus Codex
  |                 If Copilot must be used -> Deep reasoning
  |
  +-- Implementation task
        |
        +-- Mechanical, trivial, and pattern-following?
        |     +-- Yes -> Mechanical
        |
        +-- Clear specification and limited scope?
        |     +-- Yes -> Normal coding
        |
        +-- Requires repository exploration, multi-file reasoning,
        |   complex debugging, or high first-pass correctness?
        |     +-- Yes -> Complex coding
        |
        +-- Primarily deep reasoning or architecture rather than code?
        |     +-- Yes -> Deep reasoning
        |
        +-- Has the Complex coding model already failed?
              +-- Yes -> Second opinion
```

## Escalation Policy

Allow at most one failed attempt at a tier before reassessing the task, prompt, and evidence.

```text
Mechanical failure
  -> Normal coding
  -> Complex coding instead if the task was misclassified as trivial

Normal coding failure
  -> Complex coding

Complex coding failure
  -> Second opinion using a different model family
```

Do not use these patterns:

```text
Normal coding -> Normal coding -> Normal coding -> Normal coding
Complex coding -> same Complex coding model indefinitely
```

A retry at the same tier is acceptable only when the failure is clearly external or mechanical, such as a transient tool error, truncated context, or an incorrect command supplied by the orchestrator. Fix that cause before retrying.

## First-Pass Correctness and Total Cost

A low multiplier does not automatically make a route economical. For example, four `0.33x` attempts can consume fewer premium requests than one `6x` attempt, but request cost alone omits:

- debugging and review time;
- user intervention;
- context rebuilding;
- agent orchestration overhead;
- hackathon opportunity cost;
- the risk of incorrect or unsafe code.

Use the stronger model immediately when the expected cost or risk of failed cheap attempts exceeds the savings. Conversely, do not pay `6x` for implementation that clear decomposition makes routine.

## Hackathon Mode

For hackathon work, rank objectives as:

1. first-pass correctness;
2. development velocity;
3. debugging cost;
4. premium request consumption.

Route critical-path work, demo blockers, integration blockers, deployment blockers, and deadline-sensitive features directly to Complex coding when a failed cheap attempt could endanger the demo. Do not save a small number of requests at the cost of delivery risk.

## Models Normally Avoided

- **GPT-5.4 mini:** its current `6x` multiplier provides no clear cost advantage over the preferred Complex coding and Deep reasoning models. Reconsider if its multiplier or demonstrated capability changes.
- **Gemini 3.5 Flash:** its current `14x` multiplier is disproportionate for normal routes. Use only with evidence of a task-specific advantage unavailable elsewhere or after a substantial price change.
- **Claude Haiku 4.5:** not prohibited, but GPT-5 mini is the Normal coding default at the same `0.33x` multiplier. Use Haiku for evidence-based task fit or model diversity.
- **MAI-Code-1-Flash:** prefer MAI-Code-1.1-Flash while the newer model is available at `0.25x`, unless evidence shows the older model performs better for the specific task.

## Default Mapping

```text
Mechanical       -> MAI-Code-1.1-Flash
Normal coding    -> GPT-5 mini
Complex coding   -> GPT-5.3-Codex
Deep reasoning   -> GPT-5.4
Second opinion   -> Claude Sonnet 4.6
```

## Model Selection Transparency

Hermes may route autonomously without asking for confirmation each time.

For a selected model with a multiplier of `6x` or higher, record one concise note before execution or in the execution log:

```text
Model: GPT-5.3-Codex
Reason: Multi-file feature requiring repository exploration and high first-pass correctness.
```

No model-selection notice is required for the low-cost default routes. Always keep the selected model available in the command or execution record so that failures can inform escalation.

## Operational Checklist

Before delegating implementation:

1. Confirm the current authenticated Copilot model table and quota when the information is material to routing.
2. Refine the task specification and acceptance criteria.
3. Classify the task by complexity, uncertainty, risk, and first-pass requirements.
4. Select the cheapest eligible routing role with a high probability of first-pass success.
5. Record the model and reason when its multiplier is `6x` or higher.
6. Execute the task in a bounded feature branch or worktree and prohibit push, merge, rebase, force-push, and unrelated edits.
7. Independently inspect the diff and rerun tests; do not trust an agent's success claim.
8. On failure, improve the specification or escalate once according to this policy, favoring model diversity after a strong-model failure.
