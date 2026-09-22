---
name: jev-decide
description: "Structure uncertain decisions with calibrated judgments."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [decisions, probability, risk, model-selection]
    category: productivity
    related_skills: []
---

# Jev Decide Skill

Use Jev when a consequential choice benefits from explicit criteria and calibrated uncertainty. It supports structured judgment; it does not replace evidence gathering, authorization, or the user's final decision.

## When to Use

- Gate a tool call or deployment on a clear yes/no condition.
- Choose one option from a small, mutually exclusive set.
- Place an item on an ordered rubric such as risk, urgency, or readiness.
- Compare model choices against task-specific requirements.

Do not use Jev for factual lookup, open-ended brainstorming, or decisions whose criteria have not been defined. Gather missing evidence with `web_extract`, `read_file`, or `search_files` first.

## Prerequisites

- `OPENROUTER_API_KEY` is present in the active profile's `.env` file.
- The `hermes jev` command is available.
- Network access to OpenRouter is allowed.

## How to Run

Call the command through `terminal` with a state and a JSON question mapping:

```text
hermes jev '{"tests":"green","review":"approved"}' --questions questions.json
```

The state may be plain text or an inline JSON object. Questions may be inline JSON or a JSON file path. Read the returned JSON directly: each answer may include probabilities and confidence, while `usage` reports tokens and cost.

For a standard orchestration recommendation, pass the ticket or task text directly:

```text
hermes jev classify-task "refactor the production auth module"
```

This returns four decisions (`risk_level`, `agent_choice`, `needs_review`, and `model_class`) plus a readable recommendation. The command is advisory only: it never runs tools, approves changes, deploys, publishes, or changes permissions.

## Quick Reference

| Type | Use it for | `criteria` shape |
| --- | --- | --- |
| `noul` | A binary gate or proposition | Optional `{"true":"...","false":"..."}` |
| `choice` | One option among named alternatives | `{"option-a":"...","option-b":"..."}` |
| `score` | Placement on an ordered rubric | `["low","medium","high"]` |

Prefer one focused question per decision dimension. Multiple questions may share the same state when their answers should be judged from identical evidence.

## Procedure

1. Build a compact state containing only decision-relevant facts, constraints, and unknowns. Label observations separately from assumptions.
2. Select the question type by answer shape: `noul` for a proposition, `choice` for unordered alternatives, or `score` for ordered levels.
3. Write instructions that identify the decision and time horizon. Avoid telling Jev which answer you prefer.
4. Make criteria observable and mutually distinguishable. Include disqualifiers where they matter.
5. Run `hermes jev` through `terminal`, then inspect the answer, probability distribution, confidence, and cost.
6. Treat low confidence or close probabilities as a request for more evidence, not as permission to pick arbitrarily.

### Tool gating

Use `noul` when an action must pass a binary safety or readiness gate:

```json
{
  "run_migration": {
    "type": "noul",
    "instructions": "Should the production migration run now?",
    "criteria": {
      "true": "Backup verified, rollback tested, maintenance window open, and approvals present",
      "false": "Any required safeguard, evidence, or approval is missing"
    }
  }
}
```

For a simple boundary, omit `criteria`; add its exact `true` and `false` keys when the boundary needs clarification. Jev's answer is advisory. Existing approval and security controls still apply; never use a favorable result to bypass them.

### Risk classification

Use `score` because risk levels are ordered:

```json
{
  "change_risk": {
    "type": "score",
    "instructions": "Classify the operational risk of this change.",
    "criteria": [
      "low: isolated, reversible, and covered by existing tests",
      "medium: limited blast radius or manual rollback",
      "high: broad blast radius, irreversible effects, or weak validation"
    ]
  }
}
```

Order the list from least to most severe and say what separates adjacent levels.

### Model selection

Use `choice` because model candidates are named alternatives rather than an ordered scale:

```json
{
  "model": {
    "type": "choice",
    "instructions": "Choose the best model for this task under the stated budget and latency constraints.",
    "criteria": {
      "fast-model": "Best when latency and low cost dominate and the task is routine",
      "reasoning-model": "Best when ambiguity, long context, or failure cost requires deeper reasoning",
      "vision-model": "Best when interpreting images is essential"
    }
  }
}
```

Include measured constraints in the state: modality, context size, latency target, budget, and error cost.

### Orchestration classification

Use `hermes jev classify-task "TASK"` when a coordinator needs a consistent first-pass routing recommendation. The built-in rubric assesses:

- Risk from Trivial/reversible through Critical/human approval required.
- `codex` for implementation, debugging, and tests; `claude` for planning, research, specifications, writing, and critique; `either` for routine work; or `both` when implementation should receive independent review.
- Whether security, architecture, data, permissions, or ambiguous requirements require review.
- The least costly suitable model class: `cheap_fast`, `mid`, `premium`, or `no_llm`.

The policy recommends human escalation when risk is High or Critical, or when review probability is at least 0.9. It marks a task as eligible for coordinator-controlled automatic approval only when risk is Trivial or Low and review probability is below 0.1; all other cases recommend review. Eligibility is not approval: the coordinator remains responsible for authorization, and existing approval, security, deployment, and publication controls always apply.

## Pitfalls

- Do not encode the desired answer in instructions or criteria.
- Do not use `choice` for ordered severity levels; use `score`.
- Do not use `score` for labels with no meaningful order; use `choice`.
- Do not collapse an uncertain compound question into one `noul`; split independent gates.
- Do not interpret confidence as probability that the world is safe. Read it alongside the returned probabilities and the evidence quality.
- Do not send secrets or unnecessary private data in the state.
- Do not treat `eligible_for_coordinator_auto_approval` as an approval event. Jev only advises the coordinator and cannot authorize or execute an action.

## Verification

- Confirm the request state matches the current evidence.
- Confirm every question uses the intended criteria shape.
- Confirm all relevant alternatives or rubric levels are represented.
- Confirm the result includes `answers` and `usage`, and record cost when decisions run repeatedly.
- If the result drives a consequential action, verify required human approval independently.
- For `classify-task`, confirm `recommendation.advisory_only` is `true` and apply authorization outside Jev.
