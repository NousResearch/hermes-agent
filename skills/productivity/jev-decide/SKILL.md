---
name: jev-decide
description: "Structure uncertain decisions with calibrated judgments."
version: 1.1.0
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

This returns four decisions (`risk_level`, `agent_choice`, `needs_review`, and `model_class`) plus a readable recommendation. The recommended model is resolved against the live OpenRouter catalog and includes its provider, exact model ID, tier, reasons, and ordered fallback chain. The command is advisory only: it never runs tools, approves changes, deploys, publishes, or changes permissions.

Inspect the authorized catalog without classifying a task:

```text
hermes jev models
hermes jev models --tier mid --min-context-length 1000000 --require-reasoning
hermes jev models --catalog-only
```

The JSON groups all current candidates into `cheap_fast`, `mid`, and `premium`, reports exclusion reasons, and optionally shows the safe fallback order for a requested tier. Jev assigns tiers from the current metadata; `--catalog-only` skips that paid advisory pass and reports deterministic catalog-only sources and policy. Catalog metadata and tier assignments use a profile-scoped one-hour disk cache, so repeated CLI processes do not repeat the catalog fetch or paid classification calls. Cache reads, locks, and writes are best-effort: an unavailable or invalid cache causes a direct live fetch and, unless `--catalog-only` is used, classification; that live result and its usage remain authoritative even when it cannot be cached.

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

Keep agent and model selection as independent axes:

- `agent_choice` selects the execution arrangement (`codex`, `claude`, `either`, or `both`). An agent name is not a model ID.
- `model_class` selects the task tier (`cheap_fast`, `mid`, `premium`, or `no_llm`).
- Jev assigns each current candidate a tier and recommended use from catalog-declared availability, pricing, context, tool support, reasoning, structured output support, reliability evidence, and explicit task requirements. Deterministic code then selects the canonical model and provider route.

Do not attach a model capability to an agent name. For example, say “agent `codex` with model `openai/...`,” not “the Codex model,” and never describe Claude Code as a model.

The absolute floor is the overall capability and reliability represented by Sonnet 4.6. Anthropic Sonnet/Opus below 4.6 and OpenAI GPT below 5.5 are deterministically rejected. Every family and vendor must also declare the Sonnet 4.6 baseline's observable 1M-token context, 64k output, text input/output, availability, and valid pricing. Tool support is always required for task resolution. Reasoning and structured-output support are independent task filters, not properties inferred from a `mid` or `premium` tier; request them explicitly with `--require-reasoning` or `--require-structured-outputs` when the task needs them. Later named variants such as Luna, Sol, Terra, Astra, or a newly named family are candidates only when the exact ID is present and valid in the current provider catalog. Batch-only, unavailable, malformed, and below-floor entries are excluded. Family or version never determines the tier by itself.

A canonical model and a provider route are also separate axes. For OpenAI and Anthropic canonical models, add and prefer the compatible native OpenAI/Codex or Anthropic/Claude Code route only when that exact canonical model appears in Hermes' current native provider catalog (Anthropic dot and dash version spellings are equivalent); then try the catalog-verified OpenRouter route for the same model. If no native match exists, OpenRouter is the primary route rather than evidence that a native route exists. OpenRouter is a provider usable by either execution agent, not a third agent, and OpenRouter-only candidates remain distinct canonical models with compatible execution agents.

When OpenRouter lists multiple routes for one canonical model, candidate-level context, capabilities, and pricing come from the exact unsuffixed canonical row when it exists; otherwise the lexicographically smallest provider model ID is the stable representative. The resolver never blends route-specific metadata into synthetic capabilities or lets a suffix such as `:free` replace the base row. It preserves every unique route and orders a verified native route before deterministic OpenRouter IDs.

When requirements are known, make them explicit:

```text
hermes jev classify-task "review a large repository migration" --min-context-length 1000000 --require-reasoning --require-structured-outputs
```

### Orchestration classification

Use `hermes jev classify-task "TASK"` when a coordinator needs a consistent first-pass routing recommendation. The built-in rubric assesses:

- Risk from Trivial/reversible through Critical/human approval required.
- `codex` for implementation, debugging, and tests; `claude` for planning, research, specifications, writing, and critique; `either` for routine work; or `both` when implementation should receive independent review.
- Whether security, architecture, data, permissions, or ambiguous requirements require review.
- The least costly suitable model class: `cheap_fast`, `mid`, `premium`, or `no_llm`.

The policy recommends human escalation when risk is High or Critical, or when review probability is at least 0.9. It marks a task as eligible for coordinator-controlled automatic approval only when risk is Trivial or Low and review probability is below 0.1; all other cases recommend review. Eligibility is not approval: the coordinator remains responsible for authorization, and existing approval, security, deployment, and publication controls always apply.

After Jev advises a task tier, use the resolver's first route. If it fails at runtime, follow `recommendation.model.fallback_chain` in order: alternate provider routes for that canonical model first, then other models in the requested tier, then higher tiers. Never substitute a lower tier silently, never bypass the authorized floor, and stop for coordinator review if the chain is exhausted.

## Pitfalls

- Do not encode the desired answer in instructions or criteria.
- Do not use `choice` for ordered severity levels; use `score`.
- Do not use `score` for labels with no meaningful order; use `choice`.
- Do not collapse an uncertain compound question into one `noul`; split independent gates.
- Do not interpret confidence as probability that the world is safe. Read it alongside the returned probabilities and the evidence quality.
- Do not send secrets or unnecessary private data in the state.
- Do not invent a model ID from a naming pattern or a rumored release; the current catalog is authoritative.
- Do not fall back below the recommended tier, even to save cost or reduce latency.
- Do not couple `agent_choice` to a model family. Choose each axis from its own evidence.
- Do not count the same canonical model once per provider. Merge its native and OpenRouter routes.
- Do not treat `eligible_for_coordinator_auto_approval` as an approval event. Jev only advises the coordinator and cannot authorize or execute an action.

## Verification

- Confirm the request state matches the current evidence.
- Confirm every question uses the intended criteria shape.
- Confirm all relevant alternatives or rubric levels are represented.
- Confirm the result includes `answers` and `usage`, and record cost when decisions run repeatedly.
- If the result drives a consequential action, verify required human approval independently.
- For `classify-task`, confirm `recommendation.advisory_only` is `true`, both the agent and exact model are stated separately, and apply authorization outside Jev.
- Confirm the selected canonical model and provider are the first fallback entry, duplicate provider routes are merged under one model, every fallback meets the policy floor, and every fallback tier is the requested tier or higher.
- Confirm the profile-scoped catalog snapshot is no more than one hour old when catalog availability matters to an important dispatch.
