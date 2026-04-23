---
name: agent-audit
description: >
  Evidence-first audit workflow for agent runtimes, wrappers, memory layers,
  tool routing, and delivery paths. Use when an agent becomes less reliable
  than the base model, skips tools, leaks stale memory, or mutates otherwise
  correct answers.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [agent-runtime, audit, diagnostics, reliability, tools, memory]
    related_skills: [systematic-debugging, requesting-code-review, subagent-driven-development]
---

# Agent Audit

Audit the agent system itself, not the user's domain task.

Use this skill when an assistant, wrapper, gateway, CLI agent, browser agent,
or long-running runtime:

- behaves worse than the underlying model
- skips required tools
- reuses stale session or memory evidence
- mutates correct answers during retries, formatting, or transport
- hides repair, retry, recap, or summarization layers
- gives confident operational answers without current evidence

## Core rule

Work evidence-first and JSON-first.

Do not jump directly to prose conclusions. Build the structured artifacts first,
then render the user-facing diagnosis from those artifacts.

Required artifacts, in order:

1. `agent_check_scope.json`
2. `evidence_pack.json`
3. `failure_map.json`
4. `agent_check_report.json`

## Audit target

Inspect the full agent stack:

1. system prompt and role shaping
2. session history injection
3. long-term memory retrieval
4. summaries or distillation
5. active recall or recap layers
6. tool routing and selection
7. tool execution
8. tool-output interpretation
9. answer shaping
10. platform rendering or transport
11. fallback or repair loops
12. persistence and stale state

## Required working style

- Prefer direct evidence: code, config, logs, payloads, DB rows, screenshots, and tests.
- Treat a clean current state as insufficient when the reported failure was historical.
- Prefer code and configuration fixes over prompt-only fixes.
- Be explicit about confidence and contradictory evidence.
- Do not blame the base model unless wrapper layers have been falsified.

## Artifact contracts

Read these references before or during the audit:

- `references/report-schema.json`
- `references/rubric.md`
- `references/playbooks.md`
- `references/advanced-playbooks.md`
- `references/example-report.json`
- `references/trigger-prompts.md`

### `agent_check_scope.json`

Define:

- target system
- runtime entrypoints
- channels or surfaces
- model stack
- time window of interest
- known symptoms
- suspected layers
- layers to audit

### `evidence_pack.json`

Capture:

- exact files and code locations
- logs, payloads, DB rows, config files, and screenshots
- whether each evidence item is current, historical, or mixed
- missing evidence that blocks confidence

### `failure_map.json`

For each failure mode include:

- severity
- symptom
- user impact
- source layer
- mechanism
- root cause
- evidence refs
- confidence
- recommended fix

### `agent_check_report.json`

Render the final structured report with:

- executive verdict
- severity-ranked findings
- conflict map across layers
- contamination paths
- ordered fix plan

## Standard playbooks

Use the closest playbook from `references/playbooks.md`:

- `wrapper-regression`
- `memory-contamination`
- `tool-discipline`
- `rendering-transport`
- `hidden-agent-layers`

If the runtime is deeply compromised, use an advanced playbook from
`references/advanced-playbooks.md`.

## Recommended workflow

1. Create the scope artifact.
2. Gather direct evidence.
3. Map failure modes.
4. Build the final report from the structured artifacts.
5. Present:
   - severity-ranked findings first
   - architecture diagnosis second
   - ordered fix plan third

## Output rules

- Lead with findings, not compliments.
- Do not hide uncertainty.
- Do not improvise a new theory after producing the report.
- If the main problem is wrapper design, say so directly.
- If the user asks for JSON, provide `agent_check_report.json`.

## Example prompt

Use `agent-audit` to inspect this agent runtime for wrapper regression and
tool-discipline failures. Focus on stale evidence reuse, hidden repair layers,
and whether tool requirements are enforced in code or only described in prompts.
Build the JSON artifacts first, then give severity-ranked findings and a fix order.
