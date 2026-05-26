# Town / OpenClaw Agent Flow Contract

status: DRAFT / NOT ACTIVE
created: 2026-05-26
scope: reference documentation only

## Purpose

Define how Hermes, OpenClaw wrapper skills, Town agents, MCP tools, and future
self-improvement proposals should interact safely.

This is a governance and routing contract. It is not runtime behavior, not an
active monitoring system, not automation, and not a Hermes skill.

## 1. Current State

- Hermes repo is clean/current at the time this contract is drafted.
- MCP exposes the current context tools:
  - `town_brief`
  - `fleet_context_snapshot`
  - `agent_health_summary`
  - `knowledge_query`
  - `skills_list` / `skills_read`
  - `agents_list` / `agents_get`
  - `knowledge_read`
  - `learnings_read`
  - `artifacts_list`
- `skills/openclaw/*` files are shallow wrapper skills, not behavioral truth.
- `/workspace/agents` is registry-side only in the cloud checkout.
- Runtime SOUL/HEARTBEAT validation requires a real local `HERMES_AGENTS_DIR`.
- The seven-layer identity stack is conceptually approved but
  implementation-blocked until the runtime identity source is verified.

## 2. Flow Model

Intended read-only flow:

```text
User / Cursor request
-> Cursor rule / command
-> Hermes MCP context tools
-> Town brief / fleet snapshot / agent health / knowledge query
-> OpenClaw wrapper skill as routing hint only
-> proposed action
-> operator approval
-> only then implementation
```

OpenClaw wrapper skills may help route a request to a likely agent or domain,
but they do not define agent identity or behavioral authority.

## 3. Source-of-Truth Hierarchy

Precedence order:

1. Runtime `HERMES_AGENTS_DIR` identity files, once verified.
2. `AGENT_REGISTRY.json` for registered agent metadata.
3. `docs/skills/*` for documented skill behavior.
4. `skills/openclaw/*` for lightweight routing wrappers only.
5. Generated knowledge ledgers as summaries, not authority.

Rules:

- Do not synthesize identity from registry metadata.
- Do not treat OpenClaw wrappers as behavioral truth.
- Do not persist `ACTIVE_CONTEXT` as durable memory.
- Do not write `HISTORY_MAP` or `HISTORY_NEW` until an append/provenance policy
  exists.

## 4. Agent Handoff Rules

For any Town/OpenClaw task:

1. Identify the target agent or skill.
2. Read relevant skill/context.
3. Check agent health.
4. Check `docs/FAILURE_PATTERN_LIBRARY.md`.
5. Classify the requested action as:
   - docs
   - proposal
   - test
   - runtime
   - production
6. Stop if the action touches any of the following without explicit approval:
   - runtime identity
   - `AGENT_REGISTRY.json`
   - MCP config
   - cron
   - production KG
   - model/ranker/selector/sizing
   - agent behavior

## 5. Self-Improvement Loop

Self-improvement is proposal-only:

```text
observe
-> classify failure/opportunity
-> search Failure Pattern Library
-> search knowledge/learnings
-> propose bounded improvement
-> attach risk classification
-> operator approval
-> implementation in a separate step
```

Self-improvement proposals must include:

- problem observed
- evidence
- affected agents/skills/tools
- proposed change
- files touched
- rollback
- risk class
- whether behavior changes
- whether approval is required

## 6. Promotion Gates

A prevention rule may be proposed for promotion from
`docs/FAILURE_PATTERN_LIBRARY.md` only when:

- `recurrence_count >= 3`
- root cause is confirmed
- affected skill/agent is identified
- proposed text is diff-only
- operator approves

No automatic promotion is allowed.

## 7. Do-Not-Cross Boundaries

Block without explicit approval:

- `AGENT_REGISTRY.json` edits
- `skills/openclaw/*` edits
- MCP config changes
- gateway/platform changes
- cron changes
- runtime identity files
- SOUL/HEARTBEAT/HISTORY writes
- production model/ranker/selector/sizing/KG changes
- Codegraph Hermes registration

## 8. Recommended Cursor Commands

### Run Town agent context preflight

```text
Run Town/OpenClaw agent context preflight.

Use:
- docs/TOWN_OPENCLAW_AGENT_FLOW_CONTRACT.md
- docs/FAILURE_PATTERN_LIBRARY.md
- fleet_context_snapshot
- agent_health_summary
- town_brief
- skills_list / skills_read
- agents_list / agents_get
- knowledge_query

Return:
1. target agent/skill
2. source-of-truth consulted
3. relevant known failure patterns
4. health warnings
5. risk class: SAFE / GATED / BLOCKED
6. proposed next action
7. exact files that would be touched

Stop before edits unless explicitly approved.
```

### Draft self-improvement proposal

```text
Draft a self-improvement proposal only.

Include:
- observed problem
- evidence
- related failure patterns
- affected agents/skills/tools
- proposed change
- files to touch
- behavior change: yes/no
- runtime state change: yes/no
- risk class: SAFE / GATED / BLOCKED
- approval required: yes/no
- validation plan
- rollback plan

Do not implement.
```

### Check failure pattern before investigation

```text
Check docs/FAILURE_PATTERN_LIBRARY.md before investigating this failure.

If matched and resolved:
- apply the known fix before re-investigating

If matched and UNRESOLVED:
- update recurrence/diagnostic information only
- do not invent a fix

If unmatched:
- continue investigation
- add a new entry only after confirmed root cause
```

### Classify OpenClaw wrapper vs runtime truth

```text
Classify whether the referenced OpenClaw skill is a routing wrapper or runtime truth.

Return:
- wrapper skill path
- registry metadata
- runtime identity path, if verified
- whether SOUL/HEARTBEAT exists
- whether HERMES_AGENTS_DIR is required
- safe next action

Do not synthesize identity from wrapper text.
```

### Prepare guarded skill promotion proposal

```text
Prepare a prevention-rule promotion proposal.

Requirements:
- source failure_id
- recurrence_count
- confirmed root cause
- affected skill/agent
- exact proposed diff text
- validation plan
- rollback plan
- operator approval required

Do not apply the promotion.
```

## Non-Goals

- This contract does not create runtime identity layers.
- This contract does not write SOUL, HEARTBEAT, HISTORY, or ACTIVE_CONTEXT.
- This contract does not register Codegraph with Hermes.
- This contract does not mutate MCP configuration.
- This contract does not implement Town-to-Hermes feedback automation.
