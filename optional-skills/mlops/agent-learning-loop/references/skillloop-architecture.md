# SkillLoop-Style Agent Learning Loop Reference

This reference describes a clean sidecar architecture for improving agents from their own traces while avoiding unsafe self-mutation.

## Goals

- Make agent learning inspectable and reviewable
- Preserve a clean boundary between runtime state and learning artifacts
- Support multiple agent runtimes through adapters
- Export training data without requiring training inside Hermes
- Avoid storing credentials or private data in public datasets

## Non-goals

- Replacing Hermes memory or skill tools
- Automatically fine-tuning models after every session
- Writing directly into live user memory or skills
- Adding mandatory cloud dependencies

## Components

### 1. Trace adapters

Adapters convert runtime-specific outputs into a common trace object.

Recommended adapters:

- generic JSONL messages
- Hermes session exports
- Codex/Claude/OpenCode transcripts
- custom application logs

Adapters should be tolerant of unknown fields and preserve source metadata.

### 2. Local store

A local SQLite database is enough for the first version. Store:

- traces
- evaluations
- proposals
- review status
- export metadata

Keep it under the chosen project root, not under live profile-scoped Hermes state.

### 3. Evaluation engine

Start deterministic:

- final answer present?
- tool failures?
- tests or verification present?
- user correction detected?
- success signal detected?

Add LLM judges later as optional evaluators, not as the only source of truth.

### 4. Distillation engine

Distill two proposal types first:

- memory proposals for durable preferences/facts
- skill proposals for reusable procedures

Store proposed content plus reason/provenance. Do not apply directly.

### 5. Review queue

Every proposal should have a status:

- pending
- approved
- rejected

Approval can be manual, reviewer-agent-assisted, or policy-based, but the reviewed state should be explicit.

### 6. Exporters

Export approved or high-quality traces into:

- SFT JSONL
- DPO JSONL
- approved memory markdown
- approved skill markdown

The exporter should be deterministic and repeatable.

## Suggested record formats

### SFT

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### DPO

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### Memory proposal

```json
{
  "kind": "memory",
  "title": "Durable user preference",
  "content": "User prefers concise terminal summaries.",
  "reason": "User explicitly corrected response style.",
  "status": "pending"
}
```

### Skill proposal

```json
{
  "kind": "skill",
  "title": "Reusable debug workflow",
  "content": "# Workflow...",
  "reason": "Trace contains repeated fix/verify pattern.",
  "status": "pending"
}
```

## Review-first flow

```text
trace -> eval -> proposal -> review -> approved export -> optional import/train
```

This keeps experimental learning artifacts separate from live agent behavior.

## Integration points with Hermes

Safe first integrations:

- Skill documenting the workflow
- Optional plugin/sidecar that reads exported sessions
- CLI command that exports traces to a neutral format
- Docs showing how to curate datasets

Avoid as first integrations:

- Automatically writing to the selected profile's `$HERMES_HOME/memories`
- Automatically writing to bundled `skills/`
- Adding fine-tuning dependencies to core install
- Training jobs that run without user approval

## Minimal acceptance criteria

A sidecar implementation is useful when it can:

1. Ingest a sample trace.
2. Store it locally.
3. Evaluate it deterministically.
4. Produce at least one reviewable proposal from a learning signal.
5. Export valid SFT JSONL.
6. Leave no generated artifacts in source control.
7. Pass tests for schema, store, adapters, evaluation, review, and export.

## Contributor positioning

For Hermes Agent upstream, this architecture is best contributed incrementally:

1. Start as an optional skill and reference architecture.
2. Add a standalone plugin or sidecar package after interface feedback.
3. Only move pieces into core when they are stable, broadly useful, and dependency-light.
