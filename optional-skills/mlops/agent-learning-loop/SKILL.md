---
name: agent-learning-loop
description: "Curate agent traces into reviewed learning artifacts."
version: 1.0.0
author: lamenting-hawthorn
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [agent-learning, trace-curation, memory, skills, sft, dpo, evaluation]
    related_skills: [fine-tuning-with-trl, axolotl, requesting-code-review]
---

# Agent Learning Loop Skill

Build or operate a review-first sidecar that turns completed agent traces into
evaluations, proposals, and training records. The sidecar may read Hermes
trajectories, but it must not silently mutate live Hermes state.

## When to Use

Use this skill when:

- curating Hermes sessions or other agent traces into training data
- proposing durable memory or reusable skill updates from completed work
- designing a local SFT, DPO, or evaluation pipeline
- adding explicit review between generated proposals and live state

Do not use it to import unreviewed output, retain credentials, or treat every
successful session as training data.

## Prerequisites

- Completed trajectories; Hermes can save ShareGPT-compatible JSONL.
- A project-local sidecar directory outside live Hermes state.
- An explicit reviewer for memory, skill, and dataset proposals.
- Access to `$HERMES_HOME` when locating profile-scoped Hermes state. Never
  assume the default home directory; custom homes and named profiles keep
  separate state.

Use native Hermes tools such as `read_file`, `search_files`, `patch`, and
`terminal` to inspect traces and build the sidecar. If an independent verifier
is available, reserve it for the final readiness gate unless the user requests
earlier review.

## How to Run

This skill specifies an architecture, not a bundled executable. Implement the
pipeline in the user's sidecar project, then run that project's documented
commands with `terminal`. Do not copy command names from this document: no
sidecar scripts are shipped by this skill.

Keep generated state under the selected project root, for example:

```text
.agent-learning/
  learning.db
  proposals/
  approved/
exports/
  sft.jsonl
  dpo.jsonl
```

Importing an approved artifact into `$HERMES_HOME/memories` or
`$HERMES_HOME/skills` is a separate, explicit operation.

## Quick Reference

| Stage | Input | Output | Required boundary |
|---|---|---|---|
| Ingest | trajectory JSON/JSONL | normalized trace | preserve provenance |
| Evaluate | normalized trace | score, tags, notes | deterministic checks first |
| Distill | trace and evaluation | proposals | no live-state writes |
| Review | pending proposals | decisions | explicit approval |
| Export | approved records | Markdown/JSONL | redact and validate |
| Import/train | approved exports | live artifact/model | separate workflow |

See `references/skillloop-architecture.md` for the detailed reference design.

## Procedure

### 1. Select and isolate inputs

Locate the requested Hermes profile through `$HERMES_HOME`, or accept traces
exported by the user. Copy or stream the selected completed traces into the
sidecar boundary; do not edit the originals.

Record source, timestamp, model, completion state, and available tool metadata.
Add a stable trace ID so every proposal and export can be traced back.

### 2. Normalize traces

Map each runtime into a small internal shape:

```json
{
  "id": "trace-id",
  "source": "hermes",
  "created_at": "2026-06-05T00:00:00Z",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "name": "terminal", "content": "..."}
  ],
  "metadata": {}
}
```

Retain tool calls and results only when evaluation needs them. Redact secrets
before persistence or export, not after publication.

### 3. Evaluate

Run deterministic checks before adding an LLM judge.

Positive evidence includes a final answer, passing tests, successful tool
output, user confirmation, and recovery from earlier failures. Negative
evidence includes incomplete traces, unresolved exceptions, user corrections,
unsupported success claims, and unsafe credential handling.

Store the evidence, not only a score:

```json
{
  "trace_id": "abc123",
  "score": 82,
  "tags": ["completed", "verified"],
  "notes": ["Tests passed and the final answer cites the result."]
}
```

### 4. Distill proposals

Propose memory only for durable facts such as stable preferences, recurring
project conventions, or hard-to-rediscover environment constraints. Exclude
temporary progress, raw logs, credentials, commit IDs, and soon-stale facts.

Use declarative wording, for example: `User prefers final verification at the
pre-push gate.` Do not turn a personal preference into a global imperative.

Propose a skill only when a trace contains a reusable procedure. Include its
trigger, sequence, pitfalls, verification, and safety boundaries. Keep every
proposal pending until review.

### 5. Review

Present pending proposals with their source evidence. The reviewer must be able
to approve, reject, or edit each item. Record the decision and reviewer without
overwriting the original proposal.

Approval authorizes export only. It does not authorize an automatic write to
`$HERMES_HOME`, bundled skills, configuration, or credentials.

### 6. Export

For SFT, export only curated, high-quality instruction-following records:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For DPO, require a real chosen/rejected contrast:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Valid contrasts include an answer rejected by the user and its accepted fix,
or two responses reviewed against the same rubric. Never invent a rejected
response. Validate every JSONL line and retain provenance separately.

### 7. Verify before import or training

Scan approved output for credentials and PII. Review memory and skill content
manually. If available, use an independent verifier at this final gate.

Any later import must target the intended profile's `$HERMES_HOME`, use normal
Hermes memory or skill mechanisms, and preserve a rollback path.

## Pitfalls

1. **Assuming the default home.** Resolve the selected profile through
   `$HERMES_HOME`; named profiles and custom homes are isolated.
2. **Self-mutation by default.** Generate proposals and require explicit
   review before any live-state change.
3. **Training on unresolved failures.** Use failed traces only when they are
   clearly labeled for evaluation or genuine preference pairs.
4. **Secret leakage.** Tool results can expose tokens, cookies, URLs, paths,
   or environment values. Redact before storage and again before export.
5. **Weak provenance.** Preserve enough source metadata to locate and remove
   a bad record later.
6. **Overfitting.** Keep user-specific memory separate from general skills and
   public datasets.

## Verification

- [ ] The sidecar writes only inside its selected project root.
- [ ] Live state is referenced through `$HERMES_HOME`, not a fixed home path.
- [ ] The pipeline can normalize, evaluate, propose, review, and export a
      sample trace.
- [ ] Generated state is ignored by version control.
- [ ] Tests cover the sidecar implementation's ingestion, evaluation, review,
      and export behavior.
- [ ] Every SFT/DPO JSONL line parses independently.
- [ ] Credential and PII scans pass.
- [ ] Only approved artifacts proceed to import or training.
- [ ] Final claims are backed by real command or test output.
