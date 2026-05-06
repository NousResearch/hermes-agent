---
name: traceguard
description: Deterministically validate structured parent-synthesis claims against accepted child evidence handles. Optional evidence-gating workflow for RLM-style synthesis; not part of Hermes core tools.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Evidence, Verification, Synthesis, RLM, Traceability]
    category: research
    related_skills: []
---

# TraceGuard

TraceGuard is an optional deterministic evidence gate for synthesis workflows.
It validates structured parent claims against a bounded manifest of accepted
child evidence handles. It is not an LLM judge and is not loaded as a core
Hermes tool by default.

## When to use

Use this skill when a synthesis workflow needs every retained or observed fact
to cite an accepted `fact_id` and matching `chunk_id`/`evidence_chunk_id` from a
child-evidence manifest.

## Included files

- `traceguard.py` — deterministic validation primitives.
- `traceguard_tool.py` — optional registry-backed tool wrapper for environments
  that explicitly load this skill's Python tool module.
- `tests/` — optional-skill regression tests.

## Workflow

1. Build or receive an accepted evidence manifest.
2. Produce a structured parent synthesis with fact/chunk handles.
3. Run `traceguard_validate` when this optional tool is loaded, or call
   `validate_parent_synthesis()` directly from `traceguard.py`.
4. Treat any rejected claim as unsupported until the parent synthesis cites an
   accepted fact ID and matching evidence handle.
