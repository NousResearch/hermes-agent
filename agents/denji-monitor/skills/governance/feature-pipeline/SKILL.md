---
name: feature-pipeline
version: 1
description: >
  Enforced feature-track pipeline for tier=full tasks. Intake → Research → PRD →
  Spec → Council → Sign-off → Tech Review → Decompose → Execute → PR+QA → Audit
  → Final Sign-off → Document. Each stage has an artifact-or-block gate.
category: governance
always_skills: []
---

# Feature Pipeline

Gated progression for tier=full features. Every stage produces an artifact.
No artifact = no promotion. No exceptions.

## Pipeline Stages (strict order)

1. **Intake** — Classify task as tier=full. Create task with `pipeline_stage=intake`.
2. **Research** — Remii produces research artifact. Gate: `validate_research_artifact()`.
3. **PRD** — Kensei-Intake produces PRD. Gate: `validate_prd_artifact()`.
4. **Spec** — Octacon produces spec. Gate: `validate_spec_artifact()`.
5. **Council** — LLM council reviews spec. Gate: `validate_council_artifact()`.
6. **Sign-off** — Human approval gate.
7. **Tech Review** — Multi-agent audit (code, arch, perf, security, UX).
8. **Decompose** — Break into executable tasks.
9. **Execute** — Workers implement.
10. **PR+QA** — Pull request and quality assurance.
11. **Audit** — Final multi-layer audit.
12. **Final Sign-off** — Human approval.
13. **Document** — Knowledge capture.

## Express Path

Drops PRD, Council, and Tech Review. Use for low-risk, well-understood changes.
Express runs are logged as bypasses for Denji review.

Config: `pipeline.express_enabled: true` (default: true)

## Artifact Storage

All artifacts stored at `~/.hermes/feature-artifacts/<task_id>/` as markdown files.
Git-tracked for audit trail.

## Gate Functions

Each stage has a deterministic gate function in `hermes_cli/feature_pipeline.py`:

- `validate_research_artifact(artifact_path)` — checks required sections
- `validate_prd_artifact(artifact_path)` — checks required sections
- `validate_spec_artifact(artifact_path)` — checks required sections
- `validate_council_artifact(artifact_path)` — checks verdict field

Gate returns: `Optional[str]` — `None` means the gate passed; a non-empty string is the human-readable failure reason.

## CLI Commands

```bash
# Create a new feature task
hermes feature create "Feature name" --priority high

# View pipeline status
hermes feature status [task_id]

# Advance to next stage (runs gate check)
hermes feature advance <task_id>
```

## Stage Ownership

| Stage   | Owner Profile    |
|---------|------------------|
| Research | remii           |
| PRD     | kensei           |
| Spec    | octacon         |
| Council | (Phase B)       |

## Config

```yaml
pipeline:
  max_revise_loops: 4      # Max council revisions before escalation
  token_cap: null           # Per-council token cap (None = no cap)
  express_enabled: true     # Allow express path
  artifact_dir: feature-artifacts  # Relative to HERMES_HOME
  stage_owners:
    research: remii
    prd: kensei
    spec: octacon
    council: ""
```

## Pitfalls

- Tier=full tasks that fail intake gate stay in triage with "needs clarification"
- Express bypasses are logged; Denji reviews them weekly
- Council has max_revise_loops backstop to prevent infinite loops
- Never skip artifact creation; the gate is the enforcement mechanism
