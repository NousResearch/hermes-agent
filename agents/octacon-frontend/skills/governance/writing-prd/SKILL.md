---
name: writing-prd
version: 1
description: >
  Guide for the PRD stage owner (kensei profile) to write a prd.md artifact
  that passes the validate_prd_artifact gate in the feature pipeline.
category: governance
always_skills: []
---

# Writing a PRD

The PRD artifact unlocks progression from the Research stage to the Spec stage.
The gate (`hermes_cli/feature_pipeline.validate_prd_artifact`) checks that five
required headings are present. Missing any one of them blocks advancement.

## Artifact Location

Write the file to:

```
<HERMES_HOME>/feature-artifacts/<task_id>/prd.md
```

The path expands to `~/.hermes/feature-artifacts/<task_id>/prd.md` on most
installs. Create the directory if it does not exist.

## Required Sections

All five headings must appear as `## Heading` (level-2 markdown). The gate
check is case-insensitive, so `## Problem` and `## problem` both pass, but a
level-3 `### Problem` does not.

### 1. Problem

What is broken, missing, or painful right now? Be concrete. One or two
paragraphs max. Link to supporting research artifact if relevant.

### 2. Users

Who is affected and how? Name the persona or profile. Include scale if known
(e.g. "all kensei-intake tasks, ~N per week"). Avoid vague "end users".

### 3. Scope

What this feature covers. List the capabilities or workflows that are
in-scope for this build. Bullet points are fine.

### 4. Out of Scope

What is explicitly excluded. This prevents scope creep in the Spec stage.
If something is deferred, say when and why.

### 5. Metrics

How success is measured. At least one quantifiable metric per goal.
Examples: reduction in gate-failure rate, latency target, adoption threshold.

## Minimal Template

Copy, fill in, delete placeholder text:

```markdown
# PRD: <Feature Name>

**Task ID:** <task_id>
**Stage:** PRD
**Author:** kensei

---

## Problem

<What is wrong or missing today. 1-2 paragraphs.>

## Users

<Who is affected, estimated volume.>

## Scope

- <Capability 1>
- <Capability 2>

## Out of Scope

- <Exclusion 1>
- <Exclusion 2>

## Metrics

- <Metric 1 — target value and measurement method>
- <Metric 2>
```

## Gate Check

Run the gate manually before calling `hermes feature advance`:

```bash
python - <<'EOF'
from hermes_cli.feature_pipeline import validate_prd_artifact
result = validate_prd_artifact("~/.hermes/feature-artifacts/<task_id>")
print("PASS" if result is None else f"BLOCKED: {result}")
EOF
```

`None` = gate passes. Any string = failure reason; add the missing section and
re-check.

## Pitfalls

- Level-3 headings (`###`) do not satisfy the gate. Use `##`.
- The gate checks the file `prd.md` inside the artifact directory, not the
  directory itself. Ensure the filename is exactly `prd.md`.
- Keep the PRD focused on *what* and *why*, not *how*. The Spec stage (Octacon)
  owns the technical design.
