---
name: trigger-mapping-validate
description: Validate Trigger Map documents against WDS quality standards
validateWorkflow: './steps-v/step-01-target-group-coverage.md'
---

# Validate Trigger Map

**Goal:** Systematically validate all Trigger Map documents against WDS quality standards and generate an actionable report.

**Your Role:** Validation specialist reviewing trigger map completeness, consistency, and strategic alignment.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:

- `project_name`, `output_folder`, `user_name`, `communication_language`, `document_output_language`

### Load Trigger Map Data

Load all trigger map documents from `{output_folder}/B-Trigger-Map/`.

### Route to Validation

Load, read completely, and execute `{validateWorkflow}` (steps-v/step-01-target-group-coverage.md)

Auto-proceed through all validation steps. Present final report at the end.

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
