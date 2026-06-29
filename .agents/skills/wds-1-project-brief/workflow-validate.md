---
name: 'workflow-validate'
description: 'Verify all Product Brief artifacts are complete, consistent, and ready for Phase 2.'
---

# Phase 1 Validation: Product Brief

**Goal:** Verify all Product Brief artifacts are complete, consistent, and ready for Phase 2.

---

## INITIALIZATION

### Design Log

Read `{output_folder}/_progress/00-design-log.md` before starting.

### Configuration Loading

1. Load project config from `{project-root}/_bmad/wds/config.yaml`
2. Locate Product Brief at `{output_folder}/A-Product-Brief/`
3. Begin validation: Load and execute `./steps-v/step-01-brief-completeness.md`

---

## Validation Sequence

Execute each step in order. Each step produces a section of the final validation report.

| Step | Name | Validates |
|------|------|-----------|
| 01 | Brief Completeness | All required sections present and filled |
| 02 | Trigger Map Consistency | Goals → personas → forces chain is valid |
| 03 | SEO Strategy | Keyword map complete, page assignments clear |
| 04 | Content & Language | Tone, personality, guidelines coherent |
| 05 | Visual Direction | Brand, style, references documented |
| 06 | Platform Requirements | Tech stack, integrations specified |

---

## Final Output

Save validation report to `{output_folder}/A-Product-Brief/validation-report.md`

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
