---
name: 'workflow-specify'
description: 'Create a complete, implementation-ready page specification with layout, components, content, interactions, and states.'
---

# [P] Specify — Detail a Page Specification

**Goal:** Create a complete, implementation-ready page specification with layout, components, content, interactions, and states.

**When to use:** When a page structure exists (from Suggest, Dream, or Sketch) and needs full specification detail.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Entry

Load page context from the existing page specification in the scenario's page folder (`{output_folder}/C-UX-Scenarios/[NN-slug]/pages/[NN].[step]-[page-slug]/`).

## Steps

Execute steps in `./steps-p/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-page-basics.md | Page metadata, purpose, entry points |
| 02 | step-02-layout-sections.md | Section layout and ordering |
| 03 | step-03-components-objects.md | Component/object definitions per section |
| 04 | step-04-content-languages.md | Content text and translations |
| 05 | step-05-interactions.md | User interactions and behaviors |
| 06 | step-06-states.md | Loading, error, empty states |
| 07 | step-07-validation.md | Form validation and constraints |
| 08 | step-08-spacing-typography.md | Spacing objects and typography tokens |
| 09 | step-09-generate-spec.md | Generate final specification document |

**Reference data:**
- `./data/object-types/` — component types and templates
- `./data/guides/WDS-SPECIFICATION-PATTERN.md` — specification format
- `./data/modular-architecture/` — three-tier architecture
- `./templates/page-specification.template.md` — output template

---

## AFTER COMPLETION

1. Update design log: status → `specified`
2. Return to the two-option transition from step-01-exploration.md (the calling step determines what comes next based on what was identified during specification)
