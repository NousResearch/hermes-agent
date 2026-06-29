---
name: create-design-system
description: Build a new design system or add components from specifications
---

# Create Design System

**Goal:** Build a design system from scratch or add new components with automatic duplicate detection.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## ENTRY ROUTING

Check design system status:

- **No design system exists** → Start at Step 1 (Initialize)
- **Design system exists, adding component** → Start at Step 2 (Assessment)
- **Known operation** → Jump directly to Step 3

---

## Steps

### Step 1: Initialize Design System

Execute `./steps-c/step-08a-initialize-design-system.md`

Sets up the design system structure: token categories, component organization, naming conventions.

→ After initialization, proceed to Step 3 for first component.

### Step 2: Duplicate Detection (Assessment)

When adding a new component, run assessment before creation:

| Step | File | Purpose |
|------|------|---------|
| 2a | step-01-scan-existing.md | Scan for similar existing components |
| 2b | step-02-compare-attributes.md | Systematic attribute comparison |
| 2c | step-03-calculate-similarity.md | Calculate similarity score |
| 2d | step-04-identify-opportunities.md | Identify reuse opportunities |
| 2e | step-05-identify-risks.md | Identify integration risks |
| 2f | step-06-present-decision.md | Present decision to user |
| 2g | step-07-execute-decision.md | Execute chosen path |

Assessment determines which operation to perform next.

### Step 3: Component Operations

Based on assessment result or direct selection:

| Operation | File | When |
|-----------|------|------|
| Create new | step-08b-create-new-component.md | No similar component exists |
| Update existing | step-08c-update-component.md | Extending an existing component |
| Add variant | step-08d-add-variant.md | Adding a variant to existing component |
| Generate catalog | step-08e-generate-catalog.md | After changes, regenerate catalog |

---

## AFTER COMPLETION

1. Update design log
1. Run catalog generation (step-08e) to update component catalog
2. Return to Phase 7 Activity Menu
