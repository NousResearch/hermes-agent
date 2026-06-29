---
name: 'workflow-design-system'
description: 'Define, update, and review design system components used across page specifications.'
---

# [M] Manage Design System — Define and Update Components

**Goal:** Define, update, and review design system components used across page specifications.

**When to use:** When the user needs to create new components, update existing ones, or review the design system for consistency.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Extraction Rules

Not everything extracts at the same time:

### Objects: Extract on Second Use
The first time a button, card, or widget appears, it stays inline in the page spec — it's a one-off. The **second** time the same pattern appears (same states, same behavior), it's a real pattern. Extract it to the design system.

**First use = one-off. Second use = pattern. Extract.**

### Spacing: Extract Immediately on First Use
Spacing extracts on **first use** — no waiting for a second occurrence. Spacing is relational: when you decide that a heading needs `space-xl` above a card grid, that's a universal design principle, not a page-specific detail.

### Component Extraction Check
Before designing the 2nd+ page, scan completed specs for shared elements. If found, suggest extraction. Don't block the flow — the user can defer.

---

## Entry

Load design system from `{output_folder}/D-Design-System/` (if exists).

## Steps

Execute steps in `./steps-m/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-review-current.md | Review existing design system state |
| 02 | step-02-define-component.md | Define or update a component |
| 03 | step-03-validate-usage.md | Check component usage across specs |

**Reference data:**
- `./data/object-types/` — component type definitions and templates
- `./data/modular-architecture/` — three-tier architecture guide
- `./data/guides/TRANSLATION-ORGANIZATION-GUIDE.md` — content organization

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Design System: [components extracted/updated]`
2. Suggest next action based on the adaptive dashboard
