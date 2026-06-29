---
name: wds-7-design-system
description: Create, import, browse, and maintain design system components and tokens
---

# Phase 7: Design System

**Goal:** Build and maintain a living design system — components, tokens, and their relationships — with visual browsing and editing through localhost and Figma.

**Your Role:** Design system architect. You extract components from specs, manage tokens, detect duplicates, and generate interactive browsing tools so the user can explore the system visually.

---

## WORKFLOW ARCHITECTURE

Phase 7 is **menu-driven**, not linear. The user picks an activity.

### Core Principles

- **Code as Source of Truth**: All tokens and components stored in code
- **Visual Browsing**: Localhost apps for exploring tokens, components, and relationships
- **Intent-Based Discovery**: Search by what you want to do, not by token name
- **Duplicate Detection**: Similarity analysis when adding new components
- **Figma Integration**: Edit components visually in Figma

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before action
2. **FOLLOW SEQUENCE**: Execute all sections in order
3. **WAIT FOR INPUT**: Halt at decision points and wait for user
4. **SAVE STATE**: Update design log when completing steps

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:
- `project_name`, `output_folder`, `user_name`
- `communication_language`, `document_output_language`
- `design_system_mode` (none / basic / full)

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Activity Menu

```
What would you like to do?

[C] Create Design System     — Build a new design system from specs
[I] Import Design System     — Bring in an existing design system
[V] View Components          — Preview selected components in localhost
[E] Edit Components          — Open selected components in Figma
[B] Browse Design System     — Search and explore tokens and components in localhost
```

### Activity Routing

| Choice | Workflow File | Steps |
|--------|--------------|-------|
| [C] | workflow-create.md | steps-c/ |
| [I] | workflow-import.md | Inline |
| [V] | workflow-view.md | Inline |
| [E] | workflow-edit.md | Inline |
| [B] | workflow-browse.md | Inline |

---

## CREATE DESIGN SYSTEM

When creating or adding components, WDS runs duplicate detection internally:
1. Scan existing components for similarity
2. Compare attributes systematically
3. Present decision if near-match found (reuse, extend, or create new)

This replaces the old assessment-first router — duplicate detection is a step within creation, not a separate workflow.

---

## BROWSE DESIGN SYSTEM

WDS generates a disposable localhost application from the current design system data:

- **Token Explorer**: Airtable-style filterable/sortable view of all tokens
- **Relationship Viewer**: Visualize how tokens connect (e.g., button styles → color tokens → spacing tokens)
- **Intent Search**: "I need a shadow for cards" → shows relevant tokens with live previews
- **Component Catalog**: Browse all components with rendered previews and state variations

The app is regenerated from current data each time — always reflects the latest state.

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/design-system-guide.md` | Comprehensive design system guide |
| `templates/` | Component, tokens, config, catalog templates |

---

## OUTPUT

- `{output_folder}/D-Design-System/components/*.md`
- `{output_folder}/D-Design-System/design-tokens.md`
- `{output_folder}/D-Design-System/component-library-config.md`

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action or return to Activity Menu
