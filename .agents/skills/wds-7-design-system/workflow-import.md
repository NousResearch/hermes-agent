---
name: import-design-system
description: Import an existing design system into the WDS format
---

# Import Design System

**Goal:** Bring an existing design system into WDS — from a URL, file export, or Figma project.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Identify Source

Ask the user for the design system source:

- **URL** — Public design system documentation (e.g., Material UI, Chakra, custom)
- **File** — Exported tokens file (JSON, CSS custom properties, SCSS variables)
- **Figma** — Figma design system file (via Figma MCP or export)
- **Code** — Existing codebase with component library

### Step 2: Extract Tokens

Read the source and extract design tokens:

1. **Colors** — Primary, secondary, semantic, neutrals
2. **Typography** — Font families, sizes, weights, line heights
3. **Spacing** — Scale values, named spacing tokens
4. **Shadows** — Elevation levels
5. **Borders** — Radius values, border styles
6. **Breakpoints** — Responsive breakpoints
7. **Motion** — Transition durations, easing curves

Present extracted tokens to user for review. Mark any ambiguous mappings.

### Step 3: Extract Components

Identify reusable components from the source:

1. List all components found
2. For each: name, props/variants, token dependencies
3. Map to WDS component template format
4. Flag components that don't map cleanly

Present component list for user approval.

### Step 4: Generate Design System Files

Create the WDS design system structure:

1. `design-tokens.md` — All extracted tokens in WDS format
2. `components/*.md` — One file per component
3. `component-library-config.md` — Configuration and metadata

### Step 5: Validate Import

Run validation:

- All tokens referenced by components exist
- No orphaned tokens (defined but never used)
- Naming conventions consistent
- Component variants complete

Present validation report. Fix issues interactively.

---

## AFTER COMPLETION

1. Update design log
1. Suggest running [B] Browse Design System to explore the import
2. Return to Phase 7 Activity Menu
