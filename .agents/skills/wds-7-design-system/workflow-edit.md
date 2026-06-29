---
name: edit-components
description: Open design system components in Figma for visual editing
---

# Edit Components

**Goal:** Open selected components in Figma for visual editing, then sync changes back to the design system.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Select Components

Present the component catalog and let the user choose what to edit:

```
Available components:
1. Button (4 variants)
2. Card (3 variants)
...

Select components to edit in Figma (comma-separated):
```

### Step 2: Prepare for Figma

For each selected component:

1. Read current specification from `{output_folder}/D-Design-System/components/`
2. Read associated design tokens
3. Generate or update the Figma-compatible representation
4. Push to Figma via MCP integration (or provide export file)

Confirm Figma file is open and ready for editing.

### Step 3: User Edits in Figma

Pause and wait for the user to make changes in Figma.

```
Components are open in Figma. Make your changes, then tell me when you're done.
```

### Step 4: Sync Changes Back

When the user signals completion:

1. Read updated component data from Figma (via MCP or user export)
2. Diff against current WDS specifications
3. Present changes for approval:
   - Token values changed
   - New variants added
   - Properties modified
4. Update WDS design system files with approved changes

### Step 5: Validate Sync

Run validation to ensure consistency:

- Changed tokens don't break other components
- Variants are complete
- Naming conventions maintained

Report any issues for resolution.

---

## AFTER COMPLETION

1. Update design log
1. Run catalog generation to update component catalog
2. Suggest [V] View Components to verify changes
3. Return to Phase 7 Activity Menu
