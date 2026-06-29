---
name: view-components
description: Preview selected design system components rendered in localhost
---

# View Components

**Goal:** Render selected components in a localhost preview so the user can see them visually with all states and variants.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Select Components

Present the component catalog and let the user choose what to view:

```
Available components:
1. Button (4 variants)
2. Card (3 variants)
3. Input (5 variants)
...

Select components to preview (comma-separated, or "all"):
```

### Step 2: Generate Preview App

Build a minimal localhost application that renders the selected components:

1. Read component specifications from `{output_folder}/D-Design-System/components/`
2. Read design tokens from `{output_folder}/D-Design-System/design-tokens.md`
3. Generate HTML/CSS that renders each component with:
   - All variants side by side
   - All interactive states (default, hover, active, disabled, focus)
   - Responsive breakpoints
   - Dark/light mode (if defined)
4. Serve on localhost

### Step 3: Interactive Review

With the preview running:

- User inspects components visually
- User can request changes → routes to [E] Edit Components or [C] Create (update)
- User can flag issues → logged for later

### Step 4: Capture Feedback

If the user notes issues or desired changes:

1. Log each item with component name, issue description, severity
2. Suggest next action: edit in Figma, update via Create, or defer

---

## AFTER COMPLETION

1. Update design log
1. Stop localhost server
2. Return to Phase 7 Activity Menu
