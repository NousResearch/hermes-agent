---
name: browse-design-system
description: Generate a disposable localhost app to explore tokens, components, and relationships
---

# Browse Design System

**Goal:** Generate and serve an interactive localhost application for exploring the design system — tokens, components, relationships, and intent-based search.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Design System Data

Read all design system files:

1. `{output_folder}/D-Design-System/design-tokens.md` — All tokens
2. `{output_folder}/D-Design-System/components/*.md` — All components
3. `{output_folder}/D-Design-System/component-library-config.md` — Config

Parse into structured data: tokens with categories, components with dependencies, relationship graph.

### Step 2: Generate Browser Application

Build a single-page localhost app with four views:

**Token Explorer**
- Airtable-style table: filterable by category, sortable by name/value
- Live preview column: colors show swatches, spacing shows bars, typography shows rendered text
- Search: filter by name, value, or category

**Component Catalog**
- Grid of all components with thumbnail previews
- Click to expand: all variants, all states, token dependencies
- Filter by category (navigation, forms, content, layout)

**Relationship Viewer**
- Interactive graph: nodes are tokens and components
- Click a component → highlight all tokens it uses
- Click a token → highlight all components that reference it
- "Show me what uses --color-primary" → highlights the chain

**Intent Search**
- Natural language input: "I need a background for warning messages"
- Matches against token names, descriptions, categories, and component usage
- Shows results with live previews and copy-ready token names

### Step 3: Serve and Interact

Start the localhost server and present:

```
Design System Browser running at http://localhost:XXXX

Views:
- /tokens     — Token Explorer
- /components — Component Catalog
- /graph      — Relationship Viewer
- /search     — Intent Search

Press any key to stop the server.
```

User browses freely. WDS stays available for questions about what they find.

### Step 4: Capture Actions

If the user identifies changes while browsing:

1. Log desired changes (rename token, add variant, fix inconsistency)
2. On exit, present action list
3. Route to appropriate activity: [C] Create, [E] Edit, or [V] View

---

## AFTER COMPLETION

1. Update design log
1. Stop localhost server, discard generated app
2. Return to Phase 7 Activity Menu
