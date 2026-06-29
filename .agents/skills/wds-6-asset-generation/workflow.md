---
name: wds-6-asset-generation
description: Generate visual and text assets from specifications through AI-powered creative production
---

# Phase 6: Asset Generation

**Goal:** Transform UX specifications into production-ready assets — wireframes, page designs, UI elements, icons, images, videos, and strategic content — using AI-powered creative tools.

**Your Role:** Creative production partner. You read specifications, craft precise prompts using style libraries, and orchestrate batch generation through MCP services. The user brings creative direction; you bring systematic execution.

---

## WORKFLOW ARCHITECTURE

Phase 6 is **menu-driven**, not linear. The user picks what to generate.

### Core Principles

- **Specification-Driven**: Every asset traces back to an approved page spec or design system
- **Style Library**: Design styles and content styles ensure visual consistency
- **Prompt Crafting**: WDS translates specs + style into optimized generation prompts
- **Batch Automation**: Generate all assets of a type in one session (e.g., "17 vehicle images for Källa")
- **Reference Image Support**: Send reference images with prompts for visual consistency across batches
- **Service Flexibility**: MCP-powered generation preferred; prompt export as fallback for external services

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before action
2. **FOLLOW SEQUENCE**: Execute all sections in order
3. **WAIT FOR INPUT**: Halt at menus and wait for user selection
4. **SAVE STATE**: Update design log when completing steps

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:
- `project_name`, `output_folder`, `user_name`
- `communication_language`, `document_output_language`

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Activity Menu

```
What would you like to generate?

[W] Wireframes            — Outline wireframes from page specs
[P] Page Designs          — Full page design compositions
[U] UI Elements           — Buttons, cards, forms, components
[I] Icons                 — Icon sets and individual icons
[M] Images                — Photos, illustrations, backgrounds
[V] Videos                — Motion content and animations
[C] Content               — Strategic text content (5-model framework)
[E] Export to Figma       — Push specs and assets to Figma
```

### Activity Routing

| Choice | Workflow File | Steps Folder |
|--------|--------------|--------------|
| [W] | workflow-wireframes.md | steps-w/ |
| [P] | workflow-page-designs.md | steps-p/ |
| [U] | workflow-ui-elements.md | steps-u/ |
| [I] | workflow-icons.md | steps-i/ |
| [M] | workflow-images.md | steps-m/ |
| [V] | workflow-videos.md | steps-v/ |
| [C] | workflow-content.md | steps-c/ |
| [E] | workflow-figma.md | steps-f/ |

---

## SHARED GENERATION PATTERN

All visual activities (W, P, U, I, M, V) follow this pattern:

1. **Load Context** — Read relevant page specs, design system, visual direction
2. **Asset Inventory** — Identify all assets needed (single or batch)
3. **Select Style** — Choose design style + content style from libraries
4. **Reference Images** — Attach reference images for visual consistency (when supported)
5. **Craft Prompts** — Translate spec + style + references into generation prompts
6. **Select Service** — Route to MCP service or export prompts for external use
7. **Generate & Review** — Execute generation, review results, iterate

### Batch Mode

For batch generation (e.g., "generate all hero images for the site"):
- Inventory all assets of the type from specs
- Apply consistent style across the batch
- Cycle through generation, using earlier results as reference for consistency
- Review as a set, flag outliers for regeneration

### Prompt Export Fallback

When MCP service is unavailable or user prefers external tools:
- Craft the full prompt with all context
- Format for copy-paste into target service
- Include style parameters, dimensions, and reference notes

---

## STYLE LIBRARIES

### Design Styles

Predefined visual approaches loaded from `data/styles/design-styles/`:
- Minimal, Brutalist, Organic, Corporate, Playful, etc.
- Each defines: color treatment, spacing, typography feel, mood

### Content Styles

Visual rendering styles loaded from `data/styles/content-styles/`:
- Photorealistic, Illustration, Watercolor, Comic Book, Pencil Sketch
- Isometric, Flat Design, 3D Render, Hyper-realistic, Line Art, etc.
- Each defines: rendering approach, detail level, texture, lighting

Style selection happens per activity session and can be mixed within a project.

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/styles/design-styles/` | Design style definitions |
| `data/styles/content-styles/` | Content style definitions |
| `data/` | Framework guides, examples, service integration docs |
| `templates/` | Content output, prompt templates |

---

## OUTPUT

- Wireframe images and annotated layouts
- Page design compositions
- UI element assets (buttons, cards, forms)
- Icon sets (SVG, PNG)
- Images (hero, product, background, illustration)
- Video/motion assets
- Strategic text content
- Figma design files

Output stored in `{output_folder}/E-Assets/` organized by type.

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action or return to Activity Menu
