---
name: dezzy-excalidraw
description: "Generate hand-drawn style Excalidraw JSON diagrams directly — architecture flows, user journeys, component trees, sequence diagrams. Saves as .excalidraw files for drag-and-drop editing on excalidraw.com."
version: 1.0.0
author: KENSEI
adoption_status: provisional
metadata:
  hermes:
    tags: [design, diagram, excalidraw, wireframe, flow]
    category: design
    related_skills: [mobile-screen-spec, site-teardown, ui-pattern-library-research]
    created_for_profile: dezzy
---

# Dezzy Excalidraw

Generate hand-drawn style diagrams as standard Excalidraw JSON. Output `.excalidraw` files that can be opened directly on excalidraw.com or embedded into design docs.

## When This Skill Activates

- "Draw me an architecture diagram"
- "Visualise this user flow"
- "Create a wireframe of the onboarding screens"
- "Show me the component hierarchy"
- "/dezzy-excalidraw"

## Output Format

Standard `.excalidraw` JSON file. The file is a valid Excalidraw document with elements array, viewport config, and metadata. No external tools — just pure JSON written to disk.

## KENSEI-Native Workflow

1. **Understand the system** — what's being diagrammed? (architecture, flow, component tree, journey)
2. **Plan the layout** — rough sketch of element positions, connections, groups
3. **Generate Excalidraw JSON** — elements array with rectangles, diamonds, ellipses, arrows, text labels, groups, and container bindings
4. **Save** — write to `research/<name>.excalidraw` or the Dezzy kanban output directory
5. **Verification** — confirm the JSON is parseable valid Excalidraw format

## Elements Reference

| Element | JSON Tag | Use For |
|---------|----------|---------|
| Rectangle | `type: rectangle` | Components, services, screens |
| Diamond | `type: diamond` | Decision points |
| Ellipse | `type: ellipse` | External systems, databases |
| Arrow | `type: arrow` | Data flow, navigation |
| Text | `type: text` | Labels, annotations |
| Line | `type: line` | Custom connectors |

All elements require: `id`, `type`, `x`, `y`, `width`, `height` at minimum. Text elements need a `text` field and `fontSize`. Arrows need `points` array.

## Verification

- File extension: `.excalidraw`
- Envelope structure: `{ type: "excalidraw", elements: [...], state: {...} }`
- Each element has a unique `id`
- Arrows reference element IDs, not coordinates
- Text elements have `containerId` when bound to shapes

## Pitfalls

- Don't use absolute pixel coordinates — use relational positioning (spacing = 20px between items)
- Groups need explicit `groupIds` on each child element
- Arrows need `points` array with at least `[[x1,y1],[x2,y2]]`
- Colour-key by purpose: system=blue, user=green, external=grey, decision=yellow