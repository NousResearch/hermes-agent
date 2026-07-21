---
name: ascii-excalidraw
description: "Convert ASCII art diagrams to hand-drawn Excalidraw JSON files. Analyzes structure first, then generates incrementally module-by-module."
version: 1.0.0
author: Claude Code
license: MIT
dependencies: []
metadata:
  hermes:
    tags: [ASCII, Excalidraw, Diagrams, Converter, Architecture, Flowchart, Visualization]
    related_skills: [excalidraw]
---

# ASCII to Excalidraw Converter

Convert ASCII art diagrams into polished, hand-drawn style Excalidraw JSON files. This skill specializes in **parsing ASCII diagrams** and producing structured `.excalidraw` files that can be opened at [excalidraw.com](https://excalidraw.com).

## When to Use

The user provides an ASCII diagram (boxes, arrows, text) and wants a visual Excalidraw file. Typical inputs:

```
┌────────────┐      ┌──────────────┐
│   Web App  │─────▶│  API Server  │
└────────────┘      └──────────────┘
                           │
                     ┌─────▼─────┐
                     │ Database  │
                     └───────────┘
```

## Conversion Workflow

**CRITICAL: Do NOT generate the entire JSON in one shot.** Follow this multi-step process to reduce errors.

### Step 1: Analyze + Quantitative Layout Planning

Parse the full ASCII diagram and produce a structured analysis **plus a quantitative layout plan** before generating any JSON.

#### 1a. Qualitative Analysis

1. **Detect all boxes**: Find `+------+`, `┌──┐` corner patterns. Record positions, sizes, and text content.
2. **Detect all arrows**: Find `--->`, `-->`, `▼`, `▲`, `◄`, `►`, `│` patterns. Record source, target, direction, label text.
3. **Detect containers**: Identify groups of elements enclosed by larger boxes or indentation patterns.
4. **Detect text labels**: Titles, standalone text, annotations.
5. **Infer layers**: Determine if the diagram has horizontal layers (top-to-bottom) or vertical columns.
6. **Assign colors**: Use keyword matching on box labels (see Color Assignment below).

#### 1b. Text Width Estimation (CRITICAL for proportion)

Before computing any coordinates, estimate the rendered width of every text element:

| Text Type | Approx char width (px) | Example |
|---|---|---|
| Shape label (fontFamily 5, fontSize 16-18) | ~9-10px per char | `"API Server"` (10 chars) → ~95px |
| Body text / code snippet (fontSize 14-15) | ~8-9px per char | `"returns[i] = ones_like(kl[i])"` → ~250px |
| Annotation badge (fontSize 12-14) | ~7-8px per char | `"标量奖励广播"` → ~75px |
| Title (fontSize 24-28) | ~14-16px per char | `"System Architecture"` → ~280px |

For monospace/code text, add a 10% buffer. For CJK text, use ~12-14px per char.

#### 1c. Compute Layout Dimensions

Using the text widths from 1b, compute:

1. **Shape widths**: `max(text_width + 30px padding, min_width)`. Min widths: 140x70 (single-line), 180x90 (multi-line).
2. **Row height**: `max(shape_height, line1_text_height, line2_text_height) + 40px buffer`.
3. **Total canvas width**: `left_margin + shape_column_width + gap + description_column_width + right_margin`.
4. **Total canvas height**: `top_margin + num_rows * row_height + bottom_margin`.
5. **Background zone**: extends 80px beyond the bounding box of all content.

#### 1d. Proportion Check

Before proceeding to Step 2, verify:
- **Width/Height ratio**: if ratio > 3 or < 0.5, consider reflowing (e.g., swap horizontal layout for vertical, or split wide rows).
- **No overflow**: ensure description text does not exceed its allocated width — if it does, either increase canvas width or split into multiple lines.
- **No overlap**: verify that `shape_width + gap < column_width` for all columns.

If proportion issues are detected, **recompute the layout** (adjust canvas width, reduce font sizes, or increase spacing) before generating JSON.

#### 1e. Produce Module Plan

Output a structured plan:

```
Diagram Analysis:
- Title: "compute_advantages_and_returns() dispatch"
- Layout: left-to-right (method column + description column), 7 rows
- Canvas: ~830px wide × ~750px tall (estimated)
- Proportion: width/height = 1.1 ✓

Module 1 (background): Outer zone [gray]
Module 2 (header): Title box [bold]
Module 3 (connector): Vertical dispatch line
Module 4 (branches): 7 horizontal arrows
Module 5 (methods): 7 method boxes [varied colors]
Module 6 (descriptions): 7 description lines + 7 annotation badges

Method list (name → color, estimated shape width):
  GRPO → blue (#1971c2), ~120px
  GSPO → teal (#087f93), ~120px
  PPO → violet (#7048e8), ~110px
  step_wise → green (#2f9e44), ~160px
  reinforce_plus_plus → orange (#f08c00), ~220px
  reinforce_plus_plus_baseline → pink (#e64980), ~260px
  on_policy_distillation → teal (#087f5b), ~220px
```

### Step 2: Generate JSON Module by Module

Generate Excalidraw JSON for each identified module **separately**. Each module output is a JSON array of element objects.

**Module ordering**: Follow drawing order — background zones first, then layers top-to-bottom, then connections.

For each module:
1. Emit background zone rectangles (if any) with `opacity: 35` and `fillStyle: "solid"`
2. Emit shapes with their bound text elements
3. Emit arrows and their label text
4. Use semantic colors (see Color Assignment below)
5. Apply sketch style (roughness: 1-2, fillStyle: "hachure", fontFamily: 5)

**Between modules**, maintain consistent coordinate space. Track cursor positions so subsequent modules align properly.

Write each module to a temporary JSON file:
```
/tmp/excalidraw_modules/module_1.json
/tmp/excalidraw_modules/module_2.json
...
```

### Step 3: Merge All Modules

Use the helper script to assemble all module arrays into the final `.excalidraw` envelope:

```bash
python ~/.skills/ascii-excalidraw/scripts/merge_modules.py \
  -o diagram.excalidraw \
  /tmp/excalidraw_modules/module_1.json \
  /tmp/excalidraw_modules/module_2.json \
  ...
```

The script handles ID deduplication, binding reference updates, and produces the standard `.excalidraw` format.

---

## ASCII Pattern Mapping

| ASCII Pattern | Excalidraw Type | Style |
|---|---|---|
| `+------+` or `┌──┐` corners | `rectangle` with `roundness: { "type": 3 }` | Standard components |
| `( oval )` or wavy borders | `ellipse` | Databases, cloud services |
| `< text >` or `◇` | `diamond` | Decision points, gates |
| `═══` double lines | `rectangle` with `strokeWidth: 3` | Emphasized nodes |
| `- - -` dashed borders | shape with `strokeStyle: "dashed"` | Optional/proxy components |
| `--->` or `──▶` | `arrow` with `endArrowhead: "arrow"` | Directional flow |
| `---` plain line | `arrow` with `endArrowhead: null` | Undirected connection |
| `- ->` dashed arrow | `arrow` with `strokeStyle: "dashed"` | Optional/async flow |
| `│` vertical line | `arrow` with vertical `points` | Vertical flow |
| Large box enclosing others | `rectangle` with `opacity: 35` | Background zone |
| `=== Title ===` | `text` with `fontSize: 24+` | Section header |
| Text inside a box | Bound text (`containerId` + `boundElements`) | Shape label |
| Text above/below arrow | Bound text on arrow (`containerId`) | Arrow label |

---

## Color Assignment

Automatically assign colors based on **keyword detection** in box label text:

| Keywords | Fill | Stroke | Use |
|---|---|---|---|
| web, app, frontend, ui, client, browser, vue, react | `#a5d8ff` | `#1971c2` | Frontend/UI |
| api, server, backend, service, proxy, gateway, nginx | `#d0bfff` | `#7048e8` | Backend/Services |
| db, database, postgres, mysql, redis, mongo, cache, data | `#b2f2bb` | `#2f9e44` | Data/Storage |
| external, third-party, stripe, aws, s3, cdn, webhook | `#ffc9c9` | `#e03131` | External services |
| queue, rabbitmq, kafka, sqs, event, stream, pubsub | `#fff3bf` | `#f08c00` | Message/Queue |
| ai, ml, model, llm, embed, vector, rag, prompt | `#eebefa` | `#9c36b5` | AI/ML |
| auth, login, oauth, token, jwt, permission | `#c3fae8` | `#087f5b` | Authentication |
| log, monitor, metric, alert, trace, observ | `#fcc2d7` | `#e64980` | Monitoring |
| (no match) | `#f8f9fa` | `#495057` | General purpose |

**Rules**:
- Background zones: same color family with `opacity: 35`, `fillStyle: "solid"`
- Arrow strokeColor: match source shape's stroke color, or `#1e1e1e` for neutral
- Arrow labels: same color as the arrow stroke

---

## Sketch Style Rules

Apply these properties to ALL elements:

- `roughness`: 1 (default sketch) or 2 (more sketchy)
- `fillStyle`: `"hachure"` for filled shapes; `"solid"` for background zones
- `fontFamily`: **5** (Excalifont) for all text — this is the key to the sketch look
- `strokeWidth`: 2 (default), 3 for emphasized elements
- `strokeStyle`: `"solid"` (default), `"dashed"` for optional/async connections

---

## Sizing and Layout

### Font Sizes
- Titles: 24-28px
- Shape labels: 16-20px
- Arrow labels: 14-16px
- Annotations: 14px minimum — NEVER below 14px

### Text Width Estimation
Estimate rendered widths **before** computing coordinates (see Step 1b):

| Text Type | Approx char width (px) | Notes |
|---|---|---|
| Shape label (fontFamily 5, fontSize 16-18) | ~9-10px | `"API Server"` → ~95px |
| Body text / code (fontSize 14-15) | ~8-9px | Add 10% for code/monospace |
| Annotation badge (fontSize 12-14) | ~7-8px | |
| Title (fontSize 24-28) | ~14-16px | |
| CJK characters | ~12-14px | Per character |

### Element Sizes
- Shape width: `max(text_width + 30px padding, min_width)`. Min widths: 140x70 (single-line), 180x90 (multi-line).
- Ellipses: minimum 120x120
- Diamonds: minimum 140x100
- Gap between sibling shapes: 50px horizontal, 60px vertical minimum
- Layer gap (top-to-bottom flow): 120px minimum
- Column gap (left-to-right flow): 150px minimum
- Background zones: extend 80px beyond contained elements on all sides

### Multi-line Text Handling
When a single text element would be too wide:
- Split into multiple text elements (each on its own line, same y + fontSize * 1.25 offset)
- Or reduce font size (but never below 14px)
- Annotation text: prefer short phrases; split long annotations into separate badges

### Proportion Rules
- **Target width/height ratio**: 0.8 to 1.5 for best display. If outside this range, consider reflowing the layout.
- **Column alignment**: when layout has distinct columns (e.g., shape column + description column), align all rows to the same baseline.
- **No element should extend beyond its column**: if description text overflows, either widen canvas or increase row height.

### Drawing Order (z-order)
1. Title text (standalone)
2. Background zone rectangles
3. Zone label text
4. Shapes in reading order (top-left → bottom-right)
5. Each shape's bound text **immediately after** the shape
6. Arrows and their bound labels
7. Decorative/annotation elements last

---

## Container Binding Pattern (CRITICAL)

For labeled shapes, use bidirectional binding:

```json
{
  "type": "rectangle",
  "id": "box_api",
  "x": 200, "y": 150, "width": 180, "height": 80,
  "backgroundColor": "#d0bfff", "fillStyle": "hachure",
  "strokeColor": "#7048e8", "roughness": 1,
  "roundness": { "type": 3 },
  "boundElements": [
    { "id": "t_box_api", "type": "text" },
    { "id": "arrow_1", "type": "arrow" }
  ]
},
{
  "type": "text",
  "id": "t_box_api",
  "x": 210, "y": 170, "width": 160, "height": 30,
  "text": "API Server",
  "fontSize": 18, "fontFamily": 5,
  "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "box_api",
  "originalText": "API Server",
  "autoResize": true
}
```

The shape's `boundElements` MUST list the text element ID, and the text MUST have `containerId` pointing back.

---

## Arrow Binding Pattern

```json
{
  "type": "arrow",
  "id": "arrow_1",
  "x": 380, "y": 190, "width": 150, "height": 0,
  "points": [[0, 0], [150, 0]],
  "endArrowhead": "arrow",
  "strokeColor": "#7048e8", "strokeWidth": 2, "roughness": 1,
  "startBinding": { "elementId": "box_api", "fixedPoint": [1, 0.5] },
  "endBinding": { "elementId": "box_db", "fixedPoint": [0, 0.5] }
}
```

`fixedPoint`: `[horizontal_ratio, vertical_ratio]` where 0=top/left, 1=bottom/right, 0.5=center.

---

## Common Pitfalls

1. **Never use `"label"` property** on shapes — use container binding (`containerId` + `boundElements`)
2. **Both sides of binding must exist** — shape needs `boundElements` AND text needs `containerId`
3. **Always include `fontFamily: 5`** on text for Excalifont sketch style
4. **Always include `originalText` and `autoResize: true`** on text elements
5. **Don't generate all at once** — break into modules, generate incrementally
6. **ID uniqueness** — every element needs a globally unique `id` across all modules
7. **Coordinate continuity** — track positions between modules
8. **No emoji in text** — Excalidraw fonts don't render emoji reliably
9. **Text contrast** — minimum `#757575` on white backgrounds; use dark variants on colored fills
10. **Drawing order matters** — emit shape → its text → its arrows in sequence, NOT all rectangles then all texts then all arrows
