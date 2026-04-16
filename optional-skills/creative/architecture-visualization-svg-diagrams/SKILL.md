---
name: architecture-visualization-svg-diagrams
description: Generate beautiful, consistent SVG architecture diagrams and host them as interactive web pages. Use this skill whenever the user asks to visualize, diagram, draw, or illustrate any system architecture, infrastructure layout, API map, data flow, CI/CD pipeline, microservice topology, deployment diagram, network topology, or any technical system overview. Also trigger when the user says "show me the architecture", "draw what you built", "visualize the system", "diagram this", or asks for a visual summary of code, infrastructure, or services. After generating the diagram, automatically host it on 0.0.0.0:22223 so the user can view it in a browser. Always use this skill even for simple architecture requests — it ensures visual consistency across all diagrams.
---

# Architecture Diagram Skill

Generate production-quality SVG architecture diagrams with a unified design system, then host them as interactive web pages on `0.0.0.0:22223`.

Every diagram you produce MUST follow this design system exactly. No exceptions. This ensures visual consistency regardless of subject matter.

---

## Design System

### Philosophy

- **Flat**: No gradients, drop shadows, blur, glow, or neon effects. Clean flat surfaces only.
- **Minimal**: Show the essential. No decorative icons, illustrations inside boxes, or ornamental elements.
- **Consistent**: Same colors, spacing, typography, and stroke widths across every diagram.
- **Dark-mode ready**: All colors auto-adapt to light and dark modes via CSS classes.

### Color Palette

9 color ramps, each with 7 stops. Use the class names (`c-purple`, `c-teal`, etc.) on SVG group (`<g>`) or shape elements — they handle light/dark mode automatically.

| Class | 50 (lightest) | 100 | 200 | 400 | 600 | 800 | 900 (darkest) |
|------------|-----------|---------|---------|---------|---------|---------|-----------|
| `c-purple` | #EEEDFE | #CECBF6 | #AFA9EC | #7F77DD | #534AB7 | #3C3489 | #26215C |
| `c-teal` | #E1F5EE | #9FE1CB | #5DCAA5 | #1D9E75 | #0F6E56 | #085041 | #04342C |
| `c-coral` | #FAECE7 | #F5C4B3 | #F0997B | #D85A30 | #993C1D | #712B13 | #4A1B0C |
| `c-pink` | #FBEAF0 | #F4C0D1 | #ED93B1 | #D4537E | #993556 | #72243E | #4B1528 |
| `c-gray` | #F1EFE8 | #D3D1C7 | #B4B2A9 | #888780 | #5F5E5A | #444441 | #2C2C2A |
| `c-blue` | #E6F1FB | #B5D4F4 | #85B7EB | #378ADD | #185FA5 | #0C447C | #042C53 |
| `c-green` | #EAF3DE | #C0DD97 | #97C459 | #639922 | #3B6D11 | #27500A | #173404 |
| `c-amber` | #FAEEDA | #FAC775 | #EF9F27 | #BA7517 | #854F0B | #633806 | #412402 |
| `c-red` | #FCEBEB | #F7C1C1 | #F09595 | #E24B4A | #A32D2D | #791F1F | #501313 |

#### Color Assignment Rules

Color encodes **meaning**, not sequence. Never cycle through colors like a rainbow.

- Group nodes by **category** — all nodes of the same type share one color.
- Use **gray** (`c-gray`) for neutral/structural nodes (start, end, generic steps, users).
- Use **2-3 colors per diagram**, not 6+. More colors = more noise.
- Prefer `c-purple`, `c-teal`, `c-coral`, `c-pink` for general categories.
- Reserve `c-blue`, `c-green`, `c-amber`, `c-red` for semantic meaning (info, success, warning, error).

**Light/dark mode stops:**
- Light mode: 50 fill + 600 stroke + 800 title / 600 subtitle
- Dark mode: 800 fill + 200 stroke + 100 title / 200 subtitle

### Typography

Only two font sizes. No exceptions.

| Class | Size | Weight | Use |
|-------|------|--------|-----|
| `th` | 14px | 500 (medium) | Node titles, region labels |
| `ts` | 12px | 400 (regular) | Subtitles, descriptions, arrow labels |
| `t` | 14px | 400 (regular) | General text |

- **Sentence case always.** Never Title Case, never ALL CAPS.
- Every `<text>` element MUST carry a class (`t`, `ts`, or `th`). No unclassed text.
- `dominant-baseline="central"` on all text inside boxes.
- `text-anchor="middle"` for centered text in boxes.

**Font width estimation** (Anthropic Sans):
- 14px weight 500: ~8px per character
- 12px weight 400: ~6.5px per character
- Always verify: `box_width >= (char_count × px_per_char) + 48` (24px padding each side)

### Spacing & Layout

- **ViewBox**: Always `viewBox="0 0 680 H"` where H is content height + 40px buffer.
- **Safe area**: x=40 to x=640, y=40 to y=(H-40).
- **Between boxes**: 60px minimum gap.
- **Inside boxes**: 24px horizontal padding, 12px vertical padding.
- **Arrowhead gap**: 10px between arrowhead and box edge.
- **Single-line box**: 44px height.
- **Two-line box**: 56px height, 18px between title and subtitle baselines.
- **Container padding**: 20px minimum inside every container.
- **Max nesting**: 2-3 levels deep. Deeper gets unreadable at 680px width.

### Stroke & Shape

- **Stroke width**: 0.5px for all borders and edges. Not 1px. Not 2px.
- **Rect rounding**: `rx="8"` for nodes, `rx="12"` for inner containers, `rx="16"` to `rx="20"` for outer containers.
- **Connector paths**: MUST have `fill="none"`. SVG defaults to `fill: black` otherwise.

### Arrow Marker

Include this `<defs>` block at the start of **every** SVG:

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

Use `marker-end="url(#arrow)"` on lines. The head inherits the line color via `context-stroke`.

### CSS Classes (Embedded in Host Page)

The hosting HTML page includes these styles. Use these class names directly in your SVG:

```css
/* Text classes */
.t  { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; fill: var(--text-primary); }
.ts { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; fill: var(--text-secondary); }
.th { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; fill: var(--text-primary); font-weight: 500; }

/* Neutral box */
.box { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.5px; }

/* Arrow line */
.arr { stroke: var(--text-secondary); stroke-width: 1.5px; fill: none; }

/* Clickable node hover */
.node { cursor: pointer; }
.node:hover { opacity: 0.85; }

/* Dashed leader line */
.leader { stroke: var(--text-tertiary); stroke-width: 0.5px; stroke-dasharray: 4 3; fill: none; }
```

### Node Patterns

#### Single-line node (44px)

```xml
<g class="node c-blue">
  <rect x="100" y="20" width="180" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service name</text>
</g>
```

#### Two-line node (56px)

```xml
<g class="node c-teal">
  <rect x="100" y="20" width="200" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="200" y="38" text-anchor="middle" dominant-baseline="central">Service name</text>
  <text class="ts" x="200" y="56" text-anchor="middle" dominant-baseline="central">Short description</text>
</g>
```

#### Connector (no label)

```xml
<line x1="200" y1="76" x2="200" y2="120" class="arr" marker-end="url(#arrow)"/>
```

#### Container (dashed or solid)

```xml
<g class="c-purple">
  <rect x="40" y="92" width="600" height="300" rx="16" stroke-width="0.5"/>
  <text class="th" x="66" y="116">Container label</text>
  <text class="ts" x="66" y="134">Subtitle info</text>
</g>
```

---

## SVG Boilerplate

Every diagram MUST start with this exact structure:

```xml
<svg width="100%" viewBox="0 0 680 {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Diagram content here -->

</svg>
```

Replace `{HEIGHT}` with the actual computed height (last element bottom + 40px).

---

## Diagram Types

### 1. Flowchart
For: CI/CD pipelines, request lifecycles, approval workflows, data processing.
Layout: Single-direction flow (top-down or left-right). Max 4-5 nodes per row.

### 2. Structural / Containment
For: Cloud infrastructure (VPC/subnet/instance), system architecture with nesting.
Layout: Large outer containers with inner regions. Dashed rects for logical groupings.

### 3. API / Endpoint Map
For: REST API routes, GraphQL schema overview, service endpoint inventory.
Layout: Tree from root, branching to resource groups, each containing endpoint nodes.

### 4. Microservice Topology
For: Service mesh, inter-service communication, event-driven systems.
Layout: Services as nodes, arrows showing communication patterns, message queues between.

### 5. Data Flow
For: ETL pipelines, streaming architectures, data lake layouts.
Layout: Left-to-right flow from sources through processing to sinks.

### 6. Physical / Structural
For: Physical objects, vehicles, buildings, hardware, mechanical systems, anatomical diagrams.
Layout: Use shapes that match the physical form — not just rectangles.

**Go beyond boxes** — when diagramming physical objects, use appropriate SVG elements:

| Physical form | SVG element | Example use |
|---------------|-------------|-------------|
| Curved bodies | `<path>` with Q/C curves | Fuselage, tanks, pipes |
| Tapered/angular shapes | `<polygon>` | Wings, fins, wedges |
| Cylindrical/round | `<ellipse>`, `<circle>` | Engines, wheels, buttons |
| Linear structures | `<line>` | Struts, beams, connections |
| Internal sections | `<rect>` inside parent | Compartments, rooms |
| Dashed boundaries | `stroke-dasharray` | Hidden parts, fuel tanks |

**Layering approach:**
1. Draw outer structure first (fuselage, frame, hull)
2. Add internal sections on top (cabins, compartments)
3. Add detail elements (engines, wheels, controls)
4. Add leader lines with labels

**CSS classes for physical diagrams:**
Define semantic classes per component type instead of using `c-*` color classes:
```css
.fuselage { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.wing { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.engine { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
```

### 7. Infrastructure / Systems Integration
For: Smart cities, IoT networks, industrial systems, multi-domain architectures.
Layout: Hub-spoke with central platform connecting multiple subsystems.

**Key patterns:**
- **Central hub**: Hexagon or circle representing integration platform
- **Radiating connections**: Data lines from hub to each subsystem with connection dots
- **Subsystem sections**: Each system (power, water, transport, etc.) in its own region
- **Dashboard on top**: Optional UI mockup showing unified view

**Hub-spoke layout:**
```xml
<!-- Central hub (hexagon) -->
<polygon class="iot-hex" points="0,-45 39,-22 39,22 0,45 -39,22 -39,-22"/>

<!-- Data lines with connection dots -->
<path class="data-line" d="M 321 248 L 200 248 L 120 380" stroke-dasharray="4 3"/>
<circle cx="321" cy="248" r="4" fill="#7F77DD"/>
```

**Semantic line styles for different systems:**
```css
.data-line { stroke: #7F77DD; stroke-width: 2; fill: none; stroke-dasharray: 4 3; }
.power-line { stroke: #EF9F27; stroke-width: 2; fill: none; }
.water-pipe { stroke: #378ADD; stroke-width: 4; stroke-linecap: round; fill: none; }
.road { stroke: #888780; stroke-width: 8; stroke-linecap: round; fill: none; }
```

### 8. UI / Dashboard Mockups
For: Admin panels, monitoring dashboards, control interfaces, status displays.
Layout: Screen frame with nested chart/gauge/indicator elements.

**Dashboard structure:**
```xml
<!-- Monitor frame -->
<rect class="dashboard" x="0" y="0" width="200" height="120" rx="8"/>
<!-- Screen -->
<rect class="screen" x="10" y="10" width="180" height="85" rx="4"/>
<!-- Mini bar chart -->
<rect class="screen-content" x="18" y="18" width="50" height="35" rx="2"/>
<rect class="screen-chart" x="22" y="38" width="8" height="12"/>
<rect class="screen-chart" x="33" y="32" width="8" height="18"/>
<!-- Gauge -->
<circle class="screen-bar" cx="100" cy="35" r="12"/>
<text x="100" y="39" text-anchor="middle" fill="#E8E6DE" style="font-size:8px">78%</text>
<!-- Status indicators -->
<circle cx="35" cy="74" r="6" fill="#97C459"/> <!-- green = ok -->
<circle cx="75" cy="74" r="6" fill="#EF9F27"/> <!-- amber = warning -->
```

**Dashboard CSS:**
```css
.dashboard { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1.5; }
.screen { fill: #1a1a18; }
.screen-content { fill: #2C2C2A; }
.screen-chart { fill: #5DCAA5; }
.screen-bar { fill: #7F77DD; }
.screen-alert { fill: #E24B4A; }
```

---

## Infrastructure Component Library

Reusable shapes for infrastructure diagrams:

### Power Systems

**Solar panel (angled):**
```xml
<polygon class="solar-panel" points="0,25 35,8 38,12 3,29"/>
<line class="solar-frame" x1="12" y1="22" x2="24" y2="13"/>
```

**Wind turbine:**
```xml
<polygon class="wind-tower" points="20,70 30,70 28,25 22,25"/>
<circle class="wind-hub" cx="25" cy="18" r="5"/>
<ellipse class="wind-blade" cx="25" cy="5" rx="3" ry="13"/>
<ellipse class="wind-blade" cx="14" cy="26" rx="3" ry="13" transform="rotate(-120, 25, 18)"/>
<ellipse class="wind-blade" cx="36" cy="26" rx="3" ry="13" transform="rotate(120, 25, 18)"/>
```

**Battery with charge level:**
```xml
<rect class="battery" x="0" y="0" width="45" height="65" rx="5"/>
<rect x="10" y="-6" width="10" height="8" rx="2" fill="#27500A"/> <!-- terminal -->
<rect class="battery-level" x="5" y="12" width="35" height="48" rx="3"/> <!-- fill level -->
```

**Power pylon:**
```xml
<polygon class="pylon" points="30,0 35,0 40,60 25,60"/>
<line x1="15" y1="10" x2="45" y2="10" stroke="#5F5E5A" stroke-width="3"/>
<circle cx="18" cy="10" r="3" fill="#FAEEDA" stroke="#854F0B"/> <!-- insulator -->
```

### Water Systems

**Reservoir/dam:**
```xml
<polygon class="reservoir-wall" points="0,60 10,0 70,0 80,60"/>
<polygon class="water" points="12,10 68,10 68,55 75,55 75,58 5,58 5,55 12,55"/>
<!-- Wave effect -->
<path d="M 15 25 Q 25 22 35 25 Q 45 28 55 25" fill="none" stroke="#378ADD" opacity="0.5"/>
```

**Treatment tank:**
```xml
<ellipse class="treatment-tank" cx="35" cy="45" rx="30" ry="18"/>
<rect class="treatment-tank" x="5" y="20" width="60" height="25"/>
<!-- Bubbles -->
<circle cx="20" cy="32" r="2" fill="#378ADD" opacity="0.6"/>
```

**Pipe with joint and valve:**
```xml
<path class="pipe" d="M 80 85 L 110 85"/>
<circle class="pipe-joint" cx="110" cy="85" r="8"/>
<circle class="valve" cx="95" cy="85" r="6"/>
```

### Transport Systems

**Road with lane markings:**
```xml
<line class="road" x1="0" y1="50" x2="170" y2="50"/>
<line class="road-mark" x1="10" y1="50" x2="160" y2="50"/>
```

**Traffic light:**
```xml
<rect class="traffic-light" x="0" y="0" width="14" height="32" rx="3"/>
<circle class="light-red" cx="7" cy="8" r="4"/>
<circle class="light-off" cx="7" cy="16" r="4"/>
<circle class="light-green" cx="7" cy="24" r="4"/>
```

**Bus:**
```xml
<rect class="bus" x="0" y="0" width="55" height="28" rx="6"/>
<rect class="bus-window" x="5" y="5" width="12" height="12" rx="2"/>
<circle cx="14" cy="30" r="6" fill="#2C2C2A"/> <!-- wheel -->
<circle cx="14" cy="30" r="3" fill="#5F5E5A"/> <!-- hubcap -->
```

### Infrastructure CSS

```css
/* Power */
.solar-panel { fill: #3C3489; stroke: #534AB7; stroke-width: 0.5; }
.wind-tower { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.wind-blade { fill: #F1EFE8; stroke: #888780; stroke-width: 0.5; }
.battery { fill: #27500A; stroke: #3B6D11; stroke-width: 1.5; }
.battery-level { fill: #97C459; }
.power-line { stroke: #EF9F27; stroke-width: 2; fill: none; }

/* Water */
.reservoir-wall { fill: #B4B2A9; stroke: #5F5E5A; stroke-width: 1; }
.water { fill: #85B7EB; stroke: #378ADD; stroke-width: 0.5; }
.pipe { fill: none; stroke: #378ADD; stroke-width: 4; stroke-linecap: round; }
.pipe-joint { fill: #185FA5; stroke: #0C447C; stroke-width: 1; }
.valve { fill: #0C447C; stroke: #185FA5; stroke-width: 1; }

/* Transport */
.road { stroke: #888780; stroke-width: 8; fill: none; stroke-linecap: round; }
.road-mark { stroke: #F1EFE8; stroke-width: 1; stroke-dasharray: 6 4; fill: none; }
.traffic-light { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }
.light-red { fill: #E24B4A; }
.light-green { fill: #97C459; }
.light-off { fill: #2C2C2A; }
.bus { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 1.5; }
```

---

## Validation Checklist

Before finalizing any SVG, verify ALL of the following:

1. Every `<text>` has class `t`, `ts`, or `th`.
2. Every `<text>` inside a box has `dominant-baseline="central"`.
3. Every connector `<path>` or `<line>` used as arrow has `fill="none"`.
4. No arrow line crosses through an unrelated box.
5. `box_width >= (longest_label_chars × 8) + 48` for 14px text.
6. `box_width >= (longest_label_chars × 6.5) + 48` for 12px text.
7. ViewBox height = bottom-most element + 40px.
8. All content stays within x=40 to x=640.
9. Color classes (`c-*`) are on `<g>` or shape elements, never on `<path>` connectors.
10. Arrow `<defs>` block is present.
11. No gradients, shadows, blur, or glow effects.
12. Stroke width is 0.5px on all node borders.

---

## Examples

Below are complete architecture diagram examples. Use these as templates.

### Example 1: Data pipeline (ETL flow)

```xml
<svg width="100%" viewBox="0 0 680 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Tier labels -->
  <text class="ts" x="40" y="68" text-anchor="start" opacity=".5">Clients</text>
  <text class="ts" x="40" y="178" text-anchor="start" opacity=".5">Gateway</text>
  <text class="ts" x="40" y="308" text-anchor="start" opacity=".5">Services</text>
  <text class="ts" x="40" y="448" text-anchor="start" opacity=".5">Data</text>

  <!-- Client tier -->
  <g class="node c-gray">
    <rect x="120" y="40" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="200" y="60" text-anchor="middle" dominant-baseline="central">React SPA</text>
    <text class="ts" x="200" y="80" text-anchor="middle" dominant-baseline="central">Next.js frontend</text>
  </g>
  <g class="node c-gray">
    <rect x="320" y="40" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="400" y="60" text-anchor="middle" dominant-baseline="central">Mobile app</text>
    <text class="ts" x="400" y="80" text-anchor="middle" dominant-baseline="central">React Native</text>
  </g>
  <g class="node c-gray">
    <rect x="520" y="40" width="120" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="580" y="60" text-anchor="middle" dominant-baseline="central">CLI</text>
    <text class="ts" x="580" y="80" text-anchor="middle" dominant-baseline="central">Node.js</text>
  </g>

  <!-- Arrows: clients to gateway -->
  <line x1="200" y1="96" x2="340" y2="150" class="arr" marker-end="url(#arrow)"/>
  <line x1="400" y1="96" x2="380" y2="150" class="arr" marker-end="url(#arrow)"/>
  <line x1="580" y1="96" x2="420" y2="150" class="arr" marker-end="url(#arrow)"/>

  <!-- API Gateway -->
  <g class="node c-purple">
    <rect x="220" y="150" width="280" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="360" y="170" text-anchor="middle" dominant-baseline="central">API gateway</text>
    <text class="ts" x="360" y="190" text-anchor="middle" dominant-baseline="central">Auth, rate limiting, routing</text>
  </g>

  <!-- Arrows: gateway to services -->
  <line x1="290" y1="206" x2="180" y2="280" class="arr" marker-end="url(#arrow)"/>
  <line x1="360" y1="206" x2="360" y2="280" class="arr" marker-end="url(#arrow)"/>
  <line x1="430" y1="206" x2="530" y2="280" class="arr" marker-end="url(#arrow)"/>

  <!-- Backend services -->
  <g class="node c-teal">
    <rect x="100" y="280" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="180" y="300" text-anchor="middle" dominant-baseline="central">User service</text>
    <text class="ts" x="180" y="320" text-anchor="middle" dominant-baseline="central">Auth, profiles</text>
  </g>
  <g class="node c-teal">
    <rect x="280" y="280" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="360" y="300" text-anchor="middle" dominant-baseline="central">Order service</text>
    <text class="ts" x="360" y="320" text-anchor="middle" dominant-baseline="central">Cart, checkout</text>
  </g>
  <g class="node c-teal">
    <rect x="460" y="280" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="540" y="300" text-anchor="middle" dominant-baseline="central">Notification svc</text>
    <text class="ts" x="540" y="320" text-anchor="middle" dominant-baseline="central">Email, push, SMS</text>
  </g>

  <!-- Arrows: services to data -->
  <line x1="180" y1="336" x2="220" y2="420" class="arr" marker-end="url(#arrow)"/>
  <line x1="360" y1="336" x2="320" y2="420" class="arr" marker-end="url(#arrow)"/>
  <line x1="360" y1="336" x2="480" y2="420" class="arr" marker-end="url(#arrow)"/>
  <line x1="540" y1="336" x2="480" y2="420" class="arr" marker-end="url(#arrow)"/>

  <!-- Data stores -->
  <g class="node c-coral">
    <rect x="140" y="420" width="180" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="230" y="440" text-anchor="middle" dominant-baseline="central">PostgreSQL</text>
    <text class="ts" x="230" y="460" text-anchor="middle" dominant-baseline="central">Users, orders, products</text>
  </g>
  <g class="node c-coral">
    <rect x="400" y="420" width="160" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="480" y="440" text-anchor="middle" dominant-baseline="central">Redis</text>
    <text class="ts" x="480" y="460" text-anchor="middle" dominant-baseline="central">Sessions, queue</text>
  </g>
</svg>
```

### Example 2: ML inference pipeline

```xml
<svg width="100%" viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Input -->
  <g class="node c-gray">
    <rect x="40" y="50" width="140" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="110" y="70" text-anchor="middle" dominant-baseline="central">Client request</text>
    <text class="ts" x="110" y="90" text-anchor="middle" dominant-baseline="central">REST / gRPC</text>
  </g>
  <line x1="180" y1="78" x2="220" y2="78" class="arr" marker-end="url(#arrow)"/>

  <!-- Preprocessing -->
  <g class="node c-purple">
    <rect x="220" y="50" width="140" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="290" y="70" text-anchor="middle" dominant-baseline="central">Preprocessor</text>
    <text class="ts" x="290" y="90" text-anchor="middle" dominant-baseline="central">Tokenize, embed</text>
  </g>
  <line x1="360" y1="78" x2="400" y2="78" class="arr" marker-end="url(#arrow)"/>

  <!-- Model server -->
  <g class="c-teal">
    <rect x="400" y="30" width="240" height="230" rx="12" stroke-width="0.5"/>
    <text class="th" x="520" y="56" text-anchor="middle">Model server</text>
  </g>
  <g class="node c-teal">
    <rect x="420" y="70" width="200" height="50" rx="8" stroke-width="0.5"/>
    <text class="th" x="520" y="88" text-anchor="middle" dominant-baseline="central">GPU inference</text>
    <text class="ts" x="520" y="106" text-anchor="middle" dominant-baseline="central">vLLM / TensorRT</text>
  </g>
  <g class="node c-teal">
    <rect x="420" y="135" width="200" height="50" rx="8" stroke-width="0.5"/>
    <text class="th" x="520" y="153" text-anchor="middle" dominant-baseline="central">Model registry</text>
    <text class="ts" x="520" y="171" text-anchor="middle" dominant-baseline="central">MLflow versioning</text>
  </g>
  <g class="node c-teal">
    <rect x="420" y="200" width="200" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="520" y="222" text-anchor="middle" dominant-baseline="central">A/B router</text>
  </g>

  <!-- Output -->
  <line x1="520" y1="260" x2="520" y2="290" class="arr" marker-end="url(#arrow)"/>
  <g class="node c-coral">
    <rect x="420" y="290" width="200" height="44" rx="8" stroke-width="0.5"/>
    <text class="th" x="520" y="312" text-anchor="middle" dominant-baseline="central">Response + metrics</text>
  </g>

  <!-- Side: Cache -->
  <g class="node c-amber">
    <rect x="60" y="170" width="140" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="130" y="190" text-anchor="middle" dominant-baseline="central">Redis cache</text>
    <text class="ts" x="130" y="210" text-anchor="middle" dominant-baseline="central">Prompt cache</text>
  </g>
  <path d="M200 198 L350 198 L350 95 L420 95" fill="none" class="arr"
        stroke-dasharray="4 3" marker-end="url(#arrow)"/>
  <text class="ts" x="280" y="188" text-anchor="middle" opacity=".6">cache hit</text>

  <!-- Side: Monitoring -->
  <g class="node c-amber">
    <rect x="60" y="250" width="140" height="56" rx="8" stroke-width="0.5"/>
    <text class="th" x="130" y="270" text-anchor="middle" dominant-baseline="central">Prometheus</text>
    <text class="ts" x="130" y="290" text-anchor="middle" dominant-baseline="central">Latency, errors</text>
  </g>
  <line x1="200" y1="278" x2="420" y2="312" class="arr" stroke-dasharray="4 3" marker-end="url(#arrow)"/>
</svg>
```

### Example 3: Commercial aircraft structure (physical/structural)

```xml
<svg width="100%" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg">

  <!-- FUSELAGE - main body cylinder with nose cone -->
  <path class="fuselage" d="
    M 80 180
    Q 40 180 40 200
    Q 40 220 80 220
    L 560 220
    Q 580 220 580 200
    Q 580 180 560 180
    Z
  "/>
  
  <!-- Nose cone -->
  <path class="fuselage" d="
    M 80 180
    Q 50 180 35 200
    Q 50 220 80 220
  " fill="none" stroke-width="1"/>

  <!-- COCKPIT windows -->
  <path class="cockpit" d="
    M 45 190
    L 75 185
    L 75 200
    L 50 200
    Z
  "/>
  <line x1="55" y1="188" x2="55" y2="200" stroke="#534AB7" stroke-width="0.5"/>
  <line x1="65" y1="186" x2="65" y2="200" stroke="#534AB7" stroke-width="0.5"/>

  <!-- CABIN SECTIONS (inside fuselage) -->
  <!-- First class -->
  <rect class="first-class" x="85" y="183" width="50" height="34" rx="2"/>
  <text class="tl" x="110" y="203" text-anchor="middle">First</text>
  
  <!-- Business class -->
  <rect class="business-class" x="140" y="183" width="80" height="34" rx="2"/>
  <text class="tl" x="180" y="203" text-anchor="middle">Business</text>
  
  <!-- Economy class -->
  <rect class="economy-class" x="225" y="183" width="200" height="34" rx="2"/>
  <text class="tl" x="325" y="203" text-anchor="middle">Economy</text>

  <!-- CARGO HOLD (lower section indication) -->
  <line x1="85" y1="217" x2="520" y2="217" class="leader"/>
  <text class="tl" x="300" y="228" text-anchor="middle" opacity=".6">Cargo hold below deck</text>

  <!-- WING - main wing shape -->
  <polygon class="wing" points="
    200,220
    120,300
    130,305
    160,305
    340,235
    340,220
  "/>
  
  <!-- Wing fuel tank (dashed interior) -->
  <polygon class="fuel-tank" points="
    210,225
    150,280
    160,283
    180,283
    310,232
    310,225
  "/>
  <text class="tl" x="220" y="260" opacity=".7">Fuel</text>

  <!-- Flaps (trailing edge) -->
  <polygon class="flap" points="
    130,300
    120,305
    160,310
    165,305
  "/>
  <text class="tl" x="143" y="320">Flaps</text>

  <!-- ENGINE under wing -->
  <ellipse class="engine" cx="175" cy="285" rx="25" ry="12"/>
  <ellipse cx="155" cy="285" rx="8" ry="10" fill="none" stroke="#993C1D" stroke-width="0.5"/>
  <!-- Engine pylon -->
  <line x1="175" y1="273" x2="190" y2="245" stroke="#5F5E5A" stroke-width="2"/>
  <text class="tl" x="175" y="308" text-anchor="middle">Engine</text>

  <!-- TAIL SECTION -->
  <!-- Vertical stabilizer -->
  <polygon class="tail-v" points="
    520,180
    560,100
    580,100
    580,180
  "/>
  <text class="tl" x="565" y="150" text-anchor="middle">Vertical</text>
  <text class="tl" x="565" y="162" text-anchor="middle">stabilizer</text>
  
  <!-- Rudder -->
  <polygon points="575,105 590,105 590,178 580,178" fill="none" stroke="#185FA5" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text class="tl" x="595" y="145" opacity=".6">Rudder</text>

  <!-- Horizontal stabilizer -->
  <polygon class="tail-h" points="
    500,195
    460,175
    465,170
    580,170
    580,180
    520,195
  "/>
  <text class="tl" x="510" y="166">Horizontal stabilizer</text>
  
  <!-- Elevator -->
  <polygon points="462,174 450,168 455,163 467,169" fill="none" stroke="#185FA5" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text class="tl" x="440" y="158" opacity=".6">Elevator</text>

  <!-- LANDING GEAR -->
  <!-- Nose gear -->
  <line class="gear" x1="100" y1="220" x2="100" y2="260" stroke-width="3"/>
  <ellipse class="wheel" cx="100" cy="268" rx="8" ry="10"/>
  <text class="tl" x="100" y="290" text-anchor="middle">Nose gear</text>

  <!-- Main gear (under wing/fuselage junction) -->
  <line class="gear" x1="280" y1="220" x2="280" y2="270" stroke-width="4"/>
  <line class="gear" x1="268" y1="265" x2="292" y2="265" stroke-width="3"/>
  <ellipse class="wheel" cx="268" cy="278" rx="10" ry="12"/>
  <ellipse class="wheel" cx="292" cy="278" rx="10" ry="12"/>
  <text class="tl" x="280" y="302" text-anchor="middle">Main gear</text>

  <!-- LABELS with leader lines -->
  <!-- Cockpit label -->
  <line class="leader" x1="60" y1="175" x2="60" y2="140"/>
  <text class="ts" x="60" y="132" text-anchor="middle">Cockpit</text>

  <!-- Wing label -->
  <line class="leader" x1="250" y1="250" x2="290" y2="330"/>
  <text class="ts" x="290" y="345" text-anchor="middle">Wing structure</text>
  <text class="tl" x="290" y="358" text-anchor="middle">Spars, ribs, skin</text>

  <!-- Fuselage label -->
  <line class="leader" x1="400" y1="180" x2="400" y2="140"/>
  <text class="ts" x="400" y="132" text-anchor="middle">Fuselage</text>
  <text class="tl" x="400" y="145" text-anchor="middle">Pressure vessel</text>

</svg>
```

**Aircraft CSS classes (add to hosting page):**

```css
/* Structure shapes */
.fuselage { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.wing { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.tail-v { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.tail-h { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }

/* Interior sections */
.cockpit { fill: #EEEDFE; stroke: #534AB7; stroke-width: 1; }
.first-class { fill: #FBEAF0; stroke: #993556; stroke-width: 0.5; }
.business-class { fill: #FAECE7; stroke: #993C1D; stroke-width: 0.5; }
.economy-class { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }

/* Systems */
.engine { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
.fuel-tank { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.5; stroke-dasharray: 3 2; }
.flap { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.5; }

/* Mechanical */
.gear { fill: #444441; stroke: #2C2C2A; stroke-width: 0.5; }
.wheel { fill: #2C2C2A; stroke: #1a1a18; stroke-width: 0.5; }
```

**Shape selection for physical diagrams:**

| Physical form | SVG element | Example |
|---------------|-------------|--------|
| Curved body | `<path>` with Q/C curves | Fuselage, nose cone |
| Tapered/angular | `<polygon>` | Wings, stabilizers |
| Cylindrical | `<ellipse>` | Engines, wheels |
| Linear structure | `<line>` | Struts, pylons, gear legs |
| Internal sections | `<rect>` inside parent | Cabin classes |
| Dashed boundaries | `stroke-dasharray` | Fuel tanks, control surfaces |

### Example 4: SN2 reaction mechanism (chemistry)

**Chemistry CSS classes (add to hosting page):**

```css
/* Atom colors */
.carbon { fill: #2C2C2A; }
.hydrogen { fill: #F1EFE8; stroke: #888780; stroke-width: 1; }
.oxygen { fill: #E24B4A; }
.bromine { fill: #993C1D; }
.nitrogen { fill: #378ADD; }

/* Bond types */
.bond { stroke: var(--text-primary); stroke-width: 2.5; fill: none; stroke-linecap: round; }
.bond-thin { stroke: var(--text-primary); stroke-width: 1.5; fill: none; }
.bond-partial { stroke: var(--text-primary); stroke-width: 2; fill: none; stroke-dasharray: 4 3; }
.bond-wedge { fill: var(--text-primary); stroke: none; }
.bond-dash { stroke: var(--text-primary); stroke-width: 2; fill: none; stroke-dasharray: 2 2; }

/* Electron movement arrows */
.arrow-curved { stroke: #534AB7; stroke-width: 2; fill: none; }
.arrow-fill { fill: #534AB7; }

/* Transition state brackets */
.ts-bracket { stroke: var(--text-primary); stroke-width: 1.5; fill: none; }

/* Energy profile */
.energy-curve { stroke: #534AB7; stroke-width: 2.5; fill: none; }
.energy-fill { fill: rgba(83, 74, 183, 0.1); }
.energy-level { stroke: var(--text-secondary); stroke-width: 1; stroke-dasharray: 4 2; fill: none; }
.delta-arrow { stroke: #3B6D11; stroke-width: 1.5; fill: none; }

/* Chemistry text */
.chem { font-family: "Times New Roman", Georgia, serif; font-size: 16px; fill: var(--text-primary); }
.chem-sm { font-family: "Times New Roman", Georgia, serif; font-size: 12px; fill: var(--text-primary); }
.chem-lg { font-family: "Times New Roman", Georgia, serif; font-size: 18px; fill: var(--text-primary); }
.charge { font-family: "Times New Roman", Georgia, serif; font-size: 12px; }
.partial { font-family: "Times New Roman", Georgia, serif; font-size: 11px; font-style: italic; }
```

**Molecular structure building blocks:**

```xml
<!-- Atom rendering -->
<circle cx="0" cy="0" r="14" class="carbon"/>
<text class="chem" x="0" y="5" text-anchor="middle" fill="white" font-weight="500">C</text>

<circle cx="0" cy="0" r="14" class="oxygen"/>
<text class="chem" x="0" y="5" text-anchor="middle" fill="white" font-weight="500">O</text>

<circle cx="38" cy="0" r="8" class="hydrogen"/>
<text class="chem-sm" x="38" y="4" text-anchor="middle">H</text>

<circle cx="52" cy="0" r="16" class="bromine"/>
<text class="chem" x="52" y="5" text-anchor="middle" fill="white" font-weight="500">Br</text>
```

**Bond types:**

```xml
<!-- Single bond (solid) -->
<line x1="14" y1="0" x2="38" y2="0" class="bond"/>

<!-- Wedge bond (coming toward viewer) -->
<polygon class="bond-wedge" points="0,-14 -6,-35 6,-35"/>

<!-- Dash bond (going away from viewer) -->
<line x1="-10" y1="10" x2="-28" y2="28" class="bond-dash"/>

<!-- Partial bond (forming/breaking) -->
<line x1="-40" y1="0" x2="-14" y2="0" class="bond-partial"/>
```

**Lone pairs and charges:**

```xml
<!-- Lone pair electrons (dots) -->
<circle cx="-8" cy="-18" r="2" fill="var(--text-primary)"/>
<circle cx="0" cy="-18" r="2" fill="var(--text-primary)"/>

<!-- Formal negative charge -->
<text class="charge" x="12" y="-12" fill="#A32D2D" font-weight="bold">⊖</text>

<!-- Partial charges (delta notation) -->
<text class="partial" x="0" y="-18" text-anchor="middle" fill="#A32D2D">δ⁻</text>
<text class="partial" x="0" y="-22" text-anchor="middle" fill="#3B6D11">δ⁺</text>
```

**Curved arrow (electron movement):**

```xml
<defs>
  <marker id="curved-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,0 L10,5 L0,10 L3,5 Z" class="arrow-fill"/>
  </marker>
</defs>

<!-- Nucleophilic attack arrow -->
<path d="M -5,15 Q 30,60 70,25" class="arrow-curved" marker-end="url(#curved-arrow)"/>
```

**Transition state brackets:**

```xml
<!-- Left bracket -->
<path d="M -75,-70 L -85,-70 L -85,75 L -75,75" class="ts-bracket"/>

<!-- Right bracket -->
<path d="M 95,-70 L 105,-70 L 105,75 L 95,75" class="ts-bracket"/>

<!-- Double dagger symbol -->
<text class="chem" x="115" y="-60" fill="var(--text-primary)">‡</text>
```

**Energy profile diagram (goes below the mechanism):**

```xml
<!-- Axes -->
<line x1="0" y1="280" x2="0" y2="0" class="axis"/>
<text class="t" x="-15" y="-10" text-anchor="middle" transform="rotate(-90 -15 140)">Potential Energy</text>
<line x1="0" y1="280" x2="600" y2="280" class="axis"/>
<text class="t" x="580" y="305" text-anchor="middle">Reaction Coordinate</text>

<!-- Filled area under curve -->
<path class="energy-fill" d="
  M 40,200 
  Q 150,200 250,50 
  Q 350,200 500,220 
  L 500,280 L 40,280 Z
"/>

<!-- Curve line -->
<path class="energy-curve" d="
  M 40,200 
  Q 100,200 150,150
  Q 200,80 250,50 
  Q 300,80 350,150
  Q 400,210 500,220
"/>

<!-- Reactants level -->
<line x1="20" y1="200" x2="80" y2="200" stroke="#3B6D11" stroke-width="2"/>
<text class="ts" x="50" y="218" text-anchor="middle">Reactants</text>

<!-- Transition state peak -->
<circle cx="250" cy="50" r="5" fill="#534AB7"/>
<line x1="250" y1="50" x2="250" y2="280" class="energy-level"/>
<text class="ts" x="250" y="30" text-anchor="middle" fill="#534AB7" font-weight="500">Transition State [‡]</text>

<!-- Products level -->
<line x1="470" y1="220" x2="530" y2="220" stroke="#3B6D11" stroke-width="2"/>

<!-- Activation energy arrow -->
<line x1="100" y1="200" x2="100" y2="55" class="delta-arrow" marker-end="url(#delta-arrow)"/>
<text class="ts" x="85" y="125" text-anchor="end" fill="#3B6D11">E<tspan baseline-shift="sub" font-size="8">a</tspan></text>
```

**Chemistry color coding:**

| Element | Color | Hex |
|---------|-------|-----|
| Carbon | Dark gray | #2C2C2A |
| Hydrogen | Light cream | #F1EFE8 |
| Oxygen | Red | #E24B4A |
| Bromine | Brown | #993C1D |
| Nitrogen | Blue | #378ADD |
| Electron arrows | Purple | #534AB7 |
| Positive charge | Green | #3B6D11 |
| Negative charge | Red | #A32D2D |

**Layout notes for chemistry diagrams:**

- **ViewBox**: 800×680 (landscape for mechanism + energy profile)
- **Mechanism section**: y=60-300, showing reactants → TS → products
- **Energy profile**: y=320-630, with axes and curve
- **Atom sizes**: C/O/Br ~12-16px radius, H ~7-8px radius
- **Bond lengths**: ~25-40px between atom centers
- **Spacing**: ~140px between mechanism stages

### Example 5: ML benchmark grouped bar chart with dual axis

```xml
<svg width="100%" viewBox="0 0 680 660" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Grid lines -->
  <line x1="90" y1="70"  x2="590" y2="70"  class="grid"/>
  <line x1="90" y1="155" x2="590" y2="155" class="grid"/>
  <line x1="90" y1="240" x2="590" y2="240" class="grid"/>
  <line x1="90" y1="325" x2="590" y2="325" class="grid"/>

  <!-- Axes -->
  <line x1="90"  y1="68" x2="90"  y2="410" class="axis"/>
  <line x1="590" y1="68" x2="590" y2="410" class="axis"/>
  <line x1="90"  y1="410" x2="590" y2="410" class="axis"/>

  <!-- Left Y-axis ticks: tok/s (0, 4, 8, 12, 16) -->
  <line x1="85" y1="70"  x2="90" y2="70"  class="axis-tick"/>
  <line x1="85" y1="155" x2="90" y2="155" class="axis-tick"/>
  <line x1="85" y1="240" x2="90" y2="240" class="axis-tick"/>
  <line x1="85" y1="325" x2="90" y2="325" class="axis-tick"/>
  <line x1="85" y1="410" x2="90" y2="410" class="axis-tick"/>
  <text class="ts" x="82" y="74"  text-anchor="end">16</text>
  <text class="ts" x="82" y="159" text-anchor="end">12</text>
  <text class="ts" x="82" y="244" text-anchor="end">8</text>
  <text class="ts" x="82" y="329" text-anchor="end">4</text>
  <text class="ts" x="82" y="414" text-anchor="end">0</text>
  <text class="ts" x="30" y="240" text-anchor="middle" transform="rotate(-90 30 240)" font-weight="500">Tokens per second</text>

  <!-- Right Y-axis ticks: VRAM GB (0, 16, 32, 48, 64) -->
  <line x1="590" y1="70"  x2="595" y2="70"  class="axis-tick"/>
  <line x1="590" y1="155" x2="595" y2="155" class="axis-tick"/>
  <line x1="590" y1="240" x2="595" y2="240" class="axis-tick"/>
  <line x1="590" y1="325" x2="595" y2="325" class="axis-tick"/>
  <line x1="590" y1="410" x2="595" y2="410" class="axis-tick"/>
  <text class="ts" x="600" y="74"  text-anchor="start">64</text>
  <text class="ts" x="600" y="159" text-anchor="start">48</text>
  <text class="ts" x="600" y="244" text-anchor="start">32</text>
  <text class="ts" x="600" y="329" text-anchor="start">16</text>
  <text class="ts" x="600" y="414" text-anchor="start">0</text>
  <text class="ts" x="650" y="240" text-anchor="middle" transform="rotate(90 650 240)" font-weight="500">VRAM usage (GB)</text>

  <!-- 24 GB VRAM threshold -->
  <line x1="90" y1="283" x2="590" y2="283" class="threshold"/>
  <text class="threshold-label" x="588" y="277" text-anchor="end">RTX 4090 — 24 GB VRAM limit</text>
  <text class="tss" x="588" y="268" text-anchor="end" opacity=".35">Exceeds VRAM — offload to CPU</text>
  <text class="tss" x="588" y="296" text-anchor="end" opacity=".35">Fits in VRAM — full GPU speed</text>

  <!-- Grouped bars: FP16 min/max -->
  <rect x="114" y="399" width="34" height="11" rx="3" class="bar-fp16-min"/>
  <rect x="156" y="368" width="34" height="42" rx="3" class="bar-fp16-max"/>
  <!-- Q8_0 min/max -->
  <rect x="239" y="346" width="34" height="64" rx="3" class="bar-q8-min"/>
  <rect x="281" y="304" width="34" height="106" rx="3" class="bar-q8-max"/>
  <!-- Q4_K_M min/max -->
  <rect x="364" y="240" width="34" height="170" rx="3" class="bar-q4-min"/>
  <rect x="406" y="155" width="34" height="255" rx="3" class="bar-q4-max"/>
  <!-- IQ3_M min/max -->
  <rect x="489" y="155" width="34" height="255" rx="3" class="bar-iq3-min"/>
  <rect x="531" y="91"  width="34" height="319" rx="3" class="bar-iq3-max"/>

  <!-- Bar value labels -->
  <text class="ts" x="152" y="358" text-anchor="middle" fill="#A32D2D" font-weight="500">0.5–2 tok/s</text>
  <text class="tss" x="152" y="346" text-anchor="middle" fill="#A32D2D">OOM / unusable</text>
  <text class="ts" x="297" y="294" text-anchor="middle" fill="#854F0B" font-weight="500">3–5 tok/s</text>
  <text class="tss" x="297" y="282" text-anchor="middle" fill="#854F0B">Partial offload</text>
  <text class="ts" x="422" y="143" text-anchor="middle" fill="#0F6E56" font-weight="500">8–12 tok/s</text>
  <text class="tss" x="422" y="131" text-anchor="middle" fill="#0F6E56">Fits in VRAM</text>
  <text class="ts" x="547" y="83" text-anchor="middle" fill="#185FA5" font-weight="500">12–15 tok/s</text>
  <text class="tss" x="547" y="96" text-anchor="middle" fill="#185FA5">Full GPU speed</text>
  <text class="tss" x="422" y="119" text-anchor="middle" fill="#0F6E56" opacity=".65">▼ Best balance</text>

  <!-- X-axis category labels -->
  <text class="th" x="152" y="430" text-anchor="middle">FP16</text>
  <text class="tss" x="152" y="443" text-anchor="middle">~62 GB</text>
  <text class="th" x="297" y="430" text-anchor="middle">Q8_0</text>
  <text class="tss" x="297" y="443" text-anchor="middle">~32 GB</text>
  <text class="th" x="422" y="430" text-anchor="middle">Q4_K_M</text>
  <text class="tss" x="422" y="443" text-anchor="middle">~16.8 GB</text>
  <text class="th" x="547" y="430" text-anchor="middle">IQ3_M</text>
  <text class="tss" x="547" y="443" text-anchor="middle">~12 GB</text>

  <!-- VRAM line + dots -->
  <polyline points="152,81 277,240 402,321 527,346" class="vram-line"/>
  <circle cx="152" cy="81"  r="6" fill="var(--bg-primary)" opacity=".7"/>
  <circle cx="277" cy="240" r="6" fill="var(--bg-primary)" opacity=".7"/>
  <circle cx="402" cy="321" r="6" fill="var(--bg-primary)" opacity=".7"/>
  <circle cx="527" cy="346" r="6" fill="var(--bg-primary)" opacity=".7"/>
  <circle cx="152" cy="81"  r="5" class="vram-dot"/>
  <circle cx="277" cy="240" r="5" class="vram-dot"/>
  <circle cx="402" cy="321" r="5" class="vram-dot"/>
  <circle cx="527" cy="346" r="5" class="vram-dot"/>
  <text class="vram-label" x="168" y="77" text-anchor="start">62 GB</text>
  <text class="vram-label" x="262" y="236" text-anchor="end">32 GB</text>
  <text class="vram-label" x="420" y="334" text-anchor="start">16.8 GB</text>
  <text class="vram-label" x="542" y="356" text-anchor="start">12 GB</text>

  <!-- Legend -->
  <rect x="90" y="458" width="14" height="10" rx="2" fill="#D3D1C7" stroke="var(--text-tertiary)" stroke-width="0.5"/>
  <text class="tss" x="109" y="467">Min speed</text>
  <rect x="168" y="458" width="14" height="10" rx="2" fill="#888780" stroke="var(--text-tertiary)" stroke-width="0.5"/>
  <text class="tss" x="187" y="467">Max speed</text>
  <line x1="253" y1="463" x2="273" y2="463" class="vram-line" stroke-width="2.5"/>
  <circle cx="263" cy="463" r="3" class="vram-dot" stroke="none"/>
  <text class="tss" x="279" y="467">VRAM usage</text>
  <line x1="352" y1="463" x2="372" y2="463" class="threshold"/>
  <text class="tss" x="378" y="467">24 GB limit</text>

  <!-- Inset table: MMLU Pro accuracy -->
  <rect x="90" y="494" width="500" height="26" rx="4" class="tbl-header"/>
  <text class="ts" x="104" y="512" font-weight="500">Quantization</text>
  <text class="ts" x="245" y="512" font-weight="500" text-anchor="middle">Model size</text>
  <text class="ts" x="370" y="512" font-weight="500" text-anchor="middle">Speed (tok/s)</text>
  <text class="ts" x="485" y="512" font-weight="500" text-anchor="middle">MMLU Pro</text>
  <rect x="90" y="520" width="500" height="22" class="tbl-alt"/>
  <text class="ts" x="104" y="535" fill="#A32D2D" font-weight="500">FP16</text>
  <text class="ts" x="245" y="535" text-anchor="middle">62.0 GB</text>
  <text class="ts" x="370" y="535" text-anchor="middle">~1 (OOM)</text>
  <text class="ts" x="485" y="535" text-anchor="middle" font-weight="500">75.2</text>
  <rect x="90" y="542" width="500" height="22" class="tbl-row"/>
  <text class="ts" x="104" y="557" fill="#854F0B" font-weight="500">Q8_0</text>
  <text class="ts" x="245" y="557" text-anchor="middle">32.0 GB</text>
  <text class="ts" x="370" y="557" text-anchor="middle">3–5</text>
  <text class="ts" x="485" y="557" text-anchor="middle" font-weight="500">75.0</text>
  <rect x="90" y="564" width="500" height="22" class="tbl-alt"/>
  <text class="ts" x="104" y="579" fill="#0F6E56" font-weight="500">Q4_K_M</text>
  <text class="ts" x="245" y="579" text-anchor="middle">16.8 GB</text>
  <text class="ts" x="370" y="579" text-anchor="middle">8–12</text>
  <text class="ts" x="485" y="579" text-anchor="middle" font-weight="500">73.1</text>
  <rect x="90" y="586" width="500" height="22" class="tbl-row"/>
  <text class="ts" x="104" y="601" fill="#185FA5" font-weight="500">IQ3_M</text>
  <text class="ts" x="245" y="601" text-anchor="middle">12.0 GB</text>
  <text class="ts" x="370" y="601" text-anchor="middle">12–15</text>
  <text class="ts" x="485" y="601" text-anchor="middle" font-weight="500">70.5</text>
  <text class="tss" x="90" y="622" opacity=".5">MMLU Pro = Massive Multitask Language Understanding (Professional). Higher is better.</text>
</svg>
```

**Chart CSS classes (add to hosting page):**

```css
/* Grid and axes */
.grid { stroke: var(--border); stroke-width: 0.5; stroke-dasharray: 4 3; fill: none; }
.axis { stroke: var(--text-secondary); stroke-width: 1; fill: none; }
.axis-tick { stroke: var(--text-secondary); stroke-width: 0.75; }

/* Bar fills per category — light mode */
.bar-fp16-min { fill: #FCEBEB; stroke: #A32D2D; stroke-width: 0.75; }
.bar-fp16-max { fill: #F7C1C1; stroke: #A32D2D; stroke-width: 0.75; }
.bar-q8-min   { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.75; }
.bar-q8-max   { fill: #FAC775; stroke: #854F0B; stroke-width: 0.75; }
.bar-q4-min   { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.75; }
.bar-q4-max   { fill: #9FE1CB; stroke: #0F6E56; stroke-width: 0.75; }
.bar-iq3-min  { fill: #E6F1FB; stroke: #185FA5; stroke-width: 0.75; }
.bar-iq3-max  { fill: #B5D4F4; stroke: #185FA5; stroke-width: 0.75; }

/* Dark mode bars */
@media (prefers-color-scheme: dark) {
  .bar-fp16-min { fill: #501313; stroke: #F09595; }
  .bar-fp16-max { fill: #791F1F; stroke: #F09595; }
  .bar-q8-min   { fill: #412402; stroke: #EF9F27; }
  .bar-q8-max   { fill: #633806; stroke: #EF9F27; }
  .bar-q4-min   { fill: #04342C; stroke: #5DCAA5; }
  .bar-q4-max   { fill: #085041; stroke: #5DCAA5; }
  .bar-iq3-min  { fill: #042C53; stroke: #85B7EB; }
  .bar-iq3-max  { fill: #0C447C; stroke: #85B7EB; }
}

/* VRAM overlay */
.vram-line { stroke: #534AB7; stroke-width: 2.5; fill: none; }
.vram-dot  { fill: #534AB7; stroke: var(--bg-primary); stroke-width: 2; }
.vram-label { font-family: system-ui, sans-serif; font-size: 10px; fill: #534AB7; font-weight: 500; }

/* Threshold line */
.threshold { stroke: #A32D2D; stroke-width: 1; stroke-dasharray: 6 3; fill: none; }
.threshold-label { font-family: system-ui, sans-serif; font-size: 10px; fill: #A32D2D; font-weight: 500; }

/* Inset table */
.tbl-header { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.5; }
.tbl-row    { fill: transparent; stroke: var(--border); stroke-width: 0.25; }
.tbl-alt    { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.25; }
```

**Chart layout formula:**

- **Chart area**: x=90–590, y=70–410 (500px wide, 340px tall)
- **Left Y-axis**: Primary metric (tok/s). Scale: `y = 410 − (val/max_val) × 340`
- **Right Y-axis**: Secondary metric (VRAM GB). Same formula, different scale labels
- **Groups**: Divide width by number of categories. Each group gets min bar + gap + max bar
- **Bars**: `rx="3"` for rounded tops, width ~34px each
- **VRAM line**: `<polyline>` connecting data points across group centers
- **Threshold**: Horizontal dashed line at the critical value (e.g., 24 GB GPU limit)
- **Inset table**: Below the chart, alternating row fills (`.tbl-alt` / `.tbl-row`)
- **Semantic bar colors**: Each category gets its own color pair (lighter=min, darker=max) using the skill color palette

---

## Hosting Diagrams

After generating an SVG diagram, host it as an interactive web page. Each diagram gets its own directory for easy organization and browsing.

### Directory Structure

Use this structure to support multiple diagrams:

```
.diagrams/
├── hospital-emergency-flow/
│   └── index.html
├── smart-city-infrastructure/
│   └── index.html
├── electricity-grid-flow/
│   └── index.html
├── banana-journey/
│   └── index.html
└── smartphone-layers/
    └── index.html
```

### Step 1: Create the diagram directory

Create a directory for each diagram using kebab-case naming:

```bash
mkdir -p .diagrams/<diagram-name>
```

**Naming conventions:**
- Use lowercase kebab-case: `smart-city-infrastructure`, `electricity-grid-flow`
- Be descriptive but concise: `user-auth-flow` not `diagram-1`
- Include the diagram type if helpful: `banana-journey`, `smartphone-layers`

### Step 2: Create the HTML page

Create `.diagrams/<diagram-name>/index.html` with the following template. This page includes the full CSS design system that makes the `c-*` classes, text classes, and dark mode work.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Architecture Diagram</title>
<style>
  :root {
    --text-primary: #1a1a18;
    --text-secondary: #5f5e5a;
    --text-tertiary: #88877f;
    --bg-primary: #ffffff;
    --bg-secondary: #f6f5f0;
    --bg-tertiary: #eeedeb;
    --border: rgba(0,0,0,0.15);
    --border-hover: rgba(0,0,0,0.3);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --text-primary: #e8e6de;
      --text-secondary: #b4b2a9;
      --text-tertiary: #888780;
      --bg-primary: #1a1a18;
      --bg-secondary: #2c2c2a;
      --bg-tertiary: #3d3d3a;
      --border: rgba(255,255,255,0.15);
      --border-hover: rgba(255,255,255,0.3);
    }
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg-tertiary);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    min-height: 100vh;
    padding: 40px 20px;
  }
  .card {
    background: var(--bg-primary);
    border-radius: 16px;
    padding: 32px;
    max-width: 780px;
    width: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  h1 {
    font-size: 18px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 8px;
  }
  .subtitle {
    font-size: 13px;
    color: var(--text-tertiary);
    margin-bottom: 24px;
  }
  svg { width: 100%; height: auto; }

  /* === SVG Design System Classes === */

  /* Text classes */
  .t  { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; fill: var(--text-primary); }
  .ts { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; fill: var(--text-secondary); }
  .th { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; fill: var(--text-primary); font-weight: 500; }

  /* Neutral box */
  .box { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.5px; }

  /* Arrow */
  .arr { stroke: var(--text-secondary); stroke-width: 1.5px; fill: none; }

  /* Leader line */
  .leader { stroke: var(--text-tertiary); stroke-width: 0.5px; stroke-dasharray: 4 3; fill: none; }

  /* Clickable node */
  .node { cursor: pointer; transition: opacity 0.15s; }
  .node:hover { opacity: 0.82; }

  /* === Color Ramp Classes (light mode) === */
  .c-purple > rect, .c-purple > circle, .c-purple > ellipse { fill: #EEEDFE; stroke: #534AB7; }
  .c-purple > .th, .c-purple > text.th { fill: #3C3489; }
  .c-purple > .ts, .c-purple > text.ts { fill: #534AB7; }
  .c-purple > .t,  .c-purple > text.t  { fill: #3C3489; }

  .c-teal > rect, .c-teal > circle, .c-teal > ellipse { fill: #E1F5EE; stroke: #0F6E56; }
  .c-teal > .th, .c-teal > text.th { fill: #085041; }
  .c-teal > .ts, .c-teal > text.ts { fill: #0F6E56; }
  .c-teal > .t,  .c-teal > text.t  { fill: #085041; }

  .c-coral > rect, .c-coral > circle, .c-coral > ellipse { fill: #FAECE7; stroke: #993C1D; }
  .c-coral > .th, .c-coral > text.th { fill: #712B13; }
  .c-coral > .ts, .c-coral > text.ts { fill: #993C1D; }
  .c-coral > .t,  .c-coral > text.t  { fill: #712B13; }

  .c-pink > rect, .c-pink > circle, .c-pink > ellipse { fill: #FBEAF0; stroke: #993556; }
  .c-pink > .th, .c-pink > text.th { fill: #72243E; }
  .c-pink > .ts, .c-pink > text.ts { fill: #993556; }
  .c-pink > .t,  .c-pink > text.t  { fill: #72243E; }

  .c-gray > rect, .c-gray > circle, .c-gray > ellipse { fill: #F1EFE8; stroke: #5F5E5A; }
  .c-gray > .th, .c-gray > text.th { fill: #444441; }
  .c-gray > .ts, .c-gray > text.ts { fill: #5F5E5A; }
  .c-gray > .t,  .c-gray > text.t  { fill: #444441; }

  .c-blue > rect, .c-blue > circle, .c-blue > ellipse { fill: #E6F1FB; stroke: #185FA5; }
  .c-blue > .th, .c-blue > text.th { fill: #0C447C; }
  .c-blue > .ts, .c-blue > text.ts { fill: #185FA5; }
  .c-blue > .t,  .c-blue > text.t  { fill: #0C447C; }

  .c-green > rect, .c-green > circle, .c-green > ellipse { fill: #EAF3DE; stroke: #3B6D11; }
  .c-green > .th, .c-green > text.th { fill: #27500A; }
  .c-green > .ts, .c-green > text.ts { fill: #3B6D11; }
  .c-green > .t,  .c-green > text.t  { fill: #27500A; }

  .c-amber > rect, .c-amber > circle, .c-amber > ellipse { fill: #FAEEDA; stroke: #854F0B; }
  .c-amber > .th, .c-amber > text.th { fill: #633806; }
  .c-amber > .ts, .c-amber > text.ts { fill: #854F0B; }
  .c-amber > .t,  .c-amber > text.t  { fill: #633806; }

  .c-red > rect, .c-red > circle, .c-red > ellipse { fill: #FCEBEB; stroke: #A32D2D; }
  .c-red > .th, .c-red > text.th { fill: #791F1F; }
  .c-red > .ts, .c-red > text.ts { fill: #A32D2D; }
  .c-red > .t,  .c-red > text.t  { fill: #791F1F; }

  /* === Dark mode overrides === */
  @media (prefers-color-scheme: dark) {
    .c-purple > rect, .c-purple > circle, .c-purple > ellipse { fill: #3C3489; stroke: #AFA9EC; }
    .c-purple > .th, .c-purple > text.th { fill: #CECBF6; }
    .c-purple > .ts, .c-purple > text.ts { fill: #AFA9EC; }

    .c-teal > rect, .c-teal > circle, .c-teal > ellipse { fill: #085041; stroke: #5DCAA5; }
    .c-teal > .th, .c-teal > text.th { fill: #9FE1CB; }
    .c-teal > .ts, .c-teal > text.ts { fill: #5DCAA5; }

    .c-coral > rect, .c-coral > circle, .c-coral > ellipse { fill: #712B13; stroke: #F0997B; }
    .c-coral > .th, .c-coral > text.th { fill: #F5C4B3; }
    .c-coral > .ts, .c-coral > text.ts { fill: #F0997B; }

    .c-pink > rect, .c-pink > circle, .c-pink > ellipse { fill: #72243E; stroke: #ED93B1; }
    .c-pink > .th, .c-pink > text.th { fill: #F4C0D1; }
    .c-pink > .ts, .c-pink > text.ts { fill: #ED93B1; }

    .c-gray > rect, .c-gray > circle, .c-gray > ellipse { fill: #444441; stroke: #B4B2A9; }
    .c-gray > .th, .c-gray > text.th { fill: #D3D1C7; }
    .c-gray > .ts, .c-gray > text.ts { fill: #B4B2A9; }

    .c-blue > rect, .c-blue > circle, .c-blue > ellipse { fill: #0C447C; stroke: #85B7EB; }
    .c-blue > .th, .c-blue > text.th { fill: #B5D4F4; }
    .c-blue > .ts, .c-blue > text.ts { fill: #85B7EB; }

    .c-green > rect, .c-green > circle, .c-green > ellipse { fill: #27500A; stroke: #97C459; }
    .c-green > .th, .c-green > text.th { fill: #C0DD97; }
    .c-green > .ts, .c-green > text.ts { fill: #97C459; }

    .c-amber > rect, .c-amber > circle, .c-amber > ellipse { fill: #633806; stroke: #EF9F27; }
    .c-amber > .th, .c-amber > text.th { fill: #FAC775; }
    .c-amber > .ts, .c-amber > text.ts { fill: #EF9F27; }

    .c-red > rect, .c-red > circle, .c-red > ellipse { fill: #791F1F; stroke: #F09595; }
    .c-red > .th, .c-red > text.th { fill: #F7C1C1; }
    .c-red > .ts, .c-red > text.ts { fill: #F09595; }
  }
</style>
</head>
<body>
<div class="card">
  <h1><!-- DIAGRAM TITLE HERE --></h1>
  <p class="subtitle"><!-- OPTIONAL SUBTITLE HERE --></p>
  <!-- PASTE SVG HERE -->
</div>
</body>
</html>
```

### Step 3: Start the HTTP server

Start the server at the `.diagrams/` root directory. This serves all diagram directories and provides a browsable listing:

```bash
cd .diagrams && python3 -m http.server 22223 --bind 0.0.0.0 &
```

**Important:** Start the server at `.diagrams/`, NOT inside a specific diagram folder. This way:
- `http://0.0.0.0:22223/` shows a directory listing of all diagrams
- `http://0.0.0.0:22223/smart-city-infrastructure/` opens that specific diagram
- Users can browse and switch between diagrams easily

### Step 4: Confirm

Print a message to the user:

```
Diagram hosted at http://0.0.0.0:22223/<diagram-name>/
Browse all diagrams at http://0.0.0.0:22223/
```

### Complete Workflow

Here is the complete workflow for creating a new diagram:

```bash
# 1. Create diagram directory (use descriptive kebab-case name)
mkdir -p .diagrams/smart-city-infrastructure

# 2. Create the HTML file with embedded SVG
cat > .diagrams/smart-city-infrastructure/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Smart City Infrastructure</title>
  <!-- ... full HTML template with styles ... -->
</head>
<body>
  <div class="card">
    <h1>Smart City Infrastructure</h1>
    <p class="subtitle">Description here</p>
    <svg><!-- SVG content --></svg>
  </div>
</body>
</html>
EOF

# 3. Start server (only needed once, serves all diagrams)
cd .diagrams && python3 -m http.server 22223 --bind 0.0.0.0 &

# 4. Confirm
echo "Diagram: http://0.0.0.0:22223/smart-city-infrastructure/"
echo "All diagrams: http://0.0.0.0:22223/"
```

### Adding More Diagrams

To add another diagram, simply create a new directory:

```bash
mkdir -p .diagrams/electricity-grid-flow
# Create .diagrams/electricity-grid-flow/index.html
```

The server (if already running) will automatically serve the new diagram. No restart needed.

### Browsing Diagrams

When you navigate to `http://0.0.0.0:22223/`, Python's HTTP server shows a directory listing:

```
Directory listing for /

• banana-journey/
• electricity-grid-flow/
• hospital-emergency-flow/
• smart-city-infrastructure/
• smartphone-layers/
```

Click any folder to view that diagram.

---

## Quick Reference: What to Use When

| User says | Diagram type | Color scheme |
|-----------|-------------|--------------|
| "show the architecture" | Structural (containment) | purple container, teal services, coral data |
| "draw the pipeline" | Flowchart (left-right or top-down) | gray start/end, purple steps, red errors, teal deploy |
| "map the endpoints" | API endpoint map (tree) | purple root, one ramp per resource group |
| "show the services" | Microservice topology | gray ingress, teal services, purple bus, coral workers |
| "visualize the data flow" | Data pipeline (left-right) | gray sources, purple processing, teal sinks |
| "draw the infrastructure" | Structural (VPC/subnet nesting) | purple VPC, teal public, coral private |
| "show the ML pipeline" | Flowchart + structural hybrid | gray input, purple preprocessing, teal model server, coral output |
| "draw the aircraft/vehicle" | Physical / Structural | Use paths, polygons, ellipses for realistic shapes |
| "smart city/IoT system" | Infrastructure / Systems Integration | Hub-spoke layout, semantic line styles per system |
| "show the dashboard" | UI / Dashboard Mockup | Dark screen, chart colors: teal, purple, coral for alerts |
| "power grid/electricity flow" | Multi-stage Flow | Left-to-right stages, voltage hierarchy (HV/MV/LV line weights) |
| "wind turbine" / "turbine structure" | Physical Cross-section | Underground foundation, tower cutaway, nacelle components color-coded |
| "journey of X" / "lifecycle" | Narrative Journey | Winding path, progressive state changes, timeline |
| "layers of X" / "exploded view" | Exploded Layer View | Vertical stack, alternating labels, component detail |
| "CPU microarchitecture" / "pipeline" | Hardware Pipeline | Vertical stages, fan-out to execution ports, sidebar for memory |
| "floor plan" / "apartment layout" | Architectural Floor Plan | Walls, doors, windows, room fills, proposed changes in dotted red |
| "reaction mechanism" / "chemistry" | Chemistry Mechanism | Atoms, bonds, curved arrows, transition state brackets, energy profile |

---

## Examples Reference

See the `examples/` directory for complete, tested diagram templates:

### hospital-emergency-department-flow.md
Multi-path flowchart with priority-based routing using semantic colors (red=critical, amber=urgent, green=stable). Demonstrates convergent/divergent flows, stage labels, and color-coded legends.

### feature-film-production-pipeline.md
Phased workflow with neutral dashed containers and colored inner nodes. Shows horizontal sub-flows within a phase (post-production pipeline) and the pattern of containers receding while content pops.

### commercial-aircraft-structure.md
Physical/structural diagram using paths, polygons, and ellipses to draw realistic shapes. Demonstrates going beyond rectangles for physical objects, layered composition, and leader lines with labels.

### smart-city-infrastructure.md
Multi-system integration diagram with hub-spoke layout. Central IoT platform connects power grid (solar, wind, battery), water system (reservoir, treatment, pipes), and transport (roads, signals, buses). Includes dashboard UI mockup and demonstrates semantic line styles for different infrastructure types.

### electricity-grid-flow.md
Left-to-right multi-stage flow showing electricity from generation (nuclear, solar, wind, gas) through transmission (pylons, substations) to distribution (pole transformers) and consumption (homes, factories, EV chargers). Demonstrates voltage level visual hierarchy, smart grid data overlay, and flow arrow markers.

### wind-turbine-structure.md
Physical cross-section of a modern onshore wind turbine from underground foundation to blade tips. Shows deep concrete foundation with rebar grid, tapered tubular steel tower with internal ladder and elevator, nacelle cutaway with gearbox (gold), generator (blue), brake (red), and yaw system (green), rotor hub with pitch motors (purple) at blade roots, and power cables running to transformer at base. Demonstrates underground/above-ground separation, component color coding with legend, cross-section layering, and mechanical system visualization.

### banana-journey-tree-to-smoothie.md
Narrative journey diagram following a banana from Costa Rica harvest through shipping, inspection, ripening, retail, and finally the consumer's kitchen (overripe → frozen → smoothie). Demonstrates storytelling visualization, winding path layout, progressive state changes (green → yellow → brown), and fun narrative details like spider inspection and price tags.

### smartphone-layer-anatomy.md
Exploded view showing 7 internal layers of a smartphone from front glass to back. Features alternating left/right labels to prevent overlap, detailed component rendering (PCB with chips, camera lenses, wireless coil), and thickness scale. Demonstrates product teardown visualization.

### cpu-ooo-microarchitecture.md
Out-of-order CPU core microarchitecture showing full superscalar pipeline: Fetch → Decode (with µop cache bypass) → Rename/Allocate (RAT, Physical RF) → Unified Scheduler → 6 Execution Ports (ALU, Branch, Load, Store, Vector) → Reorder Buffer. Memory hierarchy sidebar shows L1-I, L1-D, L2, and DRAM. Demonstrates fan-out patterns, path merging, container grouping, and hardware pipeline visualization.

### apartment-floor-plan-conversion.md
Architectural floor plan showing 3 BHK to 4 BHK apartment conversion. Features room color coding, door swings, window symbols, proposed walls in dotted red, circulation arrows for new access paths, and area comparison table. Demonstrates architectural drawing conventions, overlay technique for renovation proposals, and quantitative data integration with floor plans.

### sn2-reaction-mechanism.md
Organic chemistry SN2 reaction mechanism (OH⁻ + CH₃Br → CH₃OH + Br⁻). Shows ball-and-stick molecular structures, curved electron-pushing arrows, pentacoordinate transition state with partial charges (δ⁺/δ⁻), wedge/dash stereochemistry, and Walden inversion. Includes reaction energy profile with activation energy and exergonic product formation. Demonstrates chemistry notation, molecular rendering, and energy diagrams.

### autonomous-llm-research-agent-flow.md
Multi-section flowchart of Karpathy's autoresearch framework: human-agent handoff, autonomous experiment loop with keep/discard decision branching, and modifiable training pipeline. Demonstrates loop-back arrows with rounded corners for infinite repetition, convergent decision paths (green=keep, red=discard), semantic color coding for outcomes, highlighted key steps using contrasting color (coral "Run training" among teal action steps), neutral dashed containers, horizontal sub-flows within a detail section, and footer metadata for fixed constraints.

### ml-benchmark-grouped-bar-chart.md
Grouped bar chart comparing LLM inference speed across quantization levels (FP16, Q8_0, Q4_K_M, IQ3_M) with min/max range bars per category. Features dual Y-axis (tok/s left, VRAM GB right), VRAM usage overlay as a polyline with labeled dots, dashed red threshold line for GPU VRAM limit, semantic color coding per quantization level, zone annotations (exceeds/fits VRAM), inset data table with MMLU Pro accuracy scores and delta from baseline, and full dark mode support. Demonstrates quantitative data visualization, grouped bar layout, secondary axis line graph, threshold markers, and tabular data integration.