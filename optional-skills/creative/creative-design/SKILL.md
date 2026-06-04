---
name: creative-design
description: Unified creative & design skill — design specs, diagrams, mockups, ideation, and real-world design system references.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [design, creative, diagrams, mockups, ideation, design-systems, UI, architecture, excalidraw]
    absorbed_skills:
      - design-md
      - excalidraw
      - sketch
      - architecture-diagram
      - creative-ideation
      - popular-web-designs
---

# Creative Design

Unified umbrella skill for visual design, diagram generation, mockup exploration, creative ideation, and real-world design system references. Absorbs content from 6 former skills — use this as the single entry point for any design-related task.

---

## Skill Map

This umbrella covers 6 design modes. Pick the right mode based on the ask:

| Mode | Skill | When to use |
|------|-------|-------------|
| **Design Specs** | `design-md` section below | Author/validate/export Google's DESIGN.md token spec files |
| **Hand-drawn Diagrams** | `excalidraw` section below | Excalidraw JSON diagrams (arch, flow, seq) |
| **HTML Mockups** | `sketch` section below | Throwaway HTML mockups: 2-3 design variants to compare |
| **SVG Architecture Diagrams** | `architecture-diagram` section below | Dark-themed SVG architecture/cloud/infra diagrams as HTML |
| **Project Ideas** | `ideation` section below | Generate project ideas via creative constraints |
| **Design System References** | `popular-web-designs` section below | 54 real design systems (Stripe, Linear, Vercel) as HTML/CSS |

---

## design-md — Author/Validate/Export DESIGN.md Token Specs

DESIGN.md is Google's open spec (Apache-2.0, `google-labs-code/design.md`) for
describing a visual identity to coding agents. One file combines:

- **YAML front matter** — machine-readable design tokens (normative values)
- **Markdown body** — human-readable rationale, organized into canonical sections

Tokens give exact values. Prose tells agents *why* those values exist and how to
apply them. The CLI (`npx @google/design.md`) lints structure + WCAG contrast,
diffs versions for regressions, and exports to Tailwind or W3C DTCG JSON.

### When to use design-md

- User asks for a DESIGN.md file, design tokens, or a design system spec
- User wants consistent UI/brand across multiple projects or tools
- User pastes an existing DESIGN.md and asks to lint, diff, export, or extend it
- User asks to port a style guide into a format agents can consume
- User wants contrast / WCAG accessibility validation on their color palette

For purely visual inspiration or layout examples, use **popular-web-designs** below.
For *process and taste* when designing a one-off HTML artifact from scratch
(prototype, deck, landing page, component lab), use `claude-design`. This section
is for the *formal spec file* itself.

### File anatomy

```md
---
version: alpha
name: Heritage
description: Architectural minimalism meets journalistic gravitas.
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:
    fontFamily: Public Sans
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body-md:
    fontFamily: Public Sans
    fontSize: 1rem
rounded:
  sm: 4px
  md: 8px
  lg: 16px
spacing:
  sm: 8px
  md: 16px
  lg: 24px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary}"
---

## Overview

Architectural Minimalism meets Journalistic Gravitas...
```

### Token types

| Type | Format | Example |
|------|--------|---------|
| Color | `#` + hex (sRGB) | `"#1A1C1E"` |
| Dimension | number + unit (`px`, `em`, `rem`) | `48px`, `-0.02em` |
| Token reference | `{path.to.token}` | `{colors.primary}` |
| Typography | object with `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation` | see above |

Component property whitelist: `backgroundColor`, `textColor`, `typography`,
`rounded`, `padding`, `size`, `height`, `width`. Variants (hover, active,
pressed) are **separate component entries** with related key names
(`button-primary-hover`), not nested.

### Canonical section order

Sections are optional, but present ones MUST appear in this order. Duplicate
headings reject the file.

1. Overview (alias: Brand & Style)
2. Colors
3. Typography
4. Layout (alias: Layout & Spacing)
5. Elevation & Depth (alias: Elevation)
6. Shapes
7. Components
8. Do's and Don'ts

### Workflow: authoring a new DESIGN.md

1. **Ask the user** (or infer) the brand tone, accent color, and typography
   direction. If they provided a site, image, or vibe, translate it to the
   token shape above.
2. **Write `DESIGN.md`** in their project root using `write_file`. Always
   include `name:` and `colors:`; other sections optional but encouraged.
3. **Use token references** (`{colors.primary}`) in the `components:` section
   instead of re-typing hex values. Keeps the palette single-source.
4. **Lint it** (see below). Fix any broken references or WCAG failures
   before returning.
5. **If the user has an existing project**, also write Tailwind or DTCG
   exports next to the file (`tailwind.theme.json`, `tokens.json`).

### Workflow: lint / diff / export

```bash
# Validate structure + token references + WCAG contrast
npx -y @google/design.md lint DESIGN.md

# Compare two versions, fail on regression (exit 1 = regression)
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md

# Export to Tailwind theme JSON
npx -y @google/design.md export --format tailwind DESIGN.md > tailwind.theme.json

# Export to W3C DTCG (Design Tokens Format Module) JSON
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json
```

### Lint rule reference

- `broken-ref` (error) — `{colors.missing}` points at a non-existent token
- `duplicate-section` (error) — same `## Heading` appears twice
- `invalid-color`, `invalid-dimension`, `invalid-typography` (error)
- `wcag-contrast` (warning/info) — component `textColor` vs `backgroundColor`
  ratio against WCAG AA (4.5:1) and AAA (7:1)
- `unknown-component-property` (warning) — outside the property whitelist

### design-md Pitfalls

- **Don't nest component variants.** `button-primary.hover` is wrong;
  `button-primary-hover` as a sibling key is right.
- **Hex colors must be quoted strings.** YAML will otherwise choke on `#`.
- **Negative dimensions need quotes too.** `letterSpacing: "-0.02em"`.
- **Section order is enforced.** Reorder to match the canonical list before saving.
- **`version: alpha` is the current spec version.** Watch for breaking changes.
- **Token references resolve by dotted path.** `{colors.primary}` works; `{primary}` does not.

### design-md linked files

- `templates/starter.md` — full starter template with all sections

### design-md source

- Repo: https://github.com/google-labs-code/design.md (Apache-2.0)
- CLI: `@google/design.md` on npm

---

## excalidraw — Hand-drawn Excalidraw JSON Diagrams

Create diagrams by writing standard Excalidraw element JSON and saving as `.excalidraw` files. These files can be drag-and-dropped onto [excalidraw.com](https://excalidraw.com) for viewing and editing. No accounts, no API keys, no rendering libraries — just JSON.

### When to use excalidraw

Generate `.excalidraw` files for architecture diagrams, flowcharts, sequence diagrams, concept maps, and more. Files can be opened at excalidraw.com or uploaded for shareable links. For dark-themed tech-aesthetic SVG diagrams instead, use **architecture-diagram** below.

### Workflow

1. **Write the elements JSON** — an array of Excalidraw element objects
2. **Save the file** using `write_file` to create a `.excalidraw` file
3. **Optionally upload** for a shareable link via `scripts/upload.py`

### Saving a Diagram

Wrap your elements array in the standard `.excalidraw` envelope:

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "hermes-agent",
  "elements": [ ...your elements array here... ],
  "appState": { "viewBackgroundColor": "#ffffff" }
}
```

Save to any path, e.g. `~/diagrams/my_diagram.excalidraw`.

### Uploading for a Shareable Link

```bash
python skills/creative/creative-design/references/excalidraw-upload.py ~/diagrams/my_diagram.excalidraw
```

Requires `pip install cryptography`.

### Element Format Reference

**Required fields (all elements):** `type`, `id` (unique string), `x`, `y`, `width`, `height`

**Defaults (skip these):** `strokeColor: "#1e1e1e"`, `backgroundColor: "transparent"`, `fillStyle: "solid"`, `strokeWidth: 2`, `roughness: 1`, `opacity: 100`

**Element Types:**

```json
// Rectangle
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 100 }

// Ellipse
{ "type": "ellipse", "id": "e1", "x": 100, "y": 100, "width": 150, "height": 150 }

// Diamond
{ "type": "diamond", "id": "d1", "x": 100, "y": 100, "width": 150, "height": 150 }

// Labeled shape (container binding — DO NOT use "label" property)
{ "type": "rectangle", "id": "r1", "x": 100, "y": 100, "width": 200, "height": 80,
  "roundness": { "type": 3 }, "backgroundColor": "#a5d8ff", "fillStyle": "solid",
  "boundElements": [{ "id": "t_r1", "type": "text" }] },
{ "type": "text", "id": "t_r1", "x": 105, "y": 110, "width": 190, "height": 25,
  "text": "Hello", "fontSize": 20, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "r1", "originalText": "Hello", "autoResize": true }

// Arrow
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow" }

// Arrow with label
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 200, "height": 0,
  "points": [[0,0],[200,0]], "endArrowhead": "arrow",
  "boundElements": [{ "id": "t_a1", "type": "text" }] },
{ "type": "text", "id": "t_a1", "x": 370, "y": 130, "width": 60, "height": 20,
  "text": "connects", "fontSize": 16, "fontFamily": 1, "strokeColor": "#1e1e1e",
  "textAlign": "center", "verticalAlign": "middle",
  "containerId": "a1", "originalText": "connects", "autoResize": true }

// Standalone text (titles and annotations only — no container)
{ "type": "text", "id": "t1", "x": 150, "y": 138, "text": "Hello", "fontSize": 20,
  "fontFamily": 1, "strokeColor": "#1e1e1e", "originalText": "Hello", "autoResize": true }
```

**Arrow bindings:**
```json
{ "type": "arrow", "id": "a1", "x": 300, "y": 150, "width": 150, "height": 0,
  "points": [[0,0],[150,0]], "endArrowhead": "arrow",
  "startBinding": { "elementId": "r1", "fixedPoint": [1, 0.5] },
  "endBinding": { "elementId": "r2", "fixedPoint": [0, 0.5] } }
```
`fixedPoint` coordinates: `top=[0.5,0]`, `bottom=[0.5,1]`, `left=[0,0.5]`, `right=[1,0.5]`

**Arrow properties:** `endArrowhead: null | "arrow" | "bar" | "dot" | "triangle"`, `strokeStyle: "solid" | "dashed" | "dotted"`

### Drawing Order

- Array order = z-order (first = back, last = front)
- Emit progressively: background zones → shape → its bound text → its arrows → next shape
- **BAD:** all rectangles, then all texts, then all arrows
- **GOOD:** bg_zone → shape1 → text_for_shape1 → arrow1 → shape2 → text_for_shape2 → ...

### Sizing Guidelines

**Font sizes:** minimum `fontSize: 16` for body labels, `20` for titles, `14` for secondary annotations (sparingly). NEVER below 14.

**Element sizes:** minimum 120×60 for labeled shapes. Leave 20–30px gaps between elements.

### Color Palette

| Use | Fill Color | Hex |
|-----|-----------|-----|
| Primary / Input | Light Blue | `#a5d8ff` |
| Success / Output | Light Green | `#b2f2bb` |
| Warning / External | Light Orange | `#ffd8a8` |
| Processing / Special | Light Purple | `#d0bfff` |
| Error / Critical | Light Red | `#ffc9c9` |
| Notes / Decisions | Light Yellow | `#fff3bf` |
| Storage / Data | Light Teal | `#c3fae8` |

Full palette reference: `references/excalidraw-colors.md`

### excalidraw linked files

- `references/excalidraw-examples.md` — complete copy-pasteable examples
- `references/excalidraw-dark-mode.md` — dark mode color palettes and patterns
- `references/excalidraw-colors.md` — full color palette reference
- `references/excalidraw-upload.py` — upload script (cryptography required)

---

## sketch — Throwaway HTML Mockups (2-3 Variants)

Use when the user wants to **see a design direction before committing** — exploring a UI/UX idea as disposable HTML mockups. Generate 2-3 interactive variants so the user can compare visual directions side-by-side, not to produce shippable code.

### When to use sketch

Load this when the user says things like "sketch this screen", "show me what X could look like", "compare layout A vs B", "give me 2-3 takes on this UI", "let me see some variants", "mockup this before I build".

### When NOT to use sketch

- User wants a production component — use `claude-design` or build it properly
- User wants a polished one-off HTML artifact (landing page, deck) — `claude-design`
- User wants a diagram — use **excalidraw** or **architecture-diagram**
- The design is already locked — just build it

### Core method

```
intake  →  variants  →  head-to-head  →  pick winner (or iterate)
```

### 1. Intake (skip if user already gave enough)

Get three things — one at a time, not all at once:
1. **Feel.** "What should this feel like? Adjectives, emotions, a vibe."
2. **References.** "What apps, sites, or products capture the feel you're imagining?"
3. **Core action.** "What's the single most important thing a user does on this screen?"

### 2. Variants (2-3, never 1, rarely 4+)

Each variant is a **complete, standalone HTML file** with:
- Inline `<style>` — no build step, no external CSS (Tailwind via CDN OK)
- System fonts or one Google Font via `<link>`
- Realistic fake content — actual sentences, not "Lorem ipsum"
- **Interactive**: links clickable, hovers real, at least one state transition

**Variant naming:** describe the stance, not the number.

```
sketches/
├── 001-calm-editorial/
│   ├── index.html
│   └── README.md
└── 001-utilitarian-dense/
    ├── index.html
    └── README.md
```

**Verify variants visually** — use browser navigation and vision tools. Don't just write HTML and hope it renders.

**Default CSS reset:**
```html
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    color: #1a1a1a;
    background: #fafafa;
    line-height: 1.5;
  }
</style>
```

### 3. Head-to-head

Present as a comparison table:

```markdown
| Dimension | Calm editorial | Utilitarian dense |
|-----------|----------------|-------------------|
| Density   | Low            | High              |
| Primary action visibility | Low | High |
| Feel | Calm, trusted | Sharp, tool-like |

**My take:** Utilitarian dense for power users, calm editorial for content-forward.
```

Let the user pick a winner, or combine two into a hybrid, or ask for another round.

### Interactivity bar

A sketch is interactive enough when the user can:
1. **Click a primary action** and something visible happens (state change, modal, toast)
2. **See one meaningful state transition** (filter a list, toggle a mode, open/close a panel)
3. **Hover recognizable affordances** (buttons, rows, tabs)

### Output

- Create `sketches/` in the repo root
- One subdir per variant: `NNN-stance-name/index.html` + `README.md`
- Tell the user how to open them

### Attribution

Adapted from the GSD (Get Shit Done) project's `/gsd-sketch` workflow — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)).

---

## architecture-diagram — Dark-themed SVG Architecture Diagrams as HTML

Generate professional, dark-themed technical architecture diagrams as standalone HTML files with inline SVG graphics. No external tools, no API keys, no rendering libraries — just write the HTML file and open it in a browser.

### When to use architecture-diagram

**Best suited for:**
- Software system architecture (frontend / backend / database layers)
- Cloud infrastructure (VPC, regions, subnets, managed services)
- Microservice / service-mesh topology
- Database + API map, deployment diagrams
- Anything with a tech-infra subject that fits a dark, grid-backed aesthetic

**Look elsewhere first for:**
- Hand-drawn whiteboard sketches (use **excalidraw**)
- Physics, chemistry, math, biology, or other scientific subjects
- Physical objects, floor plans, narrative journeys (use **excalidraw**)

### Workflow

1. User describes their system architecture (components, connections, technologies)
2. Generate the HTML file following the design system below
3. Save with `write_file` to a `.html` file (e.g. `./[project-name]-architecture.html`)
4. User opens in any browser — works offline, no dependencies

### Color Palette (Semantic Mapping)

| Component Type | Fill (rgba) | Stroke (Hex) |
| :--- | :--- | :--- |
| **Frontend** | `rgba(8, 51, 68, 0.4)` | `#22d3ee` (cyan-400) |
| **Backend** | `rgba(6, 78, 59, 0.4)` | `#34d399` (emerald-400) |
| **Database** | `rgba(76, 29, 149, 0.4)` | `#a78bfa` (violet-400) |
| **AWS/Cloud** | `rgba(120, 53, 15, 0.3)` | `#fbbf24` (amber-400) |
| **Security** | `rgba(136, 19, 55, 0.4)` | `#fb7185` (rose-400) |
| **Message Bus** | `rgba(251, 146, 60, 0.3)` | `#fb923c` (orange-400) |
| **External** | `rgba(30, 41, 59, 0.5)` | `#94a3b8` (slate-400) |

### Typography & Background

- **Font:** JetBrains Mono (Monospace), loaded from Google Fonts
- **Sizes:** 12px (Names), 9px (Sublabels), 8px (Annotations), 7px (Tiny labels)
- **Background:** Slate-950 (`#020617`) with a subtle 40px grid pattern

### Technical Implementation

**Component Rendering:** Rounded rectangles (`rx="6"`) with 1.5px strokes. To prevent arrows from showing through semi-transparent fills, use a **double-rect masking technique**:
1. Draw an opaque background rect (`#0f172a`)
2. Draw the semi-transparent styled rect on top

**Connection Rules:**
- **Z-Order:** Draw arrows *early* in the SVG (after the grid) so they render behind component boxes
- **Arrowheads:** Defined via SVG markers
- **Security Flows:** Use dashed lines in rose color (`#fb7185`)
- **Boundaries:**
  - *Security Groups:* Dashed (`4,4`), rose color
  - *Regions:* Large dashed (`8,4`), amber color, `rx="12"`

**Spacing & Layout:**
- **Standard Height:** 60px (Services); 80–120px (Large components)
- **Vertical Gap:** Minimum 40px between components
- **Message Buses:** Must be placed *in the gap* between services, not overlapping them
- **Legend Placement:** **CRITICAL.** Must be placed outside all boundary boxes. Calculate the lowest Y-coordinate of all boundaries and place the legend at least 20px below it.

### Document Structure

1. **Header:** Title with a pulsing dot indicator and subtitle
2. **Main SVG:** The diagram contained within a rounded border card
3. **Summary Cards:** A grid of three cards below the diagram for high-level details
4. **Footer:** Minimal metadata

### architecture-diagram linked files

- `references/architecture-diagram-template.html` — full working HTML template with all component types, arrow styles, security groups, region boundaries, and the legend

---

## ideation — Creative Constraint-Driven Project Generation

Use when the user says 'I want to build something', 'give me a project idea', 'I'm bored', 'what should I make', 'inspire me', or any variant of 'I have tools but no direction'.

### When to use

Works for code, art, hardware, writing, tools, and anything that can be made. Generate project ideas through creative constraints. Constraint + direction = creativity.

### How It Works

1. **Pick a constraint** from the library below — random, or matched to the user's domain/mood
2. **Interpret it broadly** — a coding prompt can become a hardware project, an art prompt can become a CLI tool
3. **Generate 3 concrete project ideas** that satisfy the constraint
4. **If they pick one, build it** — create the project, write the code, ship it

### The Rule

Every prompt is interpreted as broadly as possible. "Does this include X?" → Yes. The prompts provide direction and mild constraint. Without either, there is no creativity.

### Constraint Library

#### For Developers

**Solve your own itch:**
Build the tool you wished existed this week. Under 50 lines. Ship it today.

**Automate the annoying thing:**
What's the most tedious part of your workflow? Script it away.

**The CLI tool that should exist:**
Think of a command you've wished you could type. Now build it.

**Nothing new except glue:**
Make something entirely from existing APIs, libraries, and datasets. The only original contribution is how you connect them.

**Frankenstein week:**
Take something that does X and make it do Y.

**Subtract:**
How much can you remove from a codebase before it breaks? Strip a tool to its minimum viable function.

**High concept, low effort:**
A deep idea, lazily executed. The concept should be brilliant. The implementation should take an afternoon.

#### For Makers & Artists

**Blatantly copy something:**
Pick something you admire — a tool, an artwork, an interface. Recreate it from scratch.

**One million of something:**
One million is both a lot and not that much. One million pixels is a 1MB photo.

**Make something that dies:**
A website that loses a feature every day. A chatbot that forgets.

**Do a lot of math:**
Generative geometry, shader golf, mathematical art, computational origami.

#### For Anyone

**Text is the universal interface:**
Build something where text is the only interface. No buttons, no graphics, just words in and words out.

**Start at the punchline:**
Think of something that would be a funny sentence. Work backwards to make it real.

**Hostile UI:**
Make something intentionally painful to use. A password field that requires 47 conditions.

**Take two:**
Remember an old project. Do it again from scratch. No looking at the original.

### Matching Constraints to Users

| User says | Pick from |
|-----------|-----------|
| "I want to build something" (no direction) | Random — any constraint |
| "I'm learning [language]" | Blatantly copy something, Automate the annoying thing |
| "I want something weird" | Hostile UI, Frankenstein week, Start at the punchline |
| "I want something useful" | Solve your own itch, The CLI that should exist |
| "I'm burned out" | High concept low effort, Make something that dies |
| "Weekend project" | Nothing new except glue, Start at the punchline |

### Output Format

```
## Constraint: [Name]
> [The constraint, one sentence]

### Ideas

1. **[One-line pitch]**
   [2-3 sentences: what you'd build and why it's interesting]
   ⏱ [weekend / week / month] • 🔧 [stack]
```

### ideation linked files

- `references/ideation-full-prompts.md` — 30+ additional constraints across communication, scale, philosophy, transformation, and more

### Attribution

Constraint approach inspired by [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html). Adapted and expanded for software development and general-purpose ideation.

---

## popular-web-designs — 54 Real Design Systems as HTML/CSS

54 real-world design systems ready for use when generating HTML/CSS. Each template captures a
site's complete visual language: color palette, typography hierarchy, component styles, spacing
system, shadows, responsive behavior, and practical agent prompts with exact CSS values.

### When to use popular-web-designs

- User says: "build a page that looks like Stripe", "make it look like Linear", "design like Vercel"
- User wants a UI, landing page, or dashboard design
- User wants real, proven design system tokens for HTML generation

### How to Use

1. Pick a design from the catalog below
2. Load it: `skill_view(name="creative-design", file_path="references/popular-web-designs/<site>.md")`
   (e.g. `references/popular-web-designs/stripe.md`)
3. Use the design tokens and component specs when generating HTML
4. Pair with the `generative-widgets` skill to serve the result via cloudflared tunnel

### HTML Generation Pattern

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Page Title</title>
  <!-- Paste the Google Fonts <link> from the template's Hermes notes -->
  <link href="https://fonts.googleapis.com/css2?family=..." rel="stylesheet">
  <style>
    :root {
      /* Apply the template's color palette as CSS custom properties */
    }
    body {
      /* Apply typography from template */
    }
    /* Apply component styles from template */
  </style>
</head>
<body>
  <!-- Build using component specs from the template -->
</body>
</html>
```

Write the file with `write_file`, serve via `generative-widgets`, and verify with `browser_vision`.

### Font Substitution Reference

| Proprietary Font | CDN Substitute | Character |
|---|---|---|
| Geist / Geist Sans | Geist (on Google Fonts) | Geometric, compressed tracking |
| Geist Mono | Geist Mono (on Google Fonts) | Clean monospace |
| sohne-var (Stripe) | Source Sans 3 | Light weight elegance |
| Berkeley Mono | JetBrains Mono | Technical monospace |
| Airbnb Cereal VF | DM Sans | Rounded, friendly |
| Circular (Spotify) | DM Sans | Geometric, warm |
| figmaSans | Inter | Clean humanist |
| IBM Plex Sans/Mono | IBM Plex Sans/Mono | Available on Google Fonts |

### Design Catalog

#### AI & Machine Learning

| Template | Site | Style |
|---|---|---|
| `claude.md` | Anthropic Claude | Warm terracotta accent, clean editorial layout |
| `cohere.md` | Cohere | Vibrant gradients, data-rich dashboard |
| `elevenlabs.md` | ElevenLabs | Dark cinematic UI, audio-waveform aesthetics |
| `minimax.md` | Minimax | Bold dark interface with neon accents |
| `mistral.ai.md` | Mistral AI | French-engineered minimalism, purple-toned |
| `ollama.md` | Ollama | Terminal-first, monochrome simplicity |
| `opencode.ai.md` | OpenCode AI | Developer-centric dark theme, full monospace |
| `replicate.md` | Replicate | Clean white canvas, code-forward |
| `runwayml.md` | RunwayML | Cinematic dark UI, media-rich layout |
| `together.ai.md` | Together AI | Technical, blueprint-style design |
| `voltagent.md` | VoltAgent | Void-black canvas, emerald accent, terminal-native |
| `x.ai.md` | xAI | Stark monochrome, futuristic minimalism, full monospace |

#### Developer Tools & Platforms

| Template | Site | Style |
|---|---|---|
| `cursor.md` | Cursor | Sleek dark interface, gradient accents |
| `expo.md` | Expo | Dark theme, tight letter-spacing, code-centric |
| `linear.app.md` | Linear | Ultra-minimal dark-mode, precise, purple accent |
| `lovable.md` | Lovable | Playful gradients, friendly dev aesthetic |
| `mintlify.md` | Mintlify | Clean, green-accented, reading-optimized |
| `posthog.md` | PostHog | Playful branding, developer-friendly dark UI |
| `raycast.md` | Raycast | Sleek dark chrome, vibrant gradient accents |
| `resend.md` | Resend | Minimal dark theme, monospace accents |
| `sentry.md` | Sentry | Dark dashboard, data-dense, pink-purple accent |
| `supabase.md` | Supabase | Dark emerald theme, code-first developer tool |
| `superhuman.md` | Superhuman | Premium dark UI, keyboard-first, purple glow |
| `vercel.md` | Vercel | Black and white precision, Geist font system |
| `warp.md` | Warp | Dark IDE-like interface, block-based command UI |
| `zapier.md` | Zapier | Warm orange, friendly illustration-driven |

#### Infrastructure & Cloud

| Template | Site | Style |
|---|---|---|
| `clickhouse.md` | ClickHouse | Yellow-accented, technical documentation style |
| `composio.md` | Composio | Modern dark with colorful integration icons |
| `hashicorp.md` | HashiCorp | Enterprise-clean, black and white |
| `mongodb.md` | MongoDB | Green leaf branding, developer documentation focus |
| `sanity.md` | Sanity | Red accent, content-first editorial layout |
| `stripe.md` | Stripe | Signature purple gradients, weight-300 elegance |

#### Design & Productivity

| Template | Site | Style |
|---|---|---|
| `airtable.md` | Airtable | Colorful, friendly, structured data aesthetic |
| `cal.md` | Cal.com | Clean neutral UI, developer-oriented simplicity |
| `clay.md` | Clay | Organic shapes, soft gradients, art-directed layout |
| `figma.md` | Figma | Vibrant multi-color, playful yet professional |
| `framer.md` | Framer | Bold black and blue, motion-first, design-forward |
| `intercom.md` | Intercom | Friendly blue palette, conversational UI patterns |
| `miro.md` | Miro | Bright yellow accent, infinite canvas aesthetic |
| `notion.md` | Notion | Warm minimalism, serif headings, soft surfaces |
| `pinterest.md` | Pinterest | Red accent, masonry grid, image-first layout |
| `webflow.md` | Webflow | Blue-accented, polished marketing site aesthetic |

#### Fintech & Crypto

| Template | Site | Style |
|---|---|---|
| `coinbase.md` | Coinbase | Clean blue identity, trust-focused, institutional feel |
| `kraken.md` | Kraken | Purple-accented dark UI, data-dense dashboards |
| `revolut.md` | Revolut | Sleek dark interface, gradient cards, fintech precision |
| `wise.md` | Wise | Bright green accent, friendly and clear |

#### Enterprise & Consumer

| Template | Site | Style |
|---|---|---|
| `airbnb.md` | Airbnb | Warm coral accent, photography-driven, rounded UI |
| `apple.md` | Apple | Premium white space, SF Pro, cinematic imagery |
| `bmw.md` | BMW | Dark premium surfaces, precise engineering aesthetic |
| `ibm.md` | IBM | Carbon design system, structured blue palette |
| `nvidia.md` | NVIDIA | Green-black energy, technical power aesthetic |
| `spacex.md` | SpaceX | Stark black and white, full-bleed imagery, futuristic |
| `spotify.md` | Spotify | Vibrant green on dark, bold type, album-art-driven |
| `uber.md` | Uber | Bold black and white, tight type, urban energy |

### Choosing a Design

| Category | Best fits |
|---|---|
| Developer tools / dashboards | Linear, Vercel, Supabase, Raycast, Sentry |
| Documentation / content sites | Mintlify, Notion, Sanity, MongoDB |
| Marketing / landing pages | Stripe, Framer, Apple, SpaceX |
| Dark mode UIs | Linear, Cursor, ElevenLabs, Warp, Superhuman |
| Light / clean UIs | Vercel, Stripe, Notion, Cal.com, Replicate |
| Playful / friendly | PostHog, Figma, Lovable, Zapier, Miro |
| Premium / luxury | Apple, BMW, Stripe, Superhuman, Revolut |
| Data-dense / dashboards | Sentry, Kraken, Cohere, ClickHouse |
| Monospace / terminal aesthetic | Ollama, OpenCode, x.ai, VoltAgent |

### popular-web-designs linked files

- `references/popular-web-designs/` — directory containing all 54 design system templates as `.md` files (load individually via `skill_view(file_path=...)`)