# Diagrams

kami's drawing capability. **14 diagram types** covering structural, process, and data chart scenarios. All wear kami's skin (parchment + ink-blue + warm grays). No second design system.

Every diagram is a **self-contained HTML + inline SVG**. No Mermaid, no JS, no build step. Browse them as standalone pages, or copy the `<svg>...</svg>` block into a long-doc `<figure>` to embed.

---

## 1. Selection

| Showing… | Use | Template |
|---|---|---|
| System components + connections | **Architecture** | `assets/diagrams/architecture.html` |
| Decision branches, "if A then B else C" | **Flowchart** | `assets/diagrams/flowchart.html` |
| Two-axis positioning / prioritization | **Quadrant** | `assets/diagrams/quadrant.html` |
| Category comparison (revenue, market share, quarterly) | **Bar Chart** | `assets/diagrams/bar-chart.html` |
| Trend over time (stock price, growth rate, time series) | **Line Chart** | `assets/diagrams/line-chart.html` |
| Proportional breakdown (spend, user segments, share) | **Donut Chart** | `assets/diagrams/donut-chart.html` |
| Finite states + directed transitions (lifecycle, state machine) | **State Machine** | `assets/diagrams/state-machine.html` |
| Time axis + milestone events (roadmap, project progress) | **Timeline** | `assets/diagrams/timeline.html` |
| Cross-responsibility process (multi-role, API request path) | **Swimlane** | `assets/diagrams/swimlane.html` |
| Hierarchical relationships (org chart, module deps, directory tree) | **Tree** | `assets/diagrams/tree.html` |
| Vertically stacked system layers (OSI, application stack) | **Layer Stack** | `assets/diagrams/layer-stack.html` |
| Set intersections (feature overlap, audience comparison, capability map) | **Venn** | `assets/diagrams/venn.html` |
| OHLC price action (stock price, trading days, up/down candles) | **Candlestick** | `assets/diagrams/candlestick.html` |
| Revenue bridge, valuation decomposition, cash flow breakdown | **Waterfall** | `assets/diagrams/waterfall.html` |

Not on the list:
- **Compare two things**: use a table. A three-column table beats any diagram of a binary contrast.
- **One box with a label**: delete the box, write the sentence.

### The question before drawing

> Would a well-written paragraph teach the reader less than this diagram?

If "no", don't draw. Diagrams add signal to hierarchy, direction, and magnitude. They don't decorate prose.

---

## 2. Complexity budget

**Target density: 4/10**. Enough to be technically complete, not so dense the reader needs a guide.

- Nodes > 9 -> this is two diagrams, not one
- Two nodes that always travel together -> they're one node
- A line whose meaning is obvious from layout -> remove the line
- 5 nodes in ink-blue -> you haven't decided what's focal

**Focal rule**: 1-2 focal elements per diagram (`#1B365D` stroke + `#EEF2F7` fill). Everything else goes neutral. Focal signal comes from contrast, not count.

---

## 3. Embedding in long-doc / portfolio

### Standalone preview

Open `assets/diagrams/architecture.html` (or `flowchart.html`, `quadrant.html`) directly. Each file is a complete HTML page with title, SVG, and caption.

### Embed in a kami document

Extract **only the `<svg>...</svg>` block** from the template (leave the frame / h1 / eyebrow behind). Drop it into a long-doc `<figure>`:

```html
<figure>
  <svg viewBox="0 0 960 460" xmlns="http://www.w3.org/2000/svg">
    <!-- svg content copied from architecture.html -->
  </svg>
  <figcaption>Figure 1. {{Short editorial caption in serif.}}</figcaption>
</figure>
```

`long-doc.html` already styles `figure` and `figcaption`. No extra CSS required.

### Editing nodes / text

Edit the `<text>` and `<rect>` values directly. Rules:

- **All coordinates, widths, and gaps must be divisible by 4.** This is the anti-AI-slop floor. Break it once and the diagram starts looking "close enough".
- Node widths: 128 / 144 / 160 (three tiers, don't add more). Small diagrams (viewBox width < 360) may compress to 2 tiers, but still keep it 2 - don't tailor each node.
- Node heights: 32 (pill) / 64 (standard)
- Font sizes: 7 (small mono label) / 9 (sublabel mono) / 12 (name sans)
- **Arrow endpoints land exactly on node edges**: start `(box.x + box.w, box.y + box.h/2)`, end `(box.x, box.y + box.h/2)`, not "close enough". A 10px gap is visible to the eye.
- **SVG top padding**: the `y` in `<text y="…">` is the baseline. `y` must be ≥ font-size × 1.2, otherwise the tops of capital letters extend above the viewBox and get clipped (classic symptom: "TOOLS" renders as "TOULS"). Either pad the viewBox at the top or move `y` into the safe zone.
- **Loop arc control points**: for a four-cardinal-node ring, each arc is a Q-curve whose control point sits at the **outer intersection of the two adjacent tangent axes**, not at a node corner. Example for PLAN (top) → ACT (right): start = PLAN's right-edge midpoint, end = ACT's top-edge midpoint, control = `(ACT.x + ACT.w/2, PLAN.y + PLAN.h/2)`. This gives a pure horizontal tangent at departure and pure vertical at arrival, reading as a clean quarter-circle. Control at the node corner produces a squashed arc.
- **Closed loops need a dashed framing ring**: four directed arcs alone force the reader to mentally connect them into a loop. A dashed circle centered on the visual center (radius slightly larger than center-to-inner-edge distance) makes the loop immediately readable. Draw the ring below the nodes; solid node fills mask where the ring crosses each node; the ring shows only between nodes.
- **Chevron arrows, not filled triangles**: use `<path d="M2 1 L8 5 L2 9" fill="none" stroke=... stroke-width="1.5" stroke-linecap="round"/>`. A filled triangle reads as technical UI; an open two-stroke chevron reads as editorial schematic. kami defaults to chevron. **WeasyPrint does not support `<marker orient="auto">`**: all markers render at 0° (pointing right). The fix is to skip `<marker>` and draw each arrowhead as a manual chevron `<path>` with hardcoded direction (see production.md #15).

### Color token map

Shared tokens across the three diagrams, mapping directly to kami's design system:

| SVG role | kami token | Value |
|---|---|---|
| Canvas | `--parchment` | `#f5f4ed` |
| Standard node fill | (white) | `#ffffff` |
| Standard node stroke | `--near-black` | `#141413` |
| Store node fill | near-black 5% | `rgba(20,20,19,0.05)` |
| Store node stroke | `--olive` | `#504e49` |
| Cloud node fill | near-black 3% | `rgba(20,20,19,0.03)` |
| Cloud node stroke | near-black 30% | `rgba(20,20,19,0.30)` |
| External node fill | olive 8% | `rgba(94,93,89,0.08)` |
| External node stroke | `--stone` | `#6b6a64` |
| **Focal fill** | `--brand-tint` | `#EEF2F7` |
| **Focal stroke** | `--brand` | `#1B365D` |
| Standard arrow | `--olive` | `#504e49` |
| Focal arrow | `--brand` | `#1B365D` |
| Primary text | `--near-black` | `#141413` |
| Secondary text | `--olive` | `#504e49` |
| Tertiary text / small mono label | `--stone` | `#6b6a64` |

Don't add a fourth state ("warning amber", "success green"). kami has one accent.

### Embedded font calibration (override standalone sizes)

Standalone diagram sizes (`7 / 9 / 12`) are too small once embedded in A4 long-doc / portfolio. The render width drops to about 470pt while the viewBox stays at 1000, so the scale factor is roughly `0.47`. To keep diagram text aligned with the 11pt body baseline, raise the SVG `font-size` values when embedding:

| Visual target | Visual weight | SVG `font-size` |
|---|---|---|
| Same as h2 / focal node name | 11pt | **24** |
| Same as body | 11pt | **22-24** |
| Same as h3 / sub-label | 9-10pt | **18-20** |
| Same as caption | 8pt | **15-16** |
| Mono uppercase tag (letter-spacing 2.5) | 7pt | **14** |

Compensation factor is roughly `1.8-2.0x` over standalone. `font-size: 24` with `font-weight: 600` and the body serif renders at about 1.05x the body, which reads as h2-equivalent without dominating the page.

For tall diagrams (e.g. 5-layer stack), a working layout is `viewBox: 0 0 1000 560`, layer height `88`, gap `8`, and inside each layer:

- Tag baseline `y+24`, font-size `14`, mono, letter-spacing `2.5`
- Name baseline `y+54`, font-size `24`, serif weight `600`
- Description baseline `y+76`, font-size `14`, mono, normal
- Right-side role tag `x=900`, `text-anchor=end`, font-size `13`

### In-SVG header line (figure number + title)

For embedded diagrams, put the "FIGURE N · TITLE" header inside the SVG instead of using `<figcaption>`. The diagram becomes a self-contained editorial unit, and the brand-colored header doubles as a section anchor.

```svg
<text x="80" y="38" fill="#1B365D" font-size="13" font-weight="600"
      font-family="mono" letter-spacing="3">FIGURE  1</text>
<text x="195" y="38" fill="#504e49" font-size="13"
      font-family="mono" letter-spacing="3">DIAGRAM TITLE GOES HERE</text>
<line x1="80" y1="52" x2="920" y2="52"
      stroke="#1B365D" stroke-width="0.8"/>
```

Two spaces between `FIGURE` and the number. With `letter-spacing: 3`, a single space lets the digit collide with the preceding letter.

---

## 4. Icon style

Icons live inside `<svg>` blocks alongside diagram nodes. Draw them with the same primitives (`rect`, `circle`, `line`, `path`) used for nodes - no imported icon fonts, no SVG sprites.

**Rules**:
- Single line, stroke 1pt-1.5pt, no fill
- Stroke weight stays consistent within one diagram. Never mix 1pt and 1.5pt icons in the same figure
- No drop shadow, gradient, 3D, or glassmorphism
- No emoji-style faces, mascots, or expressive characters - this is editorial schematic, not playful
- Focal icons may use `--brand` stroke or fill, but the figure's total ink-blue area still respects the 5% cap

### Canonical shapes

When an icon represents a recurring concept, use the canonical form rather than inventing a new one:

| Concept | Shape |
|---|---|
| Terminal / CLI | rounded rectangle, three dots top-left |
| Document / spec | rectangle, three short horizontal lines |
| Checklist / verification | rectangle, two check marks |
| Gear / system | 8-tooth gear outline |
| Magnifier / inspect | circle with 45° handle |
| Shield / safety | shield silhouette |
| Cloud / hosted service | three-arc cloud outline |
| Chip / hardware | square with leg lines on four sides |
| GPU / compute rack | rectangular stack with port indicators |

### Human and robot figures

Avoid human figures and anthropomorphic AI in editorial diagrams. If a person must appear, use a minimal line drawing without facial detail. Industrial robots may be line-art mechanical structures, but stop short of patent-illustration density.

When in doubt, omit the icon entirely. A clean text label beats a cute icon in editorial schematic style. Add an icon only when it carries information the label cannot (e.g. distinguishing "cloud service" from "on-device compute" at a glance).

---

## 5. AI-slop anti-patterns

Scan for these when drawing or reviewing:

| Anti-pattern | Why it fails |
|---|---|
| Dark mode + cyan / purple glow | Cheap "technical" signifier with no design decision |
| All nodes identical size | Destroys hierarchy |
| JetBrains Mono as the universal "dev" font | Mono is for technical content (ports, URLs, fields). Names go in sans. |
| Legend floating inside the diagram area | Collides with nodes |
| Arrow labels without a masking rect | Line bleeds through the text |
| Vertical `writing-mode` text on arrows | Unreadable |
| Three equal-width summary cards as a default | Template feel. Vary widths. |
| `box-shadow` on anything | kami only permits ring / whisper |
| `rounded-2xl` / border-radius above 10px | Max 6-10px. Beyond, it starts to look like App Store chrome. |
| Ink Blue on every "important" node | Focal rule is 1-2, not a signaling system |
| Decorative icons | Disaster |
| Gradient backgrounds | kami forbids them |
| Focal color contradicts the caption's claim | Caption says "Simple **core**", but the ACT node is painted ink-blue - two focals competing. Focal color must match the word emphasized (`<span class="hl">`) in the caption |
| Cycle diagram with a dashed ring AND four directed arcs | Same loop drawn twice; reader thinks there are two flows |
| SVG text clipped at the viewBox top | `text` y is the baseline; cap letters extend above y=0. Pad the top by font-size × 1.2 or adjust the viewBox |
| 5-10px gap between arrow endpoint and node edge | Reads as "arrow floating in space". Anchor endpoints to exact `box.x / box.x+w / box.y / box.y+h` |
| Per-node custom widths within one diagram | Four steps at widths 60 / 76 / 80 / 100 feel hand-patched. Small diagram: 2 tiers. Large: 3 tiers. That's the full budget |
| Porting an external diagram with one accent color per node type (purple/amber/green/red) | kami has one accent. When adapting external diagrams, migrate the focal to whichever element the caption's `<span class="hl">` emphasizes; concentrate color there, keep all other nodes neutral |
| Ring diagram: every node is a single word, center is empty | Four labeled boxes looping with no anchor. Either add a subtitle to each node or place one line of text at the center (exit condition, LOC count, etc.). Pick one. |

---

## 6. Common pairings

### Technical white paper
- Architecture (system overview) + built-in timeline (from long-doc)
- One architecture diagram per chapter, maximum. If you want two, the chapter is covering two topics and should split.

### Portfolio project page
- Quadrant (competitive positioning) or architecture (the layer you owned)
- **Not every project needs a diagram.** Only when the diagram says something prose can't.

### One-pager
- Quadrant (priority) or flowchart (decision path)
- One diagram only. If you're tempted to add a second, kill the weaker one.

### Resume
- **No diagrams.** Resume real-estate costs more than diagrams. Rare exception: a URL to a portfolio diagram when showing system-level capability.

### Slides
- One diagram per slide, max. The diagram is the body. Text is caption, not a sidebar.

---

## 7. Data charts (bar / line / donut)

Five data-driven chart types for investment reports, financial comparisons, and market-share breakdowns. Like the first three diagram types, all are self-contained HTML + inline SVG, embeddable in any kami document.

### Color palette (derived from kami warm palette)

| Role | Value | Use |
|---|---|---|
| Primary series | `#1B365D` ink-blue | First group / focal data |
| Series 2 | `#504e49` olive | Second group |
| Series 3 | `#6b6a64` stone | Third group |
| Series 4 | `#b8b7b0` light-stone | Fourth group |
| Series 5 | `#d4d3cd` mist | Fifth group |
| Series 6 | `#EEF2F7` brand-tint | Sixth group |
| Grid lines | `#e8e7e1` | Axes / reference lines |
| Data labels | `#141413` near-black | Numeric text |

### Data limits

| Chart | Max categories | Max series | Template |
|---|---|---|---|
| Bar chart | 8 groups | 3 series | `assets/diagrams/bar-chart.html` |
| Line chart | 12 points | 3 lines | `assets/diagrams/line-chart.html` |
| Donut chart | 6 segments | n/a | `assets/diagrams/donut-chart.html` |
| Candlestick | 30 days | n/a | `assets/diagrams/candlestick.html` |
| Waterfall | 8 segments | n/a | `assets/diagrams/waterfall.html` |

### Editing data

Each file has `<!-- DATA START -->` / `<!-- DATA END -->` comments. Only change SVG elements between those markers (`<rect>` coordinates, `<polyline>` points, `<path>` arcs, `<text>` values). Leave surrounding structure and styles untouched.

**Coordinate rules (same as the first three diagram types)**:
- All coordinates divisible by 4
- Bar chart corner radius `rx=2` (distinct from node radius 6-10)
- Line chart: `<polyline>` points format `"x1,y1 x2,y2 ..."`, data points marked with `<circle>`
- Donut chart: `<path>` arcs use `A R R 0 large-arc sweep_flag x y`; `large-arc=1` only when segment > 180°

**Bar / line chart Y-axis formula** (default scale: max=140, chart-height=280, scale=2):
```
bar_height = value × 2
bar_top_y  = 320 - bar_height   (baseline y = 320)
dot_y      = 320 - value × 2
```

**Donut arc coordinates** (cx=300 cy=200 R=136 r=76, clockwise from top at -90°):
```
angle_start = -90 + sum_of_previous_percentages × 3.6
angle_end   = angle_start + this_percentage × 3.6
outer_x = 300 + 136 × cos(angle_deg × π/180)
outer_y = 200 + 136 × sin(angle_deg × π/180)
inner_x = 300 + 76  × cos(angle_deg × π/180)
inner_y = 200 + 76  × sin(angle_deg × π/180)
```

**Candlestick Y-axis formula** (default: price range 100-160, chart-height=280, scale=4.67):
```
candle_y = 320 - (price - 100) * 4.67
Up candle: fill=#1B365D (close > open), body from open_y to close_y
Down candle: fill=#6b6a64 (close < open), body from close_y to open_y
Wick: 1.2px stroke from high_y to low_y, centered on candle
```

**Waterfall formula** (default: max=200, chart-height=280, scale=1.4):
```
bar_y = 320 - value * 1.4
Floating bars: top = running_total_y, height = abs(delta) * 1.4
Positive: fill=#1B365D · Negative: fill=#6b6a64 · Total: fill=#4d4c48
Connector: dashed 0.8px #b8b7b0 between adjacent bar edges
```

---

## 8. Build / preview

```bash
python3 scripts/build.py diagram-architecture
python3 scripts/build.py diagram-flowchart
python3 scripts/build.py diagram-quadrant
python3 scripts/build.py diagram-bar-chart
python3 scripts/build.py diagram-line-chart
python3 scripts/build.py diagram-donut-chart
python3 scripts/build.py diagram-state-machine
python3 scripts/build.py diagram-timeline
python3 scripts/build.py diagram-swimlane
python3 scripts/build.py diagram-tree
python3 scripts/build.py diagram-layer-stack
python3 scripts/build.py diagram-venn
python3 scripts/build.py diagram-candlestick
python3 scripts/build.py diagram-waterfall

# or all
python3 scripts/build.py
```

Or just open `assets/diagrams/*.html` in a browser.

---

## 9. Credit

This capability is inspired by Cathryn Lavery's [diagram-design](https://github.com/cathrynlavery/diagram-design) (a Claude Code skill with 13 editorial diagram types). kami borrowed the **approach** (inline SVG, semantic tokens, complexity budget, anti-slop table). Not the full catalog.
