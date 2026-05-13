# kami · Cheatsheet

One-page quick reference. Scan before filling a template or tweaking a detail. Full spec in `references/design.md`.

## Ten invariants

1. Page background `#f5f4ed` (parchment), never pure white
2. Single accent: ink-blue `#1B365D`
3. All grays **warm-toned** (yellow-brown undertone), no cool blue-gray
4. English: serif for headlines and body. Chinese: serif headlines, sans body. Sans for UI only
5. Serif weight locked at 500, no bold
6. Line-height: headlines 1.1-1.3 / dense 1.4-1.45 / reading 1.5-1.55
7. Letter-spacing: Chinese body 0.3pt; English body 0; tracking for short labels only
8. Tag backgrounds solid hex, no rgba (WeasyPrint double-rectangle bug)
9. Depth via ring / whisper shadow, no hard drop shadows
10. No italic in templates or demos

## Sources and Materials


| Trigger                                                   | Do first                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| Latest product / version / launch / funding / market data | Check reliable sources first                                        |
| Company / product / project branded doc                   | Confirm logo, product image, or UI screenshot                       |
| Key number or result                                      | Record the source; if unverifiable, write magnitude or mark missing |
| Missing material                                          | Mark the gap or ask the user; do not use unrelated imagery          |


## Color


| Role         | Hex           | Use                                                 |
| ------------ | ------------- | --------------------------------------------------- |
| Parchment    | `#f5f4ed`     | Page background                                     |
| Ivory        | `#faf9f5`     | Card / lifted container                             |
| Warm Sand    | `#e8e6dc`     | Button / interactive surface                        |
| Dark Surface | `#30302e`     | Dark container                                      |
| Deep Dark    | `#141413`     | Dark page background                                |
| **Brand**    | `**#1B365D`** | **Accent · CTA · title left bar (≤ 5% of surface)** |
| Ink Light    | `#2D5A8A`     | Links on dark surfaces                              |
| Near Black   | `#141413`     | Primary text                                        |
| Dark Warm    | `#3d3d3a`     | Secondary text · table headers · links              |
| Olive        | `#504e49`     | Subtext · descriptions                              |
| Stone        | `#6b6a64`     | Tertiary · metadata                                 |
| Border       | `#e8e6dc`     | Primary border · section divider                    |
| Border Soft  | `#e5e3d8`     | Secondary border · row separator                    |


**rgba -> solid** (parchment base + ink-blue):


| Alpha    | Solid                       |
| -------- | --------------------------- |
| 0.08     | `#EEF2F7`                   |
| 0.14     | `#E4ECF5`                   |
| **0.18** | `**#E4ECF5`** ← default tag |
| 0.22     | `#D0DCE9`                   |
| 0.30     | `#D6E1EE`                   |


## Type (print pt)


| Role       | Size | Weight | Line-height |
| ---------- | ---- | ------ | ----------- |
| Display    | 36   | 500    | 1.10        |
| H1         | 22   | 500    | 1.20        |
| H2         | 16   | 500    | 1.25        |
| H3         | 13   | 500    | 1.30        |
| Body Lead  | 11   | 400    | 1.55        |
| Body       | 10   | 400    | 1.55        |
| Body Dense | 9.2  | 400    | 1.42        |
| Caption    | 9    | 400    | 1.45        |
| Label      | 9    | 600    | 1.35        |
| Tiny       | 9    | 400    | 1.40        |


Screen (px) ≈ pt × 1.33.
Minimum floor: web text >= 12px, PDF text >= 9pt.

## Font stacks

Each language uses a single serif for the entire page. `--sans` always equals `var(--serif)`.

English:

```css
--serif: Charter, Georgia, Palatino,
         "Times New Roman", serif;
--sans:  var(--serif);
--mono:  "JetBrains Mono", "SF Mono", "Fira Code",
         Consolas, Monaco, monospace;
```

Chinese:

```css
--serif: "TsangerJinKai02", "Source Han Serif SC",
         "Noto Serif CJK SC", "Songti SC", "STSong",
         Georgia, serif;
--sans:  var(--serif);
--mono:  "JetBrains Mono", "SF Mono", Consolas,
         "TsangerJinKai02", "Source Han Serif SC",
         monospace;
```

Japanese:

```css
--serif: "YuMincho", "Yu Mincho", "Hiragino Mincho ProN",
         "Noto Serif CJK JP", "Source Han Serif JP",
         "TsangerJinKai02", Georgia, serif;
--sans:  var(--serif);
```

Any font-family that may render Chinese or Japanese must include a CJK fallback, including `@page` footer text, `pre`, `code`, and SVG labels. A pure mono stack can render missing glyph boxes in WeasyPrint.

## Spacing (4pt base)


| Tier | Value    | Use                    |
| ---- | -------- | ---------------------- |
| xs   | 2-3pt    | Inline                 |
| sm   | 4-5pt    | Tag padding            |
| md   | 8-10pt   | Component interior     |
| lg   | 16-20pt  | Between components     |
| xl   | 24-32pt  | Section-title margin   |
| 2xl  | 40-60pt  | Between major sections |
| 3xl  | 80-120pt | Between chapters       |


**Page margins (A4)**


| Document  | T · R · B · L        |
| --------- | -------------------- |
| Resume    | 11 · 13 · 11 · 13 mm |
| One-Pager | 15 · 18 · 15 · 18 mm |
| Long Doc  | 20 · 22 · 22 · 22 mm |
| Letter    | 25 mm all sides      |
| Portfolio | 12 · 15 · 12 · 15 mm |


## Radius scale

`4pt -> 6pt -> 8pt (default) -> 12pt -> 16pt -> 24pt -> 32pt (hero)`

## Common CSS snippets

### Card

```css
.card {
  background: var(--ivory);
  border: 0.5pt solid var(--border-cream);
  border-radius: 8pt;
  padding: 16pt 20pt;
  transition: box-shadow 0.2s;
}
.card:hover {
  box-shadow: 0 4pt 24pt rgba(0, 0, 0, 0.05);  /* whisper shadow */
}
```

### Tag (default lightest solid)

```css
.tag {
  background: #EEF2F7;            /* 0.08 equivalent */
  color: var(--brand);
  font-size: 9pt; font-weight: 600;
  padding: 1pt 5pt;
  border-radius: 2pt;
  letter-spacing: 0.4pt;
  text-transform: uppercase;
}
```

### Section title (brand left bar is the signature move)

```css
.section-title {
  font-family: var(--serif);
  font-size: 14pt; font-weight: 500;
  color: var(--near-black);
  margin: 24pt 0 10pt 0;
  border-left: 2.5pt solid var(--brand);
  border-radius: 1.5pt;
  padding-left: 8pt;
}
```

### Table (kami-table)

Base class works on bare `<table>` or `.kami-table`. Add variant classes for density/alignment:

```css
/* Base */
table, .kami-table {
  width: 100%; border-collapse: collapse;
  font-size: 9.5pt; margin: 12pt 0; break-inside: avoid;
}
table th { text-align: left; font-weight: 500; color: var(--dark-warm);
  padding: 6pt 8pt; border-bottom: 1pt solid var(--border); }
table td { padding: 5pt 8pt; border-bottom: 0.3pt solid var(--border-soft);
  vertical-align: top; }
```


| Variant   | Class              | Effect                                               |
| --------- | ------------------ | ---------------------------------------------------- |
| Compact   | `.compact`         | 8pt font, tight padding (data-dense tables)          |
| Financial | `.financial`       | Right-align all columns except first, `tabular-nums` |
| Striped   | `.striped`         | Alternating `var(--ivory)` row background            |
| Total row | `.total` on `<tr>` | Bold, brand top border, no bottom border             |


Combine freely: `<table class="kami-table financial striped">`.

### Metric (data card)

```css
.metric { display: flex; align-items: baseline; gap: 6pt; }
.metric-value {
  font-family: var(--serif); font-size: 16pt; font-weight: 500;
  color: var(--brand);
  font-variant-numeric: tabular-nums;
}
.metric-label { font-size: 9pt; color: var(--olive); }
```

### Quote

```css
.quote {
  border-left: 2pt solid var(--brand);
  padding: 4pt 0 4pt 14pt;
  color: var(--olive);
  line-height: 1.55;
}
```

## Diagram components

Twelve built-in diagram types. Extract the `<svg>` block and embed in a `<figure>` in long-doc / portfolio:


| Type          | File                                 | Use                                             |
| ------------- | ------------------------------------ | ----------------------------------------------- |
| Architecture  | `assets/diagrams/architecture.html`  | System components and connections               |
| Flowchart     | `assets/diagrams/flowchart.html`     | Decision branches and flows                     |
| Quadrant      | `assets/diagrams/quadrant.html`      | 2×2 positioning                                 |
| Bar Chart     | `assets/diagrams/bar-chart.html`     | Category comparison (up to 8 groups × 3 series) |
| Line Chart    | `assets/diagrams/line-chart.html`    | Trends over time (up to 12 points × 3 lines)    |
| Donut Chart   | `assets/diagrams/donut-chart.html`   | Proportional breakdown (up to 6 segments)       |
| State Machine | `assets/diagrams/state-machine.html` | Finite states + directed transitions            |
| Timeline      | `assets/diagrams/timeline.html`      | Time axis + milestone events                    |
| Swimlane      | `assets/diagrams/swimlane.html`      | Cross-responsibility process flow               |
| Tree          | `assets/diagrams/tree.html`          | Hierarchical relationships                      |
| Layer Stack   | `assets/diagrams/layer-stack.html`   | Vertically stacked system layers                |
| Venn          | `assets/diagrams/venn.html`          | Set intersections and overlaps                  |
| Candlestick   | `assets/diagrams/candlestick.html`   | OHLC price history (up to 30 days)              |
| Waterfall     | `assets/diagrams/waterfall.html`     | Revenue bridge / decomposition                  |


Usage: extract the `<svg>` block from the HTML file and paste into the template's `<figure>` container.

**Data chart colors**: primary series `#1B365D` · secondary `#504e49` → `#6b6a64` → `#b8b7b0` → `#d4d3cd` → `#EEF2F7`.

**Editing data**: only modify elements between `<!-- DATA START -->` / `<!-- DATA END -->`, leave CSS untouched. All coordinates must be divisible by 4.

## Dark section

Alternate light/dark rhythm: add `.sd-alt` to any section container.

- Background switches to `--deep-dark` (`#141413`)
- Body text switches to `--warm-silver` (`#b0aea5`)
- Headings switch to `--ivory`
- Appropriate for: section-level light/dark alternation in long-doc / portfolio
- Restriction: showcase pages only, never in print templates

## Verification checks

`python3 scripts/build.py --verify [target]` checks source templates and slides in sequence:

1. Source file exists
2. WeasyPrint render to PDF for HTML / diagram targets
3. Page count check for strict targets
4. Font embedding check
5. PPTX generation for `slides` / `slides-en`

Source templates intentionally keep `{{...}}` fields. Run `python3 scripts/build.py --check-placeholders path/to/filled.html` on completed documents. Run `python3 scripts/build.py --check-density` to warn on pages with >25% trailing whitespace (skips cover).

## Content quality (one rule per type)

Full quality bars in `references/writing.md`. The single most important rule for each document type:


| Document      | Core quality rule                                                                    |
| ------------- | ------------------------------------------------------------------------------------ |
| Resume        | Every bullet: Action + Scope + Measurable Result + Business Outcome                  |
| Portfolio     | Open with the problem and stakes, not the project name                               |
| Slides        | Slide titles are full sentences (assertions), not topic labels                       |
| Equity Report | Lead with variant perception: what you see that the market doesn't                   |
| Long Document | Each chapter claim paragraph must survive the "so what?" test                        |
| One-Pager     | Metrics are the headline; if the 4 cards don't tell the story, the metrics are wrong |
| Letter        | First paragraph states purpose in one sentence                                       |
| Changelog     | One sentence per change, verb-led, user-facing language                              |


## Per-page font size strategy (Resume two-page)

Page 1 carries the projects section, which is the densest content. Page 2 carries open source, convictions, impact, skills, and education, which has more breathing room.

| Location | Class | Default | Dense (5 projects) |
|---|---|---|---|
| Project body | `.proj-text` | 9pt / lh 1.40 | 9pt / lh 1.38 |
| Timeline body | `.tl-body` | 9pt / lh 1.40 | 8.5pt (CN) |
| Summary | `.summary` | 9.2pt | 9pt via body |
| Section titles | `.section-title` | margin-top 5mm | 3.5mm |
| OS intro | `.os-intro` | 9.2pt | unchanged |
| Conviction body | `.conv-body` | 9pt | unchanged |
| Skills body | `.skill-body` | 9pt | unchanged |

**Reference config (5 projects + full page 2)**:

```html
<body class="resume--dense">
  <!-- page 1: 5 projects, timeline, summary -->
  <!-- page 2: OS grid (6), convictions, impact, skills, edu -->
</body>
```

Page 2 font sizes stay at template defaults. The density variant only tightens page 1 elements. If page 2 has unusually long content, reduce `.os-intro`, `.conv-body`, or `.skill-body` individually, never globally.

## Quick decisions


| Need                | Use                                                            |
| ------------------- | -------------------------------------------------------------- |
| Headline            | serif 500, line-height 1.10-1.30                               |
| Reading body (EN)   | serif 400, 9.5-10pt, 1.55                                      |
| Reading body (CN)   | sans 400, 9.5-10pt, 1.55                                       |
| Emphasize a number  | `color: var(--brand)`, no bold                                 |
| Divide two sections | 2.5pt brand left bar, or 0.5pt warm dotted                     |
| Quote               | 2pt brand left border + olive color                            |
| Code                | ivory bg + 0.5pt border + 6pt radius + mono                    |
| Primary button      | brand fill + ivory text                                        |
| Secondary button    | warm-sand + dark-warm                                          |
| Chapter start       | serif heading + 2.5pt brand left bar                           |
| Cover               | Display heading + right-aligned author/date + heavy whitespace |


Not on the table -> first principles: **serif carries authority, sans carries utility, warm gray carries rhythm, ink-blue carries focus**.