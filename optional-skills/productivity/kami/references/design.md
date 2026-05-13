# Design System

## Principles

kami's aesthetic compresses into one sentence: **warm parchment canvas, ink-blue accent, serif carries hierarchy, avoid cool grays and hard shadows**.

This is not a UI framework. It is a constraint system for print, designed to keep pages stable, clear, and readable.

**The ten invariants** (each has a real cost, think before overriding):

1. Page background parchment `#f5f4ed`, never pure white
2. Single accent: ink-blue `#1B365D`, no second chromatic color
3. All grays warm-toned (yellow-brown undertone), no cool blue-grays
4. English: serif for everything (headlines and body). Chinese: serif headlines, sans body. Sans only for UI elements (labels, eyebrows, meta) in both
5. Serif weight locked at 500, no bold
6. Line-heights: tight headlines 1.1-1.3, dense body 1.4-1.45, reading body 1.5-1.55
7. Letter-spacing: Chinese body 0.3pt for comfortable reading; English body 0; tracking only for short labels and overlines
8. Tag backgrounds must be solid hex, never rgba (WeasyPrint renders a double rectangle)
9. Depth via ring shadow or whisper shadow, never hard drop shadows
10. **No italic anywhere**. No `font-style: italic` in any template or demo. No italic @font-face declarations needed

This system is a fusion of Anthropic's visual language and real Chinese / English resume iteration. Details below.

---

## 1. Color

**Single accent, warm neutrals only, zero cool tones** - this is the core.

### Brand

```css
--brand:       #1B365D;   /* Ink Blue - the only chromatic color. CTAs, accents, section-title left bar. */
--brand-light: #2D5A8A;   /* Ink Light - brighter variant, for links on dark surfaces. */
```

**Rule**: ink-blue covers ≤ **5% of document surface area**. More than that is ornament, not restraint.

### Surface

```css
--parchment:    #f5f4ed;   /* Page background - warm cream, the emotional foundation */
--ivory:        #faf9f5;   /* Card / lifted container - brighter than parchment */
--warm-sand:    #e8e6dc;   /* Button default / interactive surface */
--dark-surface: #30302e;   /* Dark-theme container - warm charcoal */
--deep-dark:    #141413;   /* Dark-theme page background - not pure black, slight olive undertone */
```

**Never**: `#ffffff` pure white as page background. `#f8f9fa` / `#f3f4f6` or any cool-gray surface.

### Text

```css
--near-black:  #141413;   /* Primary text - deepest but not pure black, warm olive undertone */
--dark-warm:   #3d3d3a;   /* Secondary text, table headers, links */
--olive:       #504e49;   /* Subtext - descriptions, captions. JA override: #4d4c48 (YuMincho thin strokes need darker text) */
--stone:       #6b6a64;   /* Tertiary - dates, metadata */
```

Four levels: near-black (primary) > dark-warm (secondary) > olive (subtext) > stone (tertiary). No fifth level needed.

**Mnemonic**: every gray has a **yellow-brown undertone**. In `rgb()`, warm gray is R ≈ G > B (or R > G > B with small gaps). Cool gray is R < G < B or R = G = B (neutral).

### Border

```css
--border:      #e8e6dc;   /* Primary border - section dividers, table headers, card borders */
--border-soft: #e5e3d8;   /* Secondary border - row separators, subtle dividers */
```

### Translucent -> Solid conversion (TAGS MUST BE SOLID)

**Why**: WeasyPrint's alpha compositing for padding vs glyph areas produces a visible double rectangle on zoom. See `production.md` Part 4 Pitfall #1.

Ink Blue `#1B365D` over parchment `#f5f4ed`:

| rgba alpha | Solid hex |
|---|---|
| 0.08 | `#EEF2F7` |
| 0.14 | `#E4ECF5` |
| **0.18** | **`#E4ECF5`** ← default tag |
| 0.22 | `#D0DCE9` |
| 0.30 | `#D6E1EE` |

---

## 2. Typography

### Stacks

```css
/* Single serif per page. --sans always equals var(--serif). */

/* English */
font-family: Charter, Georgia, Palatino,
             "Times New Roman", serif;

/* Chinese */
font-family: "TsangerJinKai02",
             "Source Han Serif SC", "Noto Serif CJK SC",
             "Songti SC", "STSong",
             Georgia, serif;

/* Japanese */
font-family: "YuMincho", "Yu Mincho",
             "Hiragino Mincho ProN",
             "Noto Serif CJK JP", "Source Han Serif JP",
             "TsangerJinKai02",
             Georgia, serif;

/* Mono, with CJK fallback for comments and labels */
font-family: "JetBrains Mono", "SF Mono", "Fira Code",
             Consolas, Monaco,
             "TsangerJinKai02", "Source Han Serif SC",
             monospace;
```

Any font-family that may render Chinese or Japanese must include a CJK fallback, including `@page` footer text, `pre`, `code`, and SVG labels. A pure mono stack can render missing glyph boxes in WeasyPrint.

### Size scale (pt for print A4, px for screen)

**Print:**

| Role | Size | Weight | Line-height | Use |
|---|---|---|---|---|
| Display | 36pt | 500 | 1.10 | Cover title, one-pager hero |
| H1 Section | 22pt | 500 | 1.20 | Chapter titles |
| H2 | 16pt | 500 | 1.25 | Subsection |
| H3 | 13pt | 500 | 1.30 | Item titles |
| Body Lead | 11pt | 400 | 1.55 | Intro paragraphs |
| Body | 10pt | 400 | 1.55 | Reading body |
| Body Dense | 9.2pt | 400 | 1.42 | Dense body (resume, one-pager) |
| Caption | 9pt | 400 | 1.45 | Notes, figure captions |
| Label | 9pt | 600 | 1.35 | Small labels, corner tags |
| Tiny | 9pt | 400 | 1.40 | Footer, minor metadata |

**Screen (px)** ≈ pt × 1.33 (9pt ≈ 12px, 18pt ≈ 24px).
**Minimum floor**: web text >= 12px, PDF text >= 9pt.

### Weight

- **Serif body**: 400 (W04 font file)
- **Serif headings**: 500 (W05 font file, real bold, not synthetic)
- **Sans body**: 400 default
- **Sans labels / small titles**: 500 or 600
- **Forbidden**: 900 black, 100 thin

**Design principle**: Serif uses only two weights (400/500), no synthetic bold (600/700), maintaining restrained typography.

- `strong { font-weight: 500 }` in long-doc templates locks bold to W05, preventing browsers from synthesizing 700 on top of W05
- **Web only**: W04 covers weight 400-500 (single `font-weight: 400 500` declaration); W05 is PDF-only because WeasyPrint cannot synthesize bold

### Line-height

Print documents are **tighter** than English web body. English web typically runs 1.6-1.75; in print at pt sizes that feels loose and floats.

| Tier | Value | Use |
|---|---|---|
| Tight headline | 1.10-1.30 | Display, H1, H2 |
| Dense body | 1.40-1.45 | Resume, one-pager, dense information |
| Reading body | 1.50-1.55 | Long-doc chapters, letters |
| Label / caption | 1.30-1.40 | Small labels, multi-line metadata |

**Forbidden**:
- 1.60+ - loose feel, web rhythm, not print
- 1.00-1.05 - lines collide except at extreme display sizes

### Letter-spacing

- Body text: **0**
- Chinese and Japanese body text with TsangerJinKai02: **0.1–0.2pt** to compensate for the font's natural density; section titles and Mincho samples: **0**
- Chinese lede text (14–22pt) with TsangerJinKai02: **0.03–0.06em** to open up large-body paragraphs without breaking density; EN and JA lede: **0** (only TsangerJinKai02 needs density compensation)
- Chinese and Japanese display text (24pt+): **0.2–1pt** optical spacing for visual breathing room at large sizes; scale with font size
- English headings may use subtle optical tightening when needed; keep it localized, never inherited by body copy
- Small labels (< 10pt): +0.2 to +0.5pt for readability
- All-caps overlines: +0.5 to +1pt mandatory
- **Slide-specific**: print tracking x0.5 at slide scale. Eyebrow max 3px (not 8px), display titles -0.5pt. Large type at 40pt+ will look scattered at print tracking values

---

## 3. Spacing

### Base unit: 4pt (4px on screen)

| Tier | Value | Use |
|---|---|---|
| xs | 2-3pt | Inline adjacent elements |
| sm | 4-5pt | Tag padding, dense layout |
| md | 8-10pt | Component interior |
| lg | 16-20pt | Between components / card padding |
| xl | 24-32pt | Section-title margins |
| 2xl | 40-60pt | Between major sections |
| 3xl | 80-120pt | Between chapters (long docs) |

### Page margins (A4)

| Document | Top | Right | Bottom | Left |
|---|---|---|---|---|
| Resume (dense) | 11mm | 13mm | 11mm | 13mm |
| One-Pager | 15mm | 18mm | 15mm | 18mm |
| Long Doc | 20mm | 22mm | 22mm | 22mm |
| Letter | 25mm | 25mm | 25mm | 25mm |
| Portfolio | 12mm | 15mm | 12mm | 15mm |

**Rule**: denser = smaller margins, more formal (letter) = larger margins.

### Slide-scale spacing

Print uses mm/pt; slides (screen) use px. The scale relationships differ:

```css
--slide-pad: 80px;   /* slide four-side padding baseline */
```

**Key rules**:
- Slide padding-top: 72-80px (print is 96-120px; slides are more compact)
- Slide letter-spacing = print value / 2 (8px tracking "falls apart" on screen; halve it)
- Macro scale (font size, padding): multiply print pt values by ~1.6
- Micro scale (letter-spacing, border, radius): multiply by ~0.6

---

## 4. Components

### Cards / Containers

```css
.card {
  background: var(--ivory);
  border: 0.5pt solid var(--border-cream);
  border-radius: 8pt;
  padding: 16pt 20pt;
}

.card-featured {
  border-radius: 16pt;
  box-shadow: 0 4pt 24pt rgba(0,0,0,0.05);   /* whisper shadow */
}
```

Radius scale: 4pt -> 6pt -> 8pt (default) -> 12pt -> 16pt -> 24pt -> 32pt (hero containers).

### Buttons

```css
/* Primary - brand-colored */
.btn-primary {
  background: var(--brand);
  color: var(--ivory);
  padding: 8pt 16pt;
  border-radius: 8pt;
  box-shadow: 0 0 0 1pt var(--brand);   /* ring shadow */
}

/* Secondary - warm-sand */
.btn-secondary {
  background: var(--warm-sand);
  color: var(--dark-warm);
  padding: 8pt 16pt;
  border-radius: 8pt;
  box-shadow: 0 0 0 1pt var(--border);
}
```

### Tags

Three tiers from weak to strong visual weight:

**Lightest solid** (default, most restrained):
```css
.tag {
  background: #EEF2F7;      /* 0.08 solid equivalent */
  color: var(--brand);
  font-size: 9pt;
  font-weight: 600;
  padding: 1pt 5pt;
  border-radius: 2pt;
  letter-spacing: 0.4pt;
  text-transform: uppercase;
}
```

**Standard solid** (when more contrast needed):
```css
.tag {
  background: #E4ECF5;      /* 0.18 solid equivalent */
  color: var(--brand);
  padding: 1pt 6pt;
  border-radius: 4pt;
}
```

**Gradient brush** (only when "hand-painted" feel is required - use sparingly):
```css
.tag {
  background: linear-gradient(to right, #D6E1EE, #E4ECF5 70%, #EEF2F7);
  color: var(--brand);
  padding: 1pt 5pt;
  border-radius: 2pt;
}
```

**Philosophy**: tint depth should be one step lighter than what decoration wants. Prefer pale over saturated. In iteration, "gradient brush" often steals focus - lightest solid wins most of the time.

**Never**: `background: rgba(201, 100, 66, 0.18)` - WeasyPrint double-rectangle bug.

### Lists

```css
ul, ol {
  padding-left: 16pt;
  line-height: 1.55;
}
ul li::marker { color: var(--brand); }
```

Editorial bookish variant - **en-dash instead of bullet**:

```css
ul.dash { list-style: none; padding-left: 0; }
ul.dash li { padding-left: 14pt; }
ul.dash li::before {
  content: "\2013";
  color: var(--brand);
}
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

### Code

```css
.code-block {
  background: var(--ivory);
  border: 0.5pt solid var(--border-cream);
  border-radius: 6pt;
  padding: 10pt 14pt;
  font-family: var(--mono);
  font-size: 9pt;
  line-height: 1.5;
}
```

### Section Title

```css
.section-title {
  font-family: var(--serif);
  font-size: 14pt;
  font-weight: 500;
  color: var(--near-black);
  margin: 24pt 0 10pt 0;
  border-left: 2.5pt solid var(--brand);
  border-radius: 1.5pt;
  padding-left: 8pt;
}
```

### Table (kami-table)

Unified table component across all templates. Base class applies to bare `<table>` or `.kami-table`.

```css
table, .kami-table {
  width: 100%; border-collapse: collapse;
  font-size: 9.5pt; margin: 12pt 0; break-inside: avoid;
}
table th, .kami-table th {
  text-align: left; font-weight: 500; color: var(--dark-warm);
  padding: 6pt 8pt; border-bottom: 1pt solid var(--border);
}
table td, .kami-table td {
  padding: 5pt 8pt; border-bottom: 0.3pt solid var(--border-soft);
  vertical-align: top;
}
```

**Variants** (combine freely on the same element):

| Class | Purpose |
|---|---|
| `.compact` | 8pt font, tighter padding. For data-dense tables in resume/one-pager. |
| `.financial` | Right-align all columns except the first, enable `tabular-nums`. For revenue, pricing, metrics. |
| `.striped` | Alternating `var(--ivory)` background on even rows. Improves scanability for wide tables. |

**Total row**: add `.total` to the final `<tr>` for a bold summary row with a `1pt` brand top border.

```html
<table class="kami-table financial striped">
  <thead><tr><th>Category</th><th>Q1</th><th>Q2</th></tr></thead>
  <tbody>
    <tr><td>Revenue</td><td>$12.4M</td><td>$14.1M</td></tr>
    <tr class="total"><td>Total</td><td>$12.4M</td><td>$14.1M</td></tr>
  </tbody>
</table>
```

### Metric

Key numbers side-by-side (one-pager header, resume top, portfolio cover):

```css
.metrics { display: flex; gap: 24pt; }
.metric  { flex: 1; display: flex; align-items: baseline; gap: 6pt; }
.metric-value {
  font-family: var(--serif);
  font-size: 16pt;
  font-weight: 500;
  color: var(--brand);
  font-variant-numeric: tabular-nums;   /* align digits in columns */
}
.metric-label { font-size: 9pt; color: var(--olive); }
```

### Section Header (`.kami-section-header`)

Lightweight section opener for content slides. Has an eyebrow and a horizontal rule.

```css
.kami-section-header {
  margin-bottom: 36px;
}
.kami-section-header .eyebrow {
  display: flex;
  align-items: center;             /* dot is geometric, center beats baseline */
  gap: 8px;
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--stone);
  margin-bottom: 14px;
}
.kami-section-header .eyebrow::before {
  content: "";
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--brand);
  flex-shrink: 0;
}
.kami-section-header .rule {
  height: 1px;
  background: var(--border-warm);
  margin-bottom: 36px;             /* gap below rule >= 36px (>= 2x the gap above) */
}
.kami-section-header h1 {
  font-family: var(--serif);
  font-size: 38px;
  font-weight: 500;
  line-height: 1.1;
  color: var(--near-black);
}
```

**Spacing rule**: eyebrow to rule: 14px; rule to H1: **≥ 36px** (the gap below must be at least double the gap above, creating a visual anchor).

### Code Card (`.kami-code-card`)

For displaying pseudocode or code snippets in slides. More structured than a plain code block.

```css
.kami-code-card {
  background: var(--ivory);
  border: 1px solid var(--border-cream);
  border-radius: 8px;
  padding: 20px 24px;
  overflow: hidden;
}
.kami-code-card pre {
  font-family: var(--mono);
  font-size: 13px;                 /* 14px for larger slides */
  line-height: 1.55;
  color: var(--near-black);
  margin: 0;
  white-space: pre;
}
/* Syntax colors: existing tokens only, no new colors */
.kami-code-card .k { color: var(--brand); }    /* keyword / string */
.kami-code-card .c { color: var(--stone); }    /* comment */

/* Optional line numbers: 1px left divider */
.kami-code-card.numbered {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0 16px;
}
.kami-code-card .line-nums {
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.55;
  color: var(--stone);
  text-align: right;
  border-right: 1px solid var(--border-soft);
  padding-right: 12px;
  user-select: none;
}
```

**Content philosophy**: use pseudocode style. Comments should outnumber code lines. The reader sees logic, not syntax.

### Glance Grid

Four key-number cells, placed after the TOC or on a chapter-opening page of a long-doc / proposal.

```html
<div class="glance-grid">
  <div class="glance-cell">
    <div class="glance-label">REPORTING PERIOD</div>
    <div class="glance-value">Q1 2026</div>
    <div class="glance-note">Three core themes</div>
  </div>
  <!-- 4 cells total -->
</div>
```

```css
.glance-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14pt;
  margin: 18pt 0;
}
.glance-cell {
  padding: 12pt 0 10pt 14pt;
  border-left: 2pt solid var(--brand);
  border-radius: 1.5pt;
}
.glance-label {
  font-family: var(--mono);
  font-size: 8.5pt;
  color: var(--brand);
  letter-spacing: 1pt;
  text-transform: uppercase;
  font-weight: 500;
}
.glance-value {
  font-size: 18pt;
  font-weight: 500;
  color: var(--near-black);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.5pt;
}
.glance-note {
  font-size: 9pt;
  color: var(--olive);
  line-height: 1.4;
}
```

### Module Block

Proposal A / B / C structure: each module gets a brand-colored letter, a Chinese title, and an uppercase English subtitle.

```html
<div class="module">
  <div class="module-head">
    <div class="module-letter">A</div>
    <div class="module-title">模块标题</div>
    <div class="module-sub">MODULE SUBTITLE</div>
  </div>
  <p>...</p>
  <ul>...</ul>
</div>
```

Visual recipe: letter at 28pt brand, title at 17pt, English subtitle at 10pt mono brand `letter-spacing: 2pt`, head separated from body by a 0.3pt warm-gray hairline (not brand color).

### Module Note (group explanation)

A short note that explains the relationship between two or more modules. Same family as `.callout`, lighter weight, no decorative bar.

```html
<div class="module-note">
  <div class="module-note-label">ABOUT B + C</div>
  <p>B 是上游能力建设，C 是下游验证。两者构成一个最小闭环。</p>
</div>
```

ivory background + 4pt radius + `module-note-label` at 8.5pt brand uppercase mono.

### Position Table

Three-column industry-comparison table whose final row highlights the current project / subject.

```html
<table class="position-table">
  <thead>
    <tr><th>Direction</th><th>Reference project</th><th>Approach</th></tr>
  </thead>
  <tbody>
    <tr><td>...</td><td>...</td><td>...</td></tr>
    <tr class="highlight"><td><strong>本项目</strong></td>...</tr>
  </tbody>
</table>
```

`.highlight` row: ivory fill + brand text. Do not bold the entire row; let the `<strong>` carry the emphasis.

### Pricing Card

Headline-figure price block. Eyebrow + price + short note.

```html
<div class="pricing-card">
  <div class="pricing-eyebrow">PROJECT TERM</div>
  <div class="pricing-price">
    <span class="currency">¥</span>
    <span class="amount">XXX,XXX</span>
    <span class="unit">/ term</span>
  </div>
  <div class="pricing-note">Paid by milestone</div>
</div>
```

Digits: serif 500, 44pt, `tabular-nums`, `letter-spacing: 0.5pt`. Without the letter-spacing, large digits crowd each other and read as too dense.

For the without-price variant (see writing.md "Proposal voice"), drop the `.pricing-price` block and follow the eyebrow with the Value Anchors list below.

### Value Anchors

Replaces pricing line-item breakdowns with a short list of capability anchors. Pairs with Pricing Card or stands alone in the without-price variant.

```html
<ul class="value-anchors">
  <li><strong>能力锚点 A</strong>：一句具体说明能力来源的事实陈述</li>
  <li><strong>能力锚点 B</strong>：一句具体说明锚点稀缺性的事实陈述</li>
  <!-- 4-6 items -->
</ul>
```

```css
.value-anchors {
  list-style: none;
  padding: 0;
  margin: 12pt 0 18pt 0;
}
.value-anchors li {
  position: relative;
  padding: 9pt 0 9pt 18pt;
  border-bottom: 0.3pt solid var(--border-soft);
  line-height: 1.55;
  font-size: 10.5pt;
}
.value-anchors li:last-child { border-bottom: none; }
.value-anchors li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 17pt;
  width: 8pt;
  height: 1.5pt;
  background: var(--brand);
}
.value-anchors li strong {
  color: var(--brand);
  font-weight: 500;
  margin-right: 6pt;
}
```

The 8pt × 1.5pt brand bar (`::before`) replaces the round `<ul>` bullet. A round bullet next to CJK body reads juvenile; the bar reads editorial.

### Decoration density: editorial vs structured

Long-doc / proposal layouts have two acceptable decoration densities. Pick one and stay consistent across the whole document.

| Context | Mode | Pattern |
|---|---|---|
| Data report, white paper, technical brief | **Structured** | Top hairlines (0.6-0.8pt brand) on callouts, glance cells, and pricing blocks. Roughly 5-8 brand lines per page. |
| Proposal, advisory pitch, founder-facing brief | **Editorial** (default) | No decorative lines. Brand color appears only in text (chapter number, `.hl`, `<strong>`, digits, labels). Containers use ivory fill + 4pt radius. |

The editorial mode reads as "content speaks"; the structured mode reads as "structure helps". The wrong mode is the third one: brand lines plus ivory plus radius plus borders, which signals over-packaging. When unsure, default to editorial.

---

## 5. Depth & Shadow

**Core rule**: do not use traditional hard shadows. Depth comes from three sources:

### 1. Ring shadow (border-like)

For **button** hover/focus states.

```css
/* Button default */
box-shadow: 0 0 0 1pt var(--ring-warm);

/* Button hover/active */
box-shadow: 0 0 0 1pt var(--ring-deep);
```

**Do not use for card hover**: ring shadow is a border replacement. Layering it over an existing border creates three-layer visual stacking (border + ring + offset), which feels digital, not paper-like.

### 2. Whisper shadow (barely visible lift)

For **card hover** and **featured card** elevation.

```css
/* Card hover - mimics paper lifting slightly */
.card {
  transition: box-shadow 0.2s;
}
.card:hover {
  box-shadow: 0 4pt 24pt rgba(0, 0, 0, 0.05);
}

/* Featured card default state */
.featured-card {
  box-shadow: 0 4pt 24pt rgba(0, 0, 0, 0.05);
}
```

**Why whisper, not ring**: paper elevation is depth change, not outline change. Whisper shadow is singular, soft, outline-free, matching the paper-like tone.

### 3. Section-level light/dark alternation

Long docs alternate parchment `#f5f4ed` and `#141413` dark sections. This section-level light change creates the strongest contrast.

**Forbidden**: `box-shadow: 0 2px 8px rgba(0,0,0,0.3)` and relatives.

---

## 6. Print & Pagination

### break-inside protection

```css
.card, .metric, .project-item, .quote, .code-block, figure, .callout,
.takeaway, .module, .module-note, .glance-grid, .pricing-card,
table.compact {
  break-inside: avoid;
}

/* Headings should never sit alone at the bottom of a page */
h1, h2, h3 { break-after: avoid; }

/* Widow / orphan minimums for body text */
body { widows: 3; orphans: 3; }
p    { widows: 2; orphans: 2; }
```

CSS alone cannot prevent "the last two lines of a chapter pushed onto a fresh page". For long-doc / proposal output, follow up with a render-time density check (see production.md "Verify & Debug").

### Force break

```css
.page-break { break-before: page; }
```

### Page background extending past margins

```css
@page {
  size: A4;
  margin: 20mm 22mm;
  background: #f5f4ed;   /* extends past margin area, prevents printed white edges */
}
```

---

## 7. Quick decisions

When you're not sure "what should I use":

| Need | Use |
|---|---|
| Big headline | serif 500, size by level, line-height 1.10-1.30 |
| Reading body (EN) | serif 400, 9.5-10pt, line-height 1.55 |
| Reading body (CN) | sans 400, 9.5-10pt, line-height 1.55 |
| Emphasize a number | `color: var(--brand)`, no bold |
| Divide two sections | 2.5pt brand left bar, or 0.5pt warm-gray dotted |
| Quote someone | 2pt brand left border + olive color |
| Show code | ivory background + 0.5pt border + 6pt radius + mono |
| Primary vs secondary button | Primary = brand fill + ivory text; Secondary = warm-sand + dark-warm |
| Highlight one card in a list | `border: 0.5pt solid var(--brand)` or `border-left: 3pt solid var(--brand)` |
| Start a chapter | serif heading + 2.5pt brand left bar |
| Cover page | Display-size heading + right-aligned author/date + heavy whitespace |
| Data card | ivory background + 8pt radius + serif big number + sans small label |

Not on this table -> return to first principles: **serif carries authority, sans carries utility, warm gray carries rhythm, ink-blue carries focus**.

---

## 8. Deck Recipe

Slides in kami use WeasyPrint HTML to PDF as the primary rendering path. The pptx path (`slides.py`) is available as a fallback when the user explicitly requires an editable PPTX file.

### Architecture

**Why WeasyPrint over python-pptx:** pptx output passed through LibreOffice loses CJK font weight, tracking, and glyph spacing. WeasyPrint embeds fonts exactly, giving pixel-level CSS control.

Use `assets/templates/slides-weasy.html` (CN) or `assets/templates/slides-weasy-en.html` (EN) as the starting point.

### Page size

Default `280mm 158mm`. Change in `@page` and `.slide` together.

| Size | `@page` | Use when |
|---|---|---|
| Compact (default) | `280mm 158mm` | Standard density, fits most content |
| Standard | `297mm 167mm` | Slightly more room per slide |
| Wide | `338mm 190mm` | Heavy content, many data points |

### Typography

Global parameters for the slide body:

```css
body {
  font-size: 13pt;
  line-height: 1.65;
  letter-spacing: 0.3pt;   /* CJK: critical for breathing room */
}
```

Heading scale:

| Element | Size | Weight | Notes |
|---|---|---|---|
| `h2` | 24pt | 500 | Page title; `margin-bottom: 14pt` |
| `h3` | 15pt | 500 | Section heading; `color: var(--brand)` |
| `.eyebrow` | 9.5pt | 400 | Mono, `letter-spacing: 2pt`, `color: var(--stone)` |
| `.lead` | 12pt | 400 | Below `h2`; `color: var(--olive)` |

Content element scale:

| Class | Size | Notes |
|---|---|---|
| `.mt` | 16pt | Module title, used with `.ml` |
| `.ml` | 24pt | Large letter prefix in `var(--brand)`, paired with `.mt` |
| `.ms` | 7.5pt mono | Module sub-label; `border-bottom` separator |
| `.mb` | 11pt | Module body description |
| `.mi` | 11pt | Module line item; `padding: 8pt 0` |
| `.mc` | 9.5pt | Delivery rhythm or cadence note; `border-top` |
| `.co` | 11pt bold | Bottom callout; `position: absolute; bottom: 12mm` |

### Layout patterns

**Two-column (`.c2`)**: CSS Grid, `grid-template-columns: 1fr 1fr; gap: 22pt`. Use for side-by-side modules with independent heights.

**2×2 aligned (`.t2x2`)**: HTML `<table>`, not CSS Grid. Grid does not guarantee row alignment across cells; table rows share height naturally.

```html
<table class="t2x2">
  <tr>
    <td> <!-- top-left --> </td>
    <td> <!-- top-right --> </td>
  </tr>
  <tr>
    <td> <!-- bottom-left --> </td>
    <td> <!-- bottom-right --> </td>
  </tr>
</table>
```

**Pinned callout (`.co`)**: `position: absolute; bottom: 12mm; left: 20mm; right: 20mm`. The whitespace above it is intentional, not empty.

### Table styles

```css
table.data td {
  padding: 8pt;
  border-bottom: 0.3pt solid var(--border);
  font-size: 11pt;
}
table.data td:first-child {
  font-weight: 500;
  color: var(--brand);   /* first column: brand blue bold */
}
```

### SVG constraints

- `viewBox` width fixed at `920`; adjust height to content
- `max-height: 105mm` on `svg` element to prevent overflow
- WeasyPrint does not support `fill="url(#gradient)"` or CSS Grid inside SVG
- Draw arrowheads as explicit `<path>` elements; `marker-end` with `orient="auto"` does not rotate in WeasyPrint

### Content rules

| Rule | Detail |
|---|---|
| No section divider slides | Use `.eyebrow` for section numbering instead; saves one slide per section |
| No CJK parentheses | Replace `（...）` with `·` or `,` |
| One line per bullet | Trim until each item fits on one line; never let it wrap |
| Empty space handling | Priority: shrink page size > pin `.co` callout > add content > merge slides |
| Cover | No horizontal rule; title centered `38pt`; subtitle on one line; bottom meta centered |

### Troubleshooting

| Symptom | Fix |
|---|---|
| Content overflows to next page | Add `max-height` or trim content |
| 2×2 columns misaligned | Switch from CSS Grid to `table.t2x2` |
| Large blank at slide bottom | Reduce to `280mm 158mm` or pin `.co` callout |
| CJK text looks tight | Add `letter-spacing: 0.3pt` |

### Core principles

1. `letter-spacing` matters more than `font-size` for CJK density
2. 2×2 layouts use `table`, not grid
3. No section divider slides
4. No white card panels on parchment; use border lines to divide
5. Callout pins to bottom; whitespace above is the design
6. Each bullet fits one line
7. Shrink page first before adding more content

---

## 9. Horizontal Funnel / Progress Bar Pattern

No dedicated `funnel.html` diagram prototype exists. When generating a horizontal bar chart with external percentage labels (conversion funnel, progress tracker, ranked list), use a three-column grid so the label column is fixed-width and never drifts with bar length.

```css
.funnel-row {
  display: grid;
  grid-template-columns: 80pt 1fr 40pt;  /* label | track | external % */
  align-items: center;
  gap: 8pt;
  margin: 4pt 0;
}
.funnel-track {
  position: relative;
  height: 18pt;
  background: var(--border-soft);
  border-radius: 2pt;
}
.funnel-fill {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--brand);
  border-radius: 2pt;
}
.funnel-pct {
  font-variant-numeric: tabular-nums;
  text-align: right;
  color: var(--stone);
  font-size: 9pt;
}
```

Example row (77.8% fill):

```html
<div class="funnel-row">
  <span>Stage Name</span>
  <div class="funnel-track">
    <div class="funnel-fill" style="width:77.8%"></div>
  </div>
  <span class="funnel-pct">77.8%</span>
</div>
```

Key rules:

- Use the three-column grid, not flex. The third column is always 40pt wide; the percentage never moves regardless of bar length.
- Color: single `--brand` fill only. No color gradients or per-row hues. Vary opacity (e.g. `opacity: 0.7`) if a visual ranking is needed, not hue.
- Inline labels inside the bar (count, name) go in an absolutely positioned child inside `.funnel-track`, not in the grid's third column.

---

## 10. Image Aspect Ratios and Cropping

Use this table when placing images in any Kami template. The ratios are defaults, not constraints; adjust by one step if the source image differs significantly.

| Context | Preferred ratio | Notes |
|---|---|---|
| Hero full-bleed (slides / cover) | 16:9 | One image fills the slide; use `object-fit: cover` |
| Main document image (long-doc / portfolio spread) | 4:3 or 16:10 | Standard editorial proportion |
| Side-by-side grid (two images per row) | 3:2 | Even weight, comfortable scanning |
| Magazine inset (text wraps around) | 3:2 or 3:4 | Portrait works when the subject is a person |
| Square thumbnail (icon grid, avatar, logo) | 1:1 | Enforced with `aspect-ratio: 1/1` |
| Slide image grid (26vh fixed-height row) | Fixed height, free width | Grid items share a row height; clip width to fit |

**Cropping rule**: always `object-position: top center`. Crop from the bottom first, then from the sides. Never crop from the top; heads, titles, and focal points live there.

```css
.frame-img,
figure img,
.hero-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: top center;
}
```

Apply this rule to any `<img>` placed inside a fixed-size container. For `object-fit: contain` (slides, logos), `object-position` has no visible effect; omit it.
