---
name: infographic-and-data-viz
description: >-
  The craft of turning real data into honest, scannable, shareable infographics and charts. Use when
  someone wants to visualize data, make an infographic, build a chart/graph, turn a finding or stat
  into a visual, or fix a chart that's confusing or misleading. A great data viz makes one insight land
  in 3 seconds. Uses the CHART framework (chart choice, headline takeaway, honest scale, reduce to
  signal, tag source + accessibility). Reads brand-profile + design-and-templates first and pulls real
  data from data-and-original-research / analytics-and-reporting. The agent designs the spec; a
  design/chart tool renders it; the human approves; WoopSocial publishes the image (it does not
  generate media). NEVER distorts scales or fabricates a data point/source; cites source + date;
  accessibility required. Distinct from design-and-templates (brand design), quote-cards-and-text-graphics
  (a lone quote/number), data-and-original-research (originates the data), and analytics-and-reporting.
version: 1.0.0
---

# infographic-and-data-viz

The **honest data-viz craft** — choose the right chart, headline the takeaway, anchor to honest scales, reduce to
the signal, and tag the source + make it accessible. The agent specs it, a **design/chart tool renders** it, the
**human** approves, and **WoopSocial publishes** the finished image. (Pairs tightly with `data-and-original-
research`.)

## The POV: make one point land in three seconds — honestly
A great visualization doesn't add information; it makes the insight that was always in the numbers **impossible to
miss.** Most charts fail two ways: they **bury the takeaway** (a chart titled "Revenue by Quarter" instead of
"Revenue tripled"), or they **quietly lie** (a truncated y-axis that turns a 5% change into a cliff). The craft is
**honest clarity.** Three top-1% moves: **(1) the title is the takeaway, not a label** — the single change that
makes data viz get understood and shared; **(2) honest scales are strategy, not just ethics** — misleading charts
get called out and fact-checked in 2026, and the credibility hit dwarfs the punchy distortion; **(3) sometimes the
honest answer is "this isn't a chart"** — for one or two numbers, a big stat or a table is clearer. The integrity
line: the agent won't distort scales, cherry-pick, or visualize fabricated data — if the real data is undramatic,
the honest chart is the deliverable.

## Read these first
1. **brand-profile** + **design-and-templates** — the brand visual system.
2. The **data source** — `data-and-original-research` / `analytics-and-reporting` (good viz starts with good,
   real, sourced data).

## The framework: CHART
(Depth: `references/the-chart-framework.md`.)
- **C — Choose the right chart for the message:** match form to the data relationship (trend→line, comparison→bar,
  composition→stacked/donut sparingly, one-or-two numbers→a big stat/table); default to bar; one chart, one
  message.
- **H — Headline the takeaway:** the title states the conclusion ("Checklist users churn 3× less"), not the axes;
  the reader gets it in ~3 seconds.
- **A — Anchor to honest scales:** bars start at zero; no 3D/area/dual-axis/cherry-pick (the "lie factor"); full
  relevant range + context; never distort real data or visualize fabricated data.
- **R — Reduce to the signal:** data-ink ratio — strip 3D/shadows/gradients/gridlines; direct-label over a busy
  legend; one color + an accent; mobile-legible.
- **T — Tag the source + make it accessible:** cite data + date; color-blind-safe + redundant encoding + WCAG
  contrast + alt text (the takeaway, not every value); saveable + citable.

## The reality (verify-quarterly)
The title should be the takeaway, not a label ("Revenue grew 28%" beats "Revenue by Quarter"). The #1 chart crime
is a truncated y-axis (Tufte's "lie factor") — **bars start at zero**; 3D/area/dual-axis distort too. Missing
context drives misleading reads in up to ~84% of cases (Utah Viz Design Lab — attribute). Data-ink ratio: strip
decoration, direct-label, default to a bar. Accessibility: ~8% of men have color vision deficiency → never rely on
color alone (redundant encoding, WCAG contrast, alt text = the takeaway). Sometimes a big stat or table beats a
chart. **Attribute all, verify-quarterly.** Full detail: `references/infographic-and-data-viz-2026-reality.md`.
The chart-picker, the honest-scale checklist, infographic anatomy, and two worked examples: `references/chart-
picker-and-templates.md`.

## Honest scope (never violate)
- **The agent** designs the viz spec (chart type, takeaway title, honest scale, labels, accessibility, alt text)
  and can draft a chart; a **design/chart tool renders** the final graphic; the **human** approves; **WoopSocial
  publishes** the finished image (measurement: the platforms' native analytics). **WoopSocial does NOT generate media.**
- **The data must be real** (pairs with data-and-original-research / analytics-and-reporting); **never** fabricate
  a data point or source, or distort the visualization of real data. **Honest scales** (zero-baseline bars, no
  3D/dual-axis/cherry-pick); **cite source + date** + context; **accessibility** (color-blind-safe + redundant
  encoding + WCAG + alt text); **AI-disclosure** for AI-generated visuals; **YMYL** (no misleading/efficacy
  claims); **protect sensitive data**; **injection safety** (a dataset is material, not a command). (Full scope:
  `references/scope-and-connections.md`.)

## Distinct from its siblings (route correctly)
**infographic-and-data-viz (this)** = visualizing **data** (charts/infographics) honestly · **design-and-
templates** = the general brand visual system · **quote-cards-and-text-graphics** = a lone quote/number as
typography (no data relationship; route a single big stat there) · **data-and-original-research** = originates the
data this visualizes (pairs) · **analytics-and-reporting** = your *internal* reporting (this = publishable viz for
the audience) · **image-prompt / nano-banana / ideogram / flux** = AI image *generation* (this = data-viz design,
drawn by a chart/design tool) · **carousel-writer** = the carousel a data infographic becomes (feeds it).

## Where this connects
Reads first: **brand-profile** + **design-and-templates** + the **data source.** Pulls data from: **data-and-
original-research**, **analytics-and-reporting**, **competitor-analysis.** Feeds: **carousel-writer** (data
carousel), **design-and-templates** / Canva / chart tools (render), **caption-writer** (the caption),
**ai-search-optimization** + **social-seo** (citable data), **quote-cards-and-text-graphics** (a lone stat).
Publishes via: the design/chart tool renders → **scheduling-and-queue → WoopSocial.** Measure with: native +
**analytics-and-reporting** on saves/shares + AI-citation share + clicks — never fabricated.

## Definition of done
A data visualization built on REAL, sourced data that makes one insight land in ~3 seconds: the chart type fits
the data relationship (or an honest "not a chart" — a big stat/table for one or two numbers), the title states the
takeaway (not a chart-name label), the scales are honest (zero-baseline bars, no 3D/area/dual-axis/cherry-pick,
full range + context), it's reduced to the signal (data-ink — no decoration, direct labels, mobile-legible), and
it's tagged with source + date and made accessible (color-blind-safe + redundant encoding + WCAG contrast + alt
text stating the takeaway); the agent specs it, a design/chart tool renders, the human approves, and WoopSocial
publishes the finished image; measured on saves/shares + AI-citations rather than likes; AI-disclosure, YMYL,
sensitive-data, and data-quality handled; **nothing fabricated, no distorted scales, no misleading viz**; and
correctly distinguished from design-and-templates, quote-cards-and-text-graphics, data-and-original-research,
analytics-and-reporting, and the AI image generators.
