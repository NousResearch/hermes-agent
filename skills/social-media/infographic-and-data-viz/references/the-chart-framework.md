# The CHART framework — honest data viz that gets understood + shared

A visualization's job is to make one insight land in three seconds — honestly. Each letter is a gate; most charts
fail at H (a label instead of a takeaway) or A (a distorted scale).

## C — Choose the right chart for the message
Start with the message, then pick the form that makes it impossible to miss. Match chart to the data relationship:
- **Trend over time → line** (consistent intervals, true order, ≤5–6 series).
- **Comparison across categories → bar** (sorted high→low; horizontal for long labels).
- **Composition → stacked bar / donut, sparingly** (avoid many-slice pies).
- **Distribution → histogram; correlation → scatter.**
- **One or two numbers → a big stat or a table, not a chart.**
Default to a **bar** when unsure (versatile, rarely misleads). **One chart, one message.**

## H — Headline the takeaway
The title states the **conclusion**, not the axes: "Checklist users churn 3× less," not "Churn by segment." The
reader should get the point before decoding the chart. Use size/weight/contrast so the eye lands on the takeaway,
then the supporting detail. The headline must match what the data actually shows.

## A — Anchor to honest scales
The integrity gate. **Bar/column charts start at zero** (a truncated axis is the #1 chart crime — the "lie
factor"). No 3D, no skewed perspective, no area/bubble distortion, no dual-axis or cherry-picked-range tricks. Use
the full relevant range; add context (denominator, method) so the read is fair. If the honest data is undramatic,
the **honest chart is the deliverable** — never distort it, never visualize fabricated data.

## R — Reduce to the signal
Apply the **data-ink ratio**: every mark shows data or essential context; strip 3D, drop shadows, gradients,
decorative backgrounds, and unnecessary gridlines. **Direct-label** rather than lean on a busy legend; drop axis
labels the title already implies; one base color + an accent to highlight the key value. Clean, mobile-legible
typography — if a viewer squints at a label, it's not done. Confirm title + annotations + visuals tell **one**
story.

## T — Tag the source + make it accessible
- **Cite the data + date** (good viz starts with good, real, sourced data — pairs with `data-and-original-
  research` / `analytics-and-reporting`); protect sensitive/confidential data.
- **Accessibility:** color-blind-safe palette + redundant encoding (labels/shapes/arrows/patterns), WCAG contrast,
  and **alt text that states the takeaway**, not every value; a data table for complex charts.
- Make it **saveable + citable** (a copy-ready stat, the source visible). Then a design/chart tool renders it;
  **WoopSocial publishes** the finished image.
