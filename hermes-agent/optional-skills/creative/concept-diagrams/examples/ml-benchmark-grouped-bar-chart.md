# ML Benchmark Grouped Bar Chart with Dual Axis

A quantitative data visualization comparing LLM inference speed across quantization levels with dual Y-axes, threshold markers, and an inset accuracy table.

## Key Patterns Used

- **Grouped bars**: Min/max range pairs per category using semantic color pairs (lighter=min, darker=max)
- **Dual Y-axis**: Left axis for primary metric (tok/s), right axis for secondary metric (VRAM GB)
- **Overlay line graph**: `<polyline>` with labeled dots showing VRAM usage across categories
- **Threshold marker**: Dashed red horizontal line indicating hardware limit (24 GB GPU)
- **Zone annotations**: Subtle text labels above/below threshold for context
- **Inset data table**: Alternating row fills below chart with quantitative accuracy data
- **Semantic color coding**: Each quantization level gets its own color from the skill palette (red=OOM, amber=slow, teal=sweet spot, blue=fast)

## Diagram Type

This is a **quantitative data chart** with:
- **Grouped vertical bars**: Range bars showing min–max performance per category
- **Secondary axis line**: VRAM usage overlaid as a connected scatter plot
- **Threshold annotation**: Hardware constraint line
- **Inset table**: Supporting accuracy metrics

## Chart Layout Formula

```
Chart area:  x=90–590, y=70–410 (500px wide, 340px tall)
Left Y-axis: Primary metric (tok/s)
             y = 410 − (val / max_val) × 340
Right Y-axis: Secondary metric (VRAM GB)
              Same formula, different scale labels
Groups:       Divide width by number of categories
Bars:         Each group → min bar (34px) + 8px gap + max bar (34px)
Line overlay: <polyline> connecting data points across group centers
Threshold:    Horizontal dashed line at critical value
Table:        Below chart, alternating row fills
```

## Data Mapped

| Quantization | Model Size | Speed (tok/s) | VRAM (GB) | MMLU Pro | Status |
|-------------|-----------|---------------|-----------|----------|--------|
| FP16 | 62 GB | 0.5–2 | 62 | 75.2 | OOM / unusable |
| Q8_0 | 32 GB | 3–5 | 32 | 75.0 | Partial offload |
| Q4_K_M | 16.8 GB | 8–12 | 16.8 | 73.1 | Fits in VRAM ✓ |
| IQ3_M | 12 GB | 12–15 | 12 | 70.5 | Full GPU speed |

## Bar CSS Classes

```css
/* Light mode */
.bar-fp16-min { fill: #FCEBEB; stroke: #A32D2D; stroke-width: 0.75; }
.bar-fp16-max { fill: #F7C1C1; stroke: #A32D2D; stroke-width: 0.75; }
.bar-q8-min   { fill: #FAEEDA; stroke: #854F0B; stroke-width: 0.75; }
.bar-q8-max   { fill: #FAC775; stroke: #854F0B; stroke-width: 0.75; }
.bar-q4-min   { fill: #E1F5EE; stroke: #0F6E56; stroke-width: 0.75; }
.bar-q4-max   { fill: #9FE1CB; stroke: #0F6E56; stroke-width: 0.75; }
.bar-iq3-min  { fill: #E6F1FB; stroke: #185FA5; stroke-width: 0.75; }
.bar-iq3-max  { fill: #B5D4F4; stroke: #185FA5; stroke-width: 0.75; }

/* Dark mode */
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
```

## Overlay Line CSS

```css
.vram-line { stroke: #534AB7; stroke-width: 2.5; fill: none; }
.vram-dot  { fill: #534AB7; stroke: var(--bg-primary); stroke-width: 2; }
.vram-label { font-family: system-ui, sans-serif; font-size: 10px; fill: #534AB7; font-weight: 500; }
```

## Threshold CSS

```css
.threshold { stroke: #A32D2D; stroke-width: 1; stroke-dasharray: 6 3; fill: none; }
.threshold-label { font-family: system-ui, sans-serif; font-size: 10px; fill: #A32D2D; font-weight: 500; }
```

## Table CSS

```css
.tbl-header { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.5; }
.tbl-row    { fill: transparent; stroke: var(--border); stroke-width: 0.25; }
.tbl-alt    { fill: var(--bg-secondary); stroke: var(--border); stroke-width: 0.25; }
```

## Layout Notes

- **ViewBox**: 680×660 (portrait, chart + legend + table)
- **Chart area**: y=70–410, x=90–590
- **Legend row**: y=458–470
- **Inset table**: y=490–620
- **Bar width**: 34px each, 8px gap between min/max pair
- **Group spacing**: 125px center-to-center
- **Dot halo**: White circle (r=6) behind colored dot (r=5) for legibility over bars/grid

## When to Use This Pattern

Use this diagram style for:
- Model benchmark comparisons across quantization levels
- Performance vs. resource usage tradeoff analysis
- Any multi-metric comparison with a hardware/software constraint
- GPU/TPU/accelerator benchmarking dashboards
- Accuracy vs. speed Pareto frontiers
- Hardware requirement sizing charts
