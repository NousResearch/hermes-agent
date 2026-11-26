# baoyu-infographic — Infographic Generator

> Full skill at `/home/<username>/.hermes/skills/.archive/creative/baoyu-infographic/`

## Summary

Create infographics with two freely combinable dimensions: layout (information structure) × style (visual aesthetics). Supports 21 layouts and 21 styles, with keyword shortcuts for common use cases.

## When to Use

Trigger when user asks for an infographic, visual summary, information graphic, or uses terms like "信息图", "可视化", or "高密度信息大图".

## Key Options

| Option | Values | Default |
|--------|--------|---------|
| Layout | 21 options (see below) | bento-grid |
| Style | 21 options (see below) | craft-handmade |
| Aspect | Named: landscape (16:9), portrait (9:16), square (1:1). Custom: any W:H | landscape |
| Language | en, zh, ja, etc. | — |

## Layout Gallery (21)

`linear-progression` | `binary-comparison` | `comparison-matrix` | `hierarchical-layers` | `tree-branching` | `hub-spoke` | `structural-breakdown` | `bento-grid` | `iceberg` | `bridge` | `funnel` | `isometric-map` | `dashboard` | `periodic-table` | `comic-strip` | `story-mountain` | `jigsaw` | `venn-diagram` | `winding-roadmap` | `circular-flow` | `dense-modules`

See `references/layouts/<layout>.md` for full definitions.

## Style Gallery (21)

`craft-handmade` | `claymation` | `kawaii` | `storybook-watercolor` | `chalkboard` | `cyberpunk-neon` | `bold-graphic` | `aged-academia` | `corporate-memphis` | `technical-schematic` | `origami` | `pixel-art` | `ui-wireframe` | `subway-map` | `ikea-manual` | `knolling` | `lego-brick` | `pop-laboratory` | `morandi-journal` | `retro-pop-grid` | `hand-drawn-edu`

See `references/styles/<style>.md` for full definitions.

## Keyword Shortcuts

| User Keyword | Layout | Recommended Styles | Default Aspect |
|-------------|--------|--------------------|----------------|
| 高密度信息大图 / high-density-info | `dense-modules` | `morandi-journal`, `pop-laboratory`, `retro-pop-grid` | portrait |
| 信息图 / infographic | `bento-grid` | `craft-handmade` | landscape |

## Recommended Combinations

| Content Type | Layout + Style |
|--------------|----------------|
| Timeline/History | `linear-progression` + `craft-handmade` |
| Step-by-step | `linear-progression` + `ikea-manual` |
| A vs B | `binary-comparison` + `corporate-memphis` |
| Hierarchy | `hierarchical-layers` + `craft-handmade` |
| Overlap | `venn-diagram` + `craft-handmade` |
| Conversion | `funnel` + `corporate-memphis` |
| Cycles | `circular-flow` + `craft-handmade` |
| Technical | `structural-breakdown` + `technical-schematic` |
| Metrics | `dashboard` + `corporate-memphis` |
| Educational | `bento-grid` + `chalkboard` |
| Journey | `winding-roadmap` + `storybook-watercolor` |
| Categories | `periodic-table` + `bold-graphic` |
| Product Guide | `dense-modules` + `morandi-journal` |
| Technical Guide | `dense-modules` + `pop-laboratory` |
| Trendy Guide | `dense-modules` + `retro-pop-grid` |
| Educational Diagram | `hub-spoke` + `hand-drawn-edu` |
| Process Tutorial | `linear-progression` + `hand-drawn-edu` |

## Workflow (7 steps)

1. **Step 1**: Analyze content → `source.md`, `analysis.md`
2. **Step 2**: Generate structured content → `structured-content.md`
3. **Step 3**: Recommend 3-5 layout×style combos (check keyword shortcuts first)
4. **Step 4**: Confirm options via `clarify` (combination → aspect → language if needed)
5. **Step 5**: Generate prompt → `prompts/infographic.md` (load layout + style refs + base prompt)
6. **Step 6**: Generate image — `image_generate` → download via `curl -o /absolute/path.png`
7. **Step 7**: Output summary

## Critical Notes

- **`image_generate` is prompt-only**: Always download the returned URL to an **absolute path**
- **Data integrity is paramount**: Never summarize, paraphrase, or alter source statistics ("73% increase" must stay "73% increase")
- **Custom aspect ratios**: Map to nearest named option (`landscape`/`portrait`/`square`)
- **Strip secrets**: Scan source content for API keys, tokens, credentials before writing output
- **Style consistency**: Apply the selected style definition consistently across the entire infographic

## Output Structure

```
infographic/{topic-slug}/
├── source-{slug}.{ext}
├── analysis.md
├── structured-content.md
├── prompts/infographic.md
└── infographic.png
```

## Pitfalls

1. Data integrity: never summarize/paraphrase source statistics
2. Strip secrets from all outputs
3. One message per section — don't overload sections
4. Style consistency across the entire infographic
5. Custom aspect ratios map to nearest named option

## Reference Files (archived)

- `references/analysis-framework.md` — Analysis methodology
- `references/structured-content-template.md` — Content format
- `references/base-prompt.md` — Prompt template
- `references/layouts/<layout>.md` — 21 layout definitions
- `references/styles/<style>.md` — 21 style definitions
