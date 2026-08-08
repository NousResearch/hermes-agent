# 8-Role Semantic Color System

The 1939 perceptual color engine derives 8 semantic roles from any image. Each role has a name, a purpose, and a clear mapping to document elements.

## The 8 Roles

| Role | Purpose | Dark Mode Use | Light Mode Use |
|------|---------|---------------|----------------|
| **Background** | Page backdrop | Page background, slide backgrounds, card surfaces | Inverted — becomes text/heading color in light mode |
| **Canvas** | Readable surface | Light foreground surfaces, content areas | Page background, card surfaces in light mode |
| **Text** | Body copy | Body text, descriptions, metadata | Same — body text on light backgrounds |
| **Highlight** | Primary accent | Headings, CTAs, hero text, emphasis | Headings, CTAs, links |
| **Support** | Secondary accent | Links, subtle emphasis, hover states | Secondary headings, links, hover |
| **Chart1** | Primary data | First chart series, main data viz color | Same |
| **Chart2** | Secondary data | Comparison series, secondary chart color | Same |
| **Muted** | Subtle elements | Borders, disabled states, tertiary text | Borders, dividers, tertiary text |

## Key Principles

1. **Every role has 10 tints** forming a perceptual scale from light (index 0) to dark (index 9). Index 4 is the "center" or "500-level" — the true role color. Index 0 is roughly the 50-level (lightest), index 9 is roughly the 900-level (darkest).

2. **The center hex is the role's identity.** When someone says "use Hugo's Mom's Highlight," they mean the center hex `#CF6F66`. The tints give you variations for hover states, gradients, and hierarchy.

3. **Contrast ratios are between role pairs, not individual colors.** The 7 contrast pairs tell you whether specific combinations are readable:
   - `text_on_background`: Can you read body text on the background? (WCAG AA requires ≥ 4.5:1 for normal text)
   - `highlight_on_background`: Are headings readable? (≥ 3:1 for large text)
   - `highlight_on_canvas`: Are headings readable on the surface? 
   - `canvas_on_background`: Is the surface distinguishable from the background?
   - `support_on_background`: Is the secondary accent visible?
   - `chart1_on_background` / `chart2_on_background`: Are chart colors visible on the background?

4. **Background and Canvas swap for light/dark mode.** In dark mode, Background is the dark page and Canvas is the light surface. In light mode, Canvas becomes the background and Background provides the dark text colors.

5. **Chart1 and Chart2 are for DATA elements only.** Charts, graphs, data viz, palette swatches. Never use them for body text or headings — that's what Text and Highlight are for.

6. **Muted is the utility role.** Borders, disabled states, placeholders, tertiary labels. It's intentionally low-contrast against the background to stay out of the way.

## Tint Index Reference

```
Index 0  →  50-level (lightest)
Index 1  →  100-level
Index 2  →  200-level
Index 3  →  300-level
Index 4  →  500-level (CENTER — the role's true color)
Index 5  →  600-level
Index 6  →  700-level
Index 7  →  800-level
Index 8  →  900-level
Index 9  →  950-level (darkest)
```

The index-to-level mapping isn't perfectly linear because OKLCH tints follow perceptual curves. "Dark" roles (like Background) have most of their variation in the light tints. "Light" roles (like Canvas) have most variation in the dark tints. The `curve` field on each role tells you which perceptual curve was used:
- `dark`: the role is dark-valued, lightest tints first
- `light` or `surface`: the role is light-valued, darkest tints last
- `standard` or `saturated`: the role has mid-range lightness with chroma

## Applying to Documents

See the SKILL.md "Applying to Documents" section for the quick mapping tables,
and the `applying-themes/` reference files for full code examples:

- `applying-themes/pptx-theme.md` — PowerPoint 12-slot XML mapping + python-pptx
- `applying-themes/word-docx-theme.md` — Word slot mapping + python-docx
- `applying-themes/web-css-custom-properties.md` — Full loadTheme() with color-mix()
- `applying-themes/chart-color-assignment.md` — Multi-series chart rules

## Legend Text

Each role includes a `legend_text` hex color — this is the color that's readable on that role's center swatch. Use it for:
- Labels on chart bars colored with this role
- Text inside buttons using this role as background
- Caption text on palette swatch strips