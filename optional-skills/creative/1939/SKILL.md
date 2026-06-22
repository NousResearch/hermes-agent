---
name: 1939
description: "Perceptual color engine: 525 curated OKLCH palettes with 8 semantic roles, 10-tint scales, and WCAG contrast ratios. Apply any theme to web, PPT, Word, or charts."
version: 1.0.0
author: "0xCuttlefish (Co-Created with Hermes Agent)"
license: CC0
dependencies: []
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [color, oklch, palette, theme, design-system, brand, contrast, wcag, accessibility, chart-colors, 6529, memes]
    category: creative
    related_skills: [concept-diagrams, pixel-art, excalidraw]
    requires_toolsets: [terminal]
---

# 1939 — Perceptual Color Engine

Named after The Wizard of Oz (1939) — the first color film to define a
coherent visual palette. 525 curated color palettes with 8 semantic roles
each, derived from iconic images using OKLCH perceptual color space
(perceptually uniform — same numeric change = same perceived change).
Each palette gives an agent everything needed to theme any document,
website, or presentation — no color theory knowledge required.

## Scope

**Best suited for:**
- Applying a coherent color system to any document, presentation, or website
- Getting WCAG-compliant contrast ratios for accessibility
- Chart and data visualization color assignment
- Dark/light mode color mapping from a single palette

**Look elsewhere first for:**
- Generating images or visual art (use pixel-art, concept-diagrams, or excalidraw)
- Creating SVG diagrams (use concept-diagrams)
- CSS framework theming systems like Tailwind (this provides color values, not a framework)

1939 provides the *color system* — it pairs with output-generating skills,
it doesn't replace them.

## When to Use

- User wants to apply a color palette or theme to a document, slide deck, or website
- User asks for brand colors, a design system, or a color scheme
- User needs WCAG-compliant contrast ratios for accessibility
- User wants chart colors that are perceptually distinct
- User needs dark/light mode color mapping
- User mentions a specific 1939 palette by name ("Hugo's Mom", "Wizard of Oz", etc.)
- User wants to search palettes by mood, use case, or character
- Another skill's output (diagrams, dashboards, web pages) needs a coherent color system

## Prerequisites

No external dependencies. The skill works fully offline with local JSON data.

Optional: for richer natural-language search, install the MCP server (see
`server/README.md`). Requires `pip install mcp fastmcp`.

## How to Run

The skill is data-only — no executable to run. The agent reads palette JSON
files and applies the 8-role mapping to the user's target format:

1. Find a palette (browse `palettes/flagship/` or search `palettes/memes/index.json`)
2. Load the JSON with `json.load()`
3. Map the 8 roles to the target format (see Applying to Documents below)
4. For richer natural-language search, optionally start the MCP server:
   ```bash
   cd "$SKILL_DIR/server" && pip install -r requirements.txt && python3 mcp_1939_server.py
   ```

## Quick Reference

```bash
SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/1939/SKILL.md' 2>/dev/null | head -1)")

# List all flagship palettes
ls "$SKILL_DIR/palettes/flagship/"

# Get a specific palette as JSON
cat "$SKILL_DIR/palettes/flagship/hugos-mom-1974.brand.json" | python3 -m json.tool

# Search the memes index by name or season
python3 -c "
import json
cards = json.load(open('$SKILL_DIR/palettes/memes/index.json'))
for c in cards:
    if 'wizard' in c.get('name','').lower():
        print(c['name'], c['slug'])
"

# Start the MCP server (optional)
cd "$SKILL_DIR/server" && pip install -r requirements.txt && python3 mcp_1939_server.py
```

## The 8 Semantic Roles

Every 1939 palette assigns colors to 8 roles with specific semantic meaning:

| Role | Use For | Example |
|------|---------|---------|
| **Background** | Page backdrop, dark base | Slide bg, dark mode page bg |
| **Canvas** | Readable surface | Card bg, light mode bg, content areas |
| **Text** | Body copy | Paragraphs, descriptions, metadata |
| **Highlight** | Primary accent | Headings, CTAs, hero text, emphasis |
| **Support** | Secondary accent | Links, hover states, heading 2 |
| **Chart1** | Primary data series | First bar/line color |
| **Chart2** | Secondary data series | Comparison series |
| **Muted** | Subtle elements | Borders, disabled states, tertiary text |

Each role has:
- A **center color** (hex) — the 500-level anchor
- **10 perceptual tints** (index 0=lightest, 4=center, 9=darkest)
- A **legend_text** color readable on the center swatch
- A **curve type**: `dark` (compressed), `surface` (compressed), or `standard`

## Applying to Documents

### PowerPoint / Google Slides
```
Slide background:  Background center
Title text:       Highlight center
Body text:        Text center
Accent bar:       Support center
Chart series 1:   Chart1 center
Chart series 2:  Chart2 center
```

### Word / Google Docs
```
Heading 1:        Highlight center
Heading 2:        Support center
Body text:        Text center
Page background:  Background (dark) or Canvas (light)
Table header bg:  Background tint 700
Table header text: Highlight tint 100
Links:            Support center
```

### Web (CSS)
See `references/applying-themes/web-css-custom-properties.md` for the full
`loadTheme()` pattern with `color-mix()`, dark/light polarity, and all CSS
custom properties.

### Dark vs Light Mode
Dark mode swaps Background and Canvas: Background (dark) becomes the page bg,
Canvas (light) becomes the content surface. In light mode, Canvas is the bg
and Background provides text colors.

### Charts
See `references/applying-themes/chart-color-assignment.md` for multi-series
rules. Chart1/Chart2 are for DATA only — never use them for body text or headings.

## Procedure

### 1. Find a palette

Browse `palettes/flagship/` for 29 brand-ready JSON files, or search the
memes index (`palettes/memes/index.json`) for 496 card-derived palettes.

```python
import json, os

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))  # or find via skill_view

# List flagship palettes
flagship_dir = os.path.join(SKILL_DIR, "palettes", "flagship")
for f in sorted(os.listdir(flagship_dir)):
    if f.endswith(".brand.json"):
        brand = json.load(open(os.path.join(flagship_dir, f)))
        print(f"{brand['name']:30s}  PV={brand.get('perceptual_volume', '?')}")
```

### 2. Load the palette JSON

```python
brand = json.load(open("palettes/flagship/wizard-of-oz-1939.brand.json"))

# Access roles
roles = brand["roles"]
print(f"Background: {roles['Background']['hex']}")
print(f"Highlight:  {roles['Highlight']['hex']}")
print(f"Tints:     {roles['Highlight']['tints']}")  # 10 hex values

# Access contrast ratios
print(f"Text on BG: {brand['contrast']['text_on_background']}:1")
```

### 3. Apply to your target format

Follow the mapping tables above for PowerPoint, Word, or Web. For full
code examples (python-pptx, python-docx, CSS loadTheme), see the reference
files under `references/applying-themes/`.

## Collections

### 1939 Flagship (29 themes)

Curated from iconic films, photographs, and cultural touchstones. Each has
full provenance, rights status, and artist credit. Themes include:

- Wizard of Oz (1939) — the namesake palette
- Hugo's Mom (1974), Lawrence of Arabia (1962), Thriller (1983)
- Sgt. Pepper's Lonely Hearts Club Band (1967), Star Gate (1968)
- Persistence of Memory (1931), In the Mood for Love (2000)
- Chanel No. 5, Soup (1962), Gutenberg, Sobol, and more

### 6529 Memes (496 cards)

Per-card palettes from The Memes by 6529 NFT collection (an Ethereum NFT
art project by 6529). Organized by season (1–15). Each card includes:

- `center_colors`: 8 role hex values (center swatches only, no tints)
- `community`: Engagement metrics — `hodl_rate` (% of holders who haven't
  sold), `tdh` (Total Diamond Hands score), `tdh_rank` (rank among 496 cards)
- `contrast`: 2 ratios (text_on_background, highlight_on_background)
- `pv`: Perceptual volume score

Note: Memes cards have center colors only — for full 10-tint scales and
7 contrast ratios, use the flagship palettes or the API.
Search via `palettes/memes/index.json` or the API.

## Optional: REST API

The API at `https://1939.cuttle.af` is a public service maintained by the
1939 author. It is NOT required for this skill — all flagship data ships
locally as JSON. The API provides enriched data for the 496 meme cards
(full tints, contrast ratios, spectra) that would be too large to bundle.

For enriched card data (full spectrum, tints, contrast for 6529 memes):

```
GET https://1939.cuttle.af/api/themes              → all 29 themes
GET https://1939.cuttle.af/api/themes/{slug}       → single theme
GET https://1939.cuttle.af/api/cards?season=1       → cards from season 1
GET https://1939.cuttle.af/api/cards/{slug}         → single card
GET https://1939.cuttle.af/api/collections          → collection info
```

Query params: `?detail=full`, `?collection=6529-memes`, `?season=N`, `?sort=pv`, `?limit=50`

The API is optional. All flagship data ships locally as JSON.

## Optional: MCP Server

The `server/` directory contains a FastMCP server that exposes palette data
as agent tools with 6-layer fuzzy matching (exact → STT alias → token subset
→ metaphone phonetic → Levenshtein → difflib):

- `palette_lookup(name)` — Find a palette by name or slug (fuzzy match)
- `palette_search(query, limit)` — Search by mood, color, use case, or character
- `palette_recommend(use_case, mood, limit)` — Get recommendations for a use case

See `server/README.md` for setup. The server reads data from `../palettes/` by
default; set `NINETEEN_DATA_DIR` to override.

## Key Facts

- **Color space:** OKLCH (perceptually uniform — same numeric change = same perceived change)
- **Tint system:** 10 perceptual levels per role, delta-based from center
- **Contrast:** 7 WCAG 2.1 ratios per palette (text_on_background, highlight_on_background, etc.)
- **License:** CC0 — public domain, no attribution required
- **Pipeline:** sRGB → Linear RGB → XYZ → OKLab → OKLCH → Clustering → Role Assignment → Tint Expansion

## Reference Files

| File | What It Covers |
|------|---------------|
| `references/brand/role-definitions.md` | Complete role semantics, dark/light mode mapping, tint index reference |
| `references/brand/tint-scale-system.md` | How the 10-tint scale works, curves, deltas, practical usage |
| `references/brand/contrast-ratios.md` | The 7 contrast pairs, thresholds, when to use which combination |
| `references/color-science/oklch-primer.md` | Why OKLCH, how it differs from HSL, perceptual volume, hex↔OKLCH conversion |
| `references/color-science/pipeline-equations.md` | Full derivation pipeline — sRGB→OKLab→OKLCH→clustering→roles→tints, with ELI5 |
| `references/color-science/perceptual-volume.md` | What PV measures, ranges, how to use it for palette selection |
| `references/applying-themes/web-css-custom-properties.md` | Full loadTheme() pattern with color-mix() and dark/light polarity |
| `references/applying-themes/chart-color-assignment.md` | Chart1/Chart2 rules, multi-series, data viz guidelines |
| `references/applying-themes/pptx-theme.md` | PowerPoint theme slot mapping, slide layout guide, python-pptx example |
| `references/applying-themes/word-docx-theme.md` | Word document element mapping, dark/light tables, python-docx example |
| `references/api/api-endpoints.md` | REST API docs, query params, response format |
| `references/api/data-schema.md` | Field descriptions for themes and cards |

## Pitfalls

- **Center swatch is index 4 (0-based), not index 5.** The 5th element (0-indexed 4) is the 500-level center color.
- **`community` not `market`.** Engagement data uses `community` with `hodl_rate`, `tdh`, `tdh_rank`. This is engagement data only — community conviction and participation.
- **Hex colors may lack `#` prefix** in raw JSON from the API. The brand-ready files always include `#`. Normalize at consumption: `hex = '#' + hex if not hex.startswith('#') else hex`.
- **`spectrum` is a plain array** of hex colors, not `{count, hexes}`. The API normalizes this; raw 6529 JSON files may still use the grouped format.
- **Dark mode swaps Background and Canvas.** Background (dark) becomes the page bg. Canvas (light) becomes the content surface. In light mode, Canvas (light) is the bg and Background (dark) provides text colors.
- **Chart1/Chart2 are for DATA only.** Never use them for body text or headings.

## Verification

```bash
# Verify all 29 flagship JSONs parse correctly
python3 -c "
import json, os
d = 'palettes/flagship'
count = 0
for f in sorted(os.listdir(d)):
    if f.endswith('.brand.json'):
        brand = json.load(open(os.path.join(d, f)))
        assert len(brand['roles']) == 8, f'{f}: expected 8 roles'
        for role_name, role in brand['roles'].items():
            assert len(role['tints']) == 10, f'{f} {role_name}: expected 10 tints'
            assert role['hex'].startswith('#'), f'{f} {role_name}: hex missing #'
        assert len(brand['contrast']) == 7, f'{f}: expected 7 contrast ratios'
        count += 1
print(f'All {count} flagship palettes validated: 8 roles, 10 tints each, 7 contrast ratios')
"
```

## See Also

- **concept-diagrams** — SVG diagrams that can be themed with 1939 palettes
- **pixel-art** — Image conversion whose output can use 1939 chart colors
- **excalidraw** — Hand-drawn sketches that benefit from perceptual color assignment