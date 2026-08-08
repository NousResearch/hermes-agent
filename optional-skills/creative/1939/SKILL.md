---
name: 1939
description: "Perceptual color engine: 525 OKLCH palettes, 8 roles each."
version: 1.0.0
author: "0xCuttlefish (Co-Created with Hermes Agent)"
license: CC0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [color, oklch, palette, theme, design-system, brand, contrast, wcag, accessibility, chart-colors]
    category: creative
    related_skills: [concept-diagrams, pixel-art, excalidraw]
    requires_toolsets: [terminal]
---

# 1939 Skill

Named after The Wizard of Oz (1939). 525 curated OKLCH color palettes with 8
semantic roles each. Every palette gives an agent everything needed to theme
any document, website, or presentation. No color theory knowledge required.

Pairs with output-generating skills (concept-diagrams, pixel-art, excalidraw)
-- provides the color system, not the output format.

## When to Use

- Apply a color palette or theme to a document, slide deck, or website
- Get WCAG-compliant contrast ratios for accessibility
- Assign perceptually distinct chart colors
- Map dark/light mode colors from a single palette
- Search palettes by name, mood, or use case

## Prerequisites

No external dependencies. Works fully offline with local JSON data.

Optional: for fuzzy natural-language search, install the MCP server (see
`server/README.md`). Requires `pip install -r server/requirements.txt`.

## How to Run

Data-only skill. Read palette JSON files and apply the 8-role mapping:

1. Find a palette: browse `palettes/flagship/` or search `palettes/memes/index.json`
2. Load the JSON with `read_file` or `json.load()`
3. Map the 8 roles to the target format (see Procedure below)
4. Optionally start the MCP server for fuzzy search:
   `cd server && pip install -r requirements.txt && python3 mcp_1939_server.py`

## Quick Reference

Use `search_files` to locate palette files, `read_file` to load them:

```python
import json

# Load a palette
brand = json.load(open("palettes/flagship/wizard-of-oz-1939.brand.json"))
roles = brand["roles"]
print(roles["Background"]["hex"])     # center color
print(roles["Highlight"]["tints"])    # 10 perceptual tints
print(brand["contrast"])              # 7 WCAG ratios
```

Search the memes index:

```python
cards = json.load(open("palettes/memes/index.json"))
matches = [c for c in cards if "wizard" in c.get("name", "").lower()]
```

## Procedure

### 1. Find a palette

29 flagship palettes in `palettes/flagship/` (films, photographs, cultural
icons). 496 card-derived palettes in `palettes/memes/index.json`.

### 2. The 8 semantic roles

| Role | Use For |
|------|---------|
| **Background** | Page backdrop, dark base |
| **Canvas** | Readable surface, card bg, light mode bg |
| **Text** | Body copy, paragraphs, metadata |
| **Highlight** | Headings, CTAs, hero text, emphasis |
| **Support** | Links, hover states, secondary headings |
| **Chart1** | Primary data series |
| **Chart2** | Secondary data series |
| **Muted** | Borders, disabled states, tertiary text |

Each role has: a center hex, 10 perceptual tints (index 4 = center), a
`legend_text` color, and a curve type (`dark`, `surface`, or `standard`).

### 3. Apply to target format

**PowerPoint:** Background = slide bg, Highlight = titles, Text = body,
Support = accents, Chart1/Chart2 = data series.

**Word:** Highlight = Heading 1, Support = Heading 2, Text = body,
Background/Canvas = page bg (dark/light mode).

**Web:** See `references/applying-themes/web-css-custom-properties.md` for
the full `loadTheme()` pattern with `color-mix()` and dark/light polarity.

**Charts:** See `references/applying-themes/chart-color-assignment.md`.
Chart1/Chart2 are for DATA only.

**Dark mode:** Swaps Background and Canvas. Background (dark) becomes page
bg, Canvas (light) becomes content surface.

For full code examples (python-pptx, python-docx, CSS), see
`references/applying-themes/`.

### 4. Collections

**Flagship (29 themes):** Wizard of Oz, Hugo's Mom, Lawrence of Arabia,
Thriller, Sgt. Pepper's, Star Gate, Gutenberg, Sobol, and more. Each has
full provenance and 10-tint scales.

**6529 Memes (496 cards):** Per-card palettes from The Memes by 6529 NFT
collection. Cards have center colors only (no tints). Community metrics
include `hodl_rate`, `tdh`, `tdh_rank`. Search via `palettes/memes/index.json`.

### 5. Optional API and MCP server

A public REST API at `https://1939.cuttle.af` provides enriched card data
(full tints, spectra) too large to bundle. Not required -- all flagship
data ships locally.

The `server/` directory has a FastMCP server with 6-layer fuzzy matching:
`palette_lookup(name)`, `palette_search(query)`, `palette_recommend(use_case)`.
See `server/README.md`.

## Pitfalls

- **Center swatch is tint index 4 (0-based), not 5.** The 500-level anchor.
- **Dark mode swaps Background and Canvas.** Background (dark) = page bg.
- **Chart1/Chart2 are for DATA only.** Never use for body text or headings.
- **Memes cards have center colors only.** Use flagship palettes for full tints.
- **`community` not `market`.** Engagement data uses `hodl_rate`, `tdh`, `tdh_rank`.

## Verification

```python
import json, os

d = "palettes/flagship"
count = 0
for f in sorted(os.listdir(d)):
    if f.endswith(".brand.json"):
        brand = json.load(open(os.path.join(d, f)))
        assert len(brand["roles"]) == 8, f"{f}: expected 8 roles"
        for role_name, role in brand["roles"].items():
            assert len(role["tints"]) == 10, f"{f} {role_name}: expected 10 tints"
            assert role["hex"].startswith("#"), f"{f} {role_name}: hex missing #"
        count += 1
print(f"All {count} flagship palettes validated")
```

See `references/` for 12 detailed reference files covering role definitions,
tint system, contrast ratios, OKLCH primer, pipeline equations, perceptual
volume, CSS/PPT/Word/Chart theme application, and API docs.