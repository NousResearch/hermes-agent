# 1939 MCP Server

Perceptual color palette tools for AI agents. Expose 525 curated palettes with 8 semantic roles via the Model Context Protocol.

## Setup

```bash
pip install -r requirements.txt
python3 mcp_1939_server.py
```

The server reads palette data from `../palettes/` (brand JSONs, memes index, flagship descriptions). Set `NINETEEN_DATA_DIR` to override the data directory.

## Tools

| Tool | Description |
|------|-------------|
| `palette_lookup(name)` | Find a palette by name or slug. Fuzzy matching supported. Returns full brand JSON with roles, tints, contrast, and document mappings. |
| `palette_search(query, limit)` | Search palettes by mood, color, use case, or character. Returns matching themes with key hex colors and descriptions. |
| `palette_recommend(use_case, mood, limit)` | Get recommendations for a specific use case (presentation, dashboard, document, website, data_viz) and optional mood (warm, cool, bold, minimal, cinematic, intimate, dramatic, neutral). |

## Resource

- `palettes://flagship/list` — JSON list of all 29 flagship themes

## Prompt

- `apply_palette_to_document(palette_name, document_type)` — Generate application instructions for applying a palette to a PowerPoint, Word, or web document.

## Example

```python
from mcp_1939_server import palette_lookup

# Look up a theme by name
result = palette_lookup("Hugo's Mom")
# Returns: name, slug, year, roles (8 roles × hex + 10 tints + curve + semantic),
#          contrast ratios, recommended_uses, pptx_mapping, docx_mapping,
#          perceptual_volume, voice-ready description
```

## Data Format

See `../SKILL.md` for the full 1939 skill documentation and `../references/` for detailed reference docs on roles, tints, contrast, OKLCH, and document mappings.

## License

CC0 — public domain. No attribution required.