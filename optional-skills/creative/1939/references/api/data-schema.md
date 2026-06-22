# 1939 Data Schema

## Flagship Theme Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name (e.g., "The Wizard of Oz") |
| `slug` | string | URL-safe slug (e.g., "wizard-of-oz-1939") |
| `collection` | string | Always `"1939-flagship"` |
| `year` | number | Year of the source image |
| `source` | object | Source attribution: `source_type`, `source_title`, `source_url`, `credited_to`, `rights_status`, `rights_note` |
| `roles` | object | 8 semantic roles, each with `hex` (center), `tints` (10 hex values), `curve`, `legend_text`, `semantic` |
| `contrast` | object | 7 WCAG contrast ratios between role pairs |
| `recommended_uses` | object | Suggested use cases (e.g., dark, light, presentation, dashboard) |
| `pptx_mapping` | object | PowerPoint slot-to-role mapping for 12 theme slots |
| `docx_mapping` | object | Word element-to-role mapping |
| `perceptual_volume` | number | 0–~0.02, measures palette color variety |

## Role Object Structure

```json
{
  "Background": {
    "hex": "#0A0A0D",
    "tints": ["#4E4E53", "#39393D", "#252529", "#17171B", "#0A0A0D", "#040406", "#000001", "#000001", "#000001", "#000001"],
    "curve": "dark",
    "legend_text": "#f0ede8",
    "semantic": "Page background, card surfaces, dark containers, slide backgrounds"
  }
}
```

- **hex** is the center color (500-level), the role's identity color
- **tints** is an array of 10 hex values, index 0=lightest, index 4=center, index 9=darkest
- **curve** is `dark`, `surface`, or `standard` — controls lightness distribution
- **legend_text** is the readable-on-background text color
- **semantic** is a human-readable description of the role's purpose

## Contrast Object

```json
{
  "text_on_background": 2.21,
  "highlight_on_background": 5.76,
  "highlight_on_canvas": 1.89,
  "support_on_background": 4.16,
  "chart1_on_background": 3.75,
  "chart2_on_background": 4.72,
  "canvas_on_background": 10.87
}
```

All ratios are WCAG 2.1 relative luminance ratios. Values >= 4.5 pass AA for normal text. Values >= 3.0 pass AA for large text (18px+ bold or 24px+).

## Source Object

```json
{
  "source_type": "film",
  "source_title": "The Wizard of Oz",
  "source_url": "https://www.imdb.com/title/tt0032138/",
  "source_creator": "Victor Fleming",
  "credited_to": "Metro-Goldwyn-Mayer",
  "rights_status": "fair_use_commentary",
  "rights_note": "Film still. Source: IMDb."
}
```

`source_type` values: `film`, `artwork`, `photograph`, `album`, `nft`, `digital`, `other`.
`rights_status` values: `fair_use_commentary`, `cc0`, `public_domain`, `permissioned`.

## 6529 Card Fields (API only)

When fetched from the API (`GET /api/cards/{slug}`), card responses include
all flagship fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `season` | number | Season number (1-15) |
| `artist` | string | Card artist |
| `community` | object | Engagement metrics: `hodl_rate`, `tdh`, `tdh_rank` |
| `mint_date` | string | Mint date (ISO format) |

## Memes Index (Local `palettes/memes/index.json`)

The bundled memes index uses a slim schema (not the full API response):

| Field | Type | Description |
|-------|------|-------------|
| `slug` | string | URL-safe slug (e.g., "100-trillion-stars-100-million-people-26") |
| `name` | string | Card display name |
| `year` | number | Year of the source image |
| `season` | number | Season number (1-15) |
| `artist` | string | Card artist |
| `collection` | string | Always `"6529-memes"` |
| `pv` | number | Perceptual volume (0-~0.05) |
| `center_colors` | object | 8 role hex values (center only, no tints) |
| `community` | object | `hodl_rate`, `tdh_rank`, `tdh` |
| `contrast` | object | 2 ratios: `text_on_background`, `highlight_on_background` |
| `spectrum_count` | number | Number of spectrum colors (typically 15) |
| `api_url` | string | API path for full card data |

Note: Memes cards have center colors only. For full 10-tint scales and
7 contrast ratios, use the flagship palettes or the API.

## Community Object (6529 cards only)

```json
{
  "hodl_rate": 5.12,
  "tdh": 4743160,
  "tdh_rank": 1
}
```

- **hodl_rate**: Percentage of holders who haven't sold (engagement conviction)
- **tdh**: Total Diamond Hands metric (aggregate engagement score from 6529.io)
- **tdh_rank**: Ranking among all 496 cards (1 = highest engagement)