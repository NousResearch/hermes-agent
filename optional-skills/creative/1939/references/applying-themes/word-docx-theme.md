# Word/DOCX Theme Mapping

## Applying a 1939 Palette to Word Documents

Word's theme system uses a similar 12-color model to PowerPoint. The same role-to-slot mapping applies with document-specific adjustments.

## Word Theme Color Slots

| Word Slot | 1939 Role | Tint Level | Use In Document |
|----------|-----------|------------|-----------------|
| `dk1` | Background | Center | Dark headings, separators |
| `lt1` | Canvas | Center | Page background (light mode) |
| `dk2` | Text | Center | Body text |
| `lt2` | Highlight | Tint 200 | Subtitle text, highlighted bg |
| `accent1` | Highlight | Center | Heading 1, emphasis |
| `accent2` | Support | Center | Heading 2, secondary accents |
| `accent3` | Chart1 | Center | Chart/bar series 1 |
| `accent4` | Chart2 | Center | Chart/bar series 2 |
| `accent5` | Muted | Tint 200 | Table borders, subtle accents |
| `accent6` | Background | Tint 600 | Footer bg, sidenote bg |
| `hlink` | Support | Center | Hyperlinks |
| `folHlink` | Support | Tint 300 | Followed hyperlinks |

## Document Element Mapping

| Element | Dark Mode | Light Mode |
|---------|-----------|------------|
| Page background | Background center | Canvas center |
| Heading 1 | Highlight center | Highlight center |
| Heading 2 | Support center | Support center |
| Body text | Canvas tint 100 | Text center |
| Table header bg | Background tint 700 | Canvas tint 200 |
| Table header text | Highlight tint 100 | Background tint 900 |
| Table alt row bg | Background tint 800 | Canvas tint 100 |
| Table border | Muted tint 700 | Muted tint 300 |
| Link text | Support center | Support center |
| Caption/footnote | Muted tint 500 | Muted tint 600 |
| Block quote bar | Support tint 600 | Support tint 300 |
| Block quote bg | Background tint 700 | Canvas tint 100 |
| Code block bg | Background tint 800 | Canvas tint 100 |
| Code block text | Canvas tint 100 | Text center |

## Python-DOCX Example

```python
from docx import Document
from docx.shared import Pt, RGBColor
import json

with open('hugos-mom-1974.brand.json') as f:
    theme = json.load(f)

doc = Document()

# Apply background (for dark mode, set page bg color)
# Note: python-docx doesn't directly support page background color.
# For dark mode docs, use section background shading.

# Heading 1 — Highlight
def hex_to_rgb(hex_str):
    """#RRGGBB → RGBColor"""
    return RGBColor.from_string(hex_str[1:])

h1 = doc.add_heading('Introduction', level=1)
for run in h1.runs:
    run.font.color.rgb = hex_to_rgb(theme['roles']['Highlight']['hex'])

# Heading 2 — Support
h2 = doc.add_heading('Section Title', level=2)
for run in h2.runs:
    run.font.color.rgb = hex_to_rgb(theme['roles']['Support']['hex'])

# Body paragraph — Text
p = doc.add_paragraph('Body text content here.')
for run in p.runs:
    run.font.color.rgb = hex_to_rgb(theme['roles']['Text']['hex'])

# Table with themed header
table = doc.add_table(rows=3, cols=3)
# Header row bg — Background tint 700
header_bg = theme['roles']['Background']['tints'][6]  # Index 6 = 700-level
header_text = theme['roles']['Highlight']['tints'][1]  # Index 1 = 100-level
for cell in table.rows[0].cells:
    shading = cell._element.get_or_add_tcPr()
    # Set background and text colors via XML manipulation
```

## Key Considerations

1. **Word doesn't support CSS `color-mix()`.** All tinted colors must be pre-computed hex values. The brand JSON provides these directly.

2. **Dark mode Word documents require explicit section shading.** Set the page background to Background center and override all text colors.

3. **Table formatting benefits from tint levels.** Use Background tint 700 for header rows and tint 800 for alternating rows. This provides visual hierarchy without flat black.

4. **Hyperlinks should always use Support center**, not Highlight. Highlight is for headings and CTAs. Support is for interactive elements like links.

5. **Captions and footnotes use Muted tint 500-600.** These are intentionally subtle — they should be readable but not draw attention from the main content.

6. **For print:** Light mode palettes work better on paper. Background (dark) palettes consume more ink and may not render well on standard printers.