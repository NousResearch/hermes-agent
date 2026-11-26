# PowerPoint Theme Mapping

## Applying a 1939 Palette to PowerPoint

An agent given a theme slug (e.g., "hugo's-mom-1939") can apply it to a PowerPoint presentation by mapping the 8 semantic roles to PowerPoint's theme XML.

## Color Slot Mapping

PowerPoint defines 12 theme color slots. Here's how the 8 roles map:

| PowerPoint Slot | 1939 Role | Tint Level | Hex Example (Hugo's Mom) |
|----------------|-----------|------------|------------------------|
| `dk1` (Dark 1) | Background | Center (#4) | `#0A0A0D` |
| `lt1` (Light 1) | Canvas | Center (#4) | `#C8BFB1` |
| `dk2` (Dark 2) | Text | Center (#4) | `#6C3D30` |
| `lt2` (Light 2) | Highlight | Tint 200 (#2) | `#FFBEB2` |
| `accent1` | Highlight | Center (#4) | `#CF6F66` |
| `accent2` | Support | Center (#4) | `#9260A2` |
| `accent3` | Chart1 | Center (#4) | `#7E6287` |
| `accent4` | Chart2 | Center (#4) | `#C25E50` |
| `accent5` | Muted | Tint 200 (#2) | `#A89F94` |
| `accent6` | Background | Tint 600 (#6) | `#170C14` |
| `hlink` | Support | Tint 100 (#1) | `#B888C4` |
| `folHlink` | Support | Tint 300 (#3) | `#7D4B97` |

## Slide Layout Mapping

| Element | Role | Tint |
|---------|------|------|
| Slide background | Background | Center |
| Title text | Highlight | Center |
| Body text | Text | Center |
| Accent bar/stripe | Support | Center |
| Subtitle text | Support | Tint 200 |
| Chart primary series | Chart1 | Center |
| Chart secondary series | Chart2 | Center |
| Table header bg | Background | Tint 700 |
| Table header text | Canvas | Tint 100 |
| Table alt row bg | Background | Tint 800 |
| Footer text | Muted | Tint 600 |

## Python-PPTX Example

```python
from pptx import Presentation
from pptx.util import Inches, Pt
import json

# Load the brand palette
with open('hugos-mom-1974.brand.json') as f:
    theme = json.load(f)

prs = Presentation()

# Apply theme colors to slide master
# (This requires creating a custom theme XML — see pptx-theme-xml reference)

# For direct slide styling:
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
bg = slide.background
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor.from_string(theme['roles']['Background']['hex'][1:])

# Title
title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
title.text_frame.text = "Hugo's Mom"
title.text_frame.paragraphs[0].font.color.rgb = RGBColor.from_string(
    theme['roles']['Highlight']['hex'][1:])
title.text_frame.paragraphs[0].font.size = Pt(36)
```

## Key Considerations

1. **PowerPoint doesn't natively support 10-tint scales.** Map the center (500-level) colors to the 12 theme slots, and use tint variations directly for custom elements.

2. **Chart colors default to accent1-accent6.** If you set accent1=Highlight and accent3=Chart1, PowerPoint's default chart will use these. Override individual series colors when fine control matters.

3. **Transitions and shadows benefit from tints.** Use Background tint 700 (dark variant) for slide transitions and shadow depths instead of pure black.

4. **Slide footer text should use Muted tint 600, not Canvas.** Muted provides subtle readability without competing with slide content.