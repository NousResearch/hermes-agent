---
name: pptx-author
description: Build PowerPoint decks headless with python-pptx. Pairs with excel-author for model-backed decks where every number traces to a workbook cell. Use for pitch decks, IC memos, earnings notes.
version: 1.0.0
author: Anthropic (adapted by Nous Research)
license: Apache-2.0
metadata:
  hermes:
    tags: [powerpoint, pptx, python-pptx, presentation, finance]
    related_skills: [excel-author, powerpoint]
---

# pptx-author

Produce a .pptx file on disk using `python-pptx`. Use when you need to deliver a deck as a file artifact, not drive a live PowerPoint session.

Adapted from Anthropic's `pptx-author` and `pitch-deck` skills in [anthropics/financial-services](https://github.com/anthropics/financial-services). The MCP / Office-JS branches of the originals are dropped — this assumes headless Python.

For the broader, already-shipped PowerPoint authoring skill (slides, speaker notes, embeds, media), see the built-in `powerpoint` skill. This skill is a lighter-weight pattern tuned for model-backed decks (pitch decks, IC memos, earnings notes) where every number must trace to a source workbook.

## Output contract

- Write to `./out/<name>.pptx`. Create `./out/` if it does not exist.
- Return the relative path in your final message.

## Setup

```bash
pip install "python-pptx>=0.6"
```

## Core conventions

### One idea per slide
Title states the takeaway; body supports it. A slide titled "Q3 Revenue" is weak; "Revenue growth accelerated to 14% Y/Y in Q3" is strong.

### Every number traces to the model
If a figure on a slide came from `./out/model.xlsx`, footnote the sheet and cell.

```
Revenue: $1,250M  (Source: model.xlsx, Inputs!C3)
```

Never transcribe numbers from memory or from a summary — open the workbook, read the named range, and bind the deck value to it programmatically when you can.

### Use the firm template when one is mounted
If `./templates/firm-template.pptx` exists, load it so the deck inherits branded colors, fonts, and master layouts.

```python
from pptx import Presentation
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()
```

### Charts: PNG-from-model beats native pptx charts
When fidelity matters (the model's chart styling must match the deck exactly), render the chart to PNG from the source workbook and embed the image. Native `pptx.chart` charts are fragile and often don't match firm conventions.

```python
from pptx.util import Inches
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(2),
                         width=Inches(8))
```

### No external sends
This skill writes a file. It never emails, uploads, or posts. Orchestration layers handle delivery.

## Skeleton

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pathlib import Path

template = Path("./templates/firm-template.pptx")
prs = Presentation(str(template)) if template.exists() else Presentation()

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Project Aurora — Strategic Alternatives"
slide.placeholders[1].text = "Preliminary Discussion Materials"

# Valuation summary slide (title-only layout)
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Valuation implies $38–$52 per share across methodologies"

# Add a table bound to model outputs
rows, cols = 5, 4
tbl_shape = slide.shapes.add_table(rows, cols,
                                   Inches(0.5), Inches(1.5),
                                   Inches(9), Inches(3))
tbl = tbl_shape.table
headers = ["Methodology", "Low ($)", "Mid ($)", "High ($)"]
for c, h in enumerate(headers):
    tbl.cell(0, c).text = h

# In a real deck, read these from the model workbook with openpyxl
data = [
    ("Trading comps",     "35", "41", "48"),
    ("Precedent M&A",     "39", "45", "52"),
    ("DCF (base)",        "36", "43", "51"),
    ("LBO (10% IRR)",     "33", "38", "44"),
]
for r, row in enumerate(data, start=1):
    for c, val in enumerate(row):
        tbl.cell(r, c).text = val

# Embed a chart rendered from the model
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = "Football field — current price $42"
slide.shapes.add_picture("./out/charts/football_field.png",
                         Inches(1), Inches(1.8), width=Inches(8))

Path("./out").mkdir(exist_ok=True)
prs.save("./out/pitch-aurora.pptx")
```

## Binding deck numbers to the source workbook

Read named ranges or specific cells from your Excel model so deck numbers never drift.

```python
from openpyxl import load_workbook

wb = load_workbook("./out/model.xlsx", data_only=True)
def nr(name):
    """Resolve a named range to its current computed value."""
    rng = wb.defined_names[name]
    sheet, coord = next(rng.destinations)
    return wb[sheet][coord].value

revenue_fy24 = nr("RevenueFY24")
implied_mid  = nr("ImpliedSharePriceBase")
```

Then build deck content using those values:
```python
slide.shapes.title.text = f"Implied share price of ${implied_mid:.2f} (base case)"
```

Remember to recalculate the workbook before reading it — openpyxl only sees computed values if something has already calculated the sheet. Run the recalc helper in the `excel-author` skill first, or open/save through a real Excel session.

## Slide-type checklist for pitch decks

A typical banking pitch deck follows this structure. Not prescriptive, but useful as a starting skeleton:

1. Cover / title
2. Disclaimer
3. Table of contents
4. Situation overview
5. Company snapshot (the target)
6. Market / sector context
7. Valuation summary (football field) — the money slide
8. Trading comps detail
9. Precedent transactions detail
10. DCF summary
11. Illustrative LBO / sponsor case
12. Process considerations
13. Appendix

## When NOT to use this skill

- Users in a live PowerPoint session with an Office MCP available — drive their live doc instead.
- Non-financial slideware (quarterly all-hands, marketing decks) — use the broader `powerpoint` skill.
- Decks with heavy animation, transitions, or speaker notes — use the broader `powerpoint` skill.

## Attribution

Conventions adapted from Anthropic's Claude for Financial Services plugin suite, Apache-2.0 licensed. Original: https://github.com/anthropics/financial-services/tree/main/plugins/agent-plugins/pitch-agent/skills/pptx-author
