# Production (Build · Verify · Troubleshoot)

The engineering runbook for kami: from HTML / Python templates to PDF / PPTX deliverables. Four parts: **HTML -> PDF** · **Python -> PPTX** · **Verify & Debug** · **16 known pitfalls**.

---

## Part 1 · HTML -> PDF (WeasyPrint)

### Install

```bash
pip install weasyprint pypdf --break-system-packages --quiet
```

Linux first-time:
```bash
apt install -y libpango-1.0-0 libpangoft2-1.0-0 fonts-noto-cjk
```

### Generate

```python
from weasyprint import HTML
HTML('doc.html').write_pdf('output.pdf')
```

**CWD matters**: `@font-face { src: url("xxx.ttf") }` uses relative paths, so run from the directory containing the font file.

```bash
cd /path/to/html-and-font
python3 -c "from weasyprint import HTML; HTML('doc.html').write_pdf('out.pdf')"
```

### Fonts

**Most stable setup**: font file alongside HTML, `@font-face` with relative path.

```html
<style>
@font-face {
  font-family: "TsangerJinKai02";
  src: url("TsangerJinKai02-W04.ttf");
  font-weight: 400 500;
}
body { font-family: Charter, Georgia, Palatino, serif; }
</style>
```

**No commercial font available**: fallback chains are embedded in every template.

```css
/* English */
font-family: Charter, Georgia, Palatino,
             "Times New Roman", serif;

/* Chinese */
font-family: "TsangerJinKai02", "Source Han Serif SC",
             "Noto Serif CJK SC", "Songti SC", Georgia, serif;

/* Japanese */
font-family: "YuMincho", "Yu Mincho", "Hiragino Mincho ProN",
             "Noto Serif CJK JP", "Source Han Serif JP",
             "TsangerJinKai02", Georgia, serif;
```

**Font fallback affects page count**. Any font swap requires re-running the page-count check. If it overflows: lower `font-size` first, then tighten margins, then cut content.

**Claude Desktop skill ZIPs do not bundle large Chinese font files**: `TsangerJinKai02-W04.ttf` and `TsangerJinKai02-W05.ttf` are close to 19MB each and can make Claude.ai / Desktop skill upload or execution time out. Release ZIPs must be generated with `scripts/package-skill.sh`, which excludes both TTF files. Templates still keep local-first and jsDelivr fallback `@font-face` paths.

**Standalone HTML export** (sending a filled HTML file to someone else): this is not guaranteed to work outside the project tree. If the recipient cannot set up the font environment, use the PDF output instead.

If you do need to share HTML: the font file and the HTML must live in the same directory, and the `@font-face src` must use a bare filename with no path prefix:

```css
@font-face {
  font-family: "TsangerJinKai02";
  src: url("TsangerJinKai02-W04.ttf") format("truetype");
}
```

Remove the `../fonts/` prefix that templates use when fonts are in the project tree. The recipient must place the `.ttf` file alongside the `.html` file before running WeasyPrint. When in doubt, deliver the PDF.

### Page spec

```css
@page {
  size: A4;                     /* or 210mm 297mm / A4 landscape / 13in 10in */
  margin: 20mm 22mm;
  background: #f5f4ed;          /* extend past margins to avoid white printed edge */
}
```

### Headers & footers

```css
@page {
  @top-right {
    content: counter(page);
    font-family: serif; font-size: 9pt; color: #6b6a64;
  }
  @bottom-center {
    content: "{{DOC_NAME}} · {{AUTHOR}}";
    font-size: 9pt; color: #6b6a64;
  }
}

@page:first {
  @top-right { content: ""; }
  @bottom-center { content: ""; }
}
```

### WeasyPrint support matrix

| Solid | Partial | Unsupported |
|---|---|---|
| CSS Grid / Flexbox | CSS filter / transform (partial) | JavaScript |
| `@page` rules | inline SVG (some attrs) | `position: sticky` |
| `@font-face` | gradients (slow, use sparingly) | CSS animations / transitions |
| `break-before` / `break-inside: avoid` | | |
| CSS variables `var(--name)` | | |
| `::before` / `::after` | | |

### PDF metadata

WeasyPrint reads standard meta tags in `<head>` and writes them into the PDF (Title / Author / Subject / Keywords). All templates have pre-built placeholders:

```html
<head>
  <title>{{DOC_TITLE}}</title>
  <meta name="author"      content="{{AUTHOR}}">
  <meta name="description" content="{{DESCRIPTION}}">
  <meta name="keywords"    content="{{KEYWORDS}}">
  <meta name="generator"   content="Kami">
</head>
```

**Auto-inference rules** (Claude fills these from the document content without asking):

| Field | Source |
|---|---|
| `<title>` | H1 heading or `.header .title` text |
| `author` | Resume / letter / portfolio: person's name from the document; everything else: `"Kami"` |
| `description` | One sentence extracted from the first 2 paragraphs, ≤150 characters |
| `keywords` | 3-5 keywords from title + section headings, comma-separated |
| `generator` | Fixed `"Kami"`, already set in template, do not change |

**Verify**:

```bash
pdfinfo assets/examples/one-pager-en.pdf   # shows Title / Author / Subject
```

---

## Part 2 · Python -> PPTX (python-pptx)

PPT shares the same design language but the medium (screen, 16:9, one-idea-per-slide) changes the details: fonts larger, layouts more rigid.

### Install

```bash
pip install python-pptx --break-system-packages --quiet
```

### Dimensions

- **16:9 widescreen** (preferred): 13.33 × 7.5 inch
- **4:3 traditional**: 10 × 7.5 inch
- **Safe zone**: 0.5 inch margin on all sides (projector crop), plus 0.3 inch at bottom for page number

### Palette (1:1 with design.md)

```python
from pptx.dml.color import RGBColor

PARCHMENT   = RGBColor(0xf5, 0xf4, 0xed)
IVORY       = RGBColor(0xfa, 0xf9, 0xf5)
BRAND       = RGBColor(0x1B, 0x36, 0x5D)
NEAR_BLACK  = RGBColor(0x14, 0x14, 0x13)
DARK_WARM   = RGBColor(0x3d, 0x3d, 0x3a)
OLIVE       = RGBColor(0x5e, 0x5d, 0x59)
STONE       = RGBColor(0x87, 0x86, 0x7f)
BORDER_WARM = RGBColor(0xe8, 0xe6, 0xdc)
TAG_BG      = RGBColor(0xee, 0xf2, 0xf7)
```

### Type (bigger than print, optimized for projection)

| Role | Size | Font |
|---|---|---|
| Title | 48pt | Serif 500 |
| Subtitle | 24pt | Serif 400 |
| H2 chapter | 32pt | Serif 500 |
| H3 subtitle | 20pt | Serif 500 |
| Body | 18pt | Serif 400 |
| Caption | 14pt | Serif 400 |
| Footer | 12pt | Serif 400 |

English stack on PowerPoint:
- Serif: `Charter` -> `Georgia` -> `Palatino`
- Sans: same as serif (single-font-per-page rule)

### Nine standard layouts

1. **Cover**: parchment background, centered display title + brand-colored short line + subtitle / author / date
2. **Contents**: parchment, left-aligned `01  Chapter title` (number serif brand-colored)
3. **Chapter divider**: full brand ink-blue background, centered white title - the **only** fully chromatic slide in the deck
4. **Content slide**: eyebrow (serif stone) + core claim (serif near-black) + brand line + body (serif dark-warm)
5. **Data slide**: top takeaway + 2-4 metric cards (big number serif brand + small label serif olive)
6. **Comparison**: eyebrow + left column (muted, OLIVE/STONE) vs. right column (full-weight, DARK_WARM/NEAR_BLACK), separated by a 1pt BORDER warm-gray vertical divider. Left = "Before/Old/Problem"; right = "After/New/Solution". Use `comparison_slide()`.
7. **Pipeline**: eyebrow + title + serif numerals 01/02/03 + step title + step description, laid out in equal-width columns. All steps visible at once (no click-reveal). Use `pipeline_slide()`.
8. **Quote**: parchment, minimal, centered serif quote + `- Source`
9. **Closing**: parchment, centered "Thank you / Q&A / Contact"

### Script skeleton

Full working example in `assets/templates/slides-en.py`. Key bits:

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

PARCHMENT = RGBColor(0xf5, 0xf4, 0xed)
BRAND     = RGBColor(0x1B, 0x36, 0x5D)
SERIF = "Charter"
SANS  = SERIF

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

def blank_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = PARCHMENT
    bg.line.fill.background(); bg.shadow.inherit = False
    return slide
```

### PPT notes

1. **One idea per slide** - if it runs over three lines, split it
2. **No default PowerPoint template** - it's cool-blue-gray, clashes with parchment
3. **Animations**: don't. Parchment is a print aesthetic, not a SaaS demo. At most `fade`
4. **Export to PDF** for sharing - cross-machine consistency is better than .pptx
 - macOS: Keynote -> Export to PDF
 - Linux: `libreoffice --headless --convert-to pdf output.pptx`

### Slide scale rules (from production decks)

Print and slide share the same palette and serif, but sizing is different.
One rule covers most adjustments: **macro spacing x1.6, micro details x0.5** (letter-spacing, border weight, border-radius).

| Item | Print | Slide | Why |
|---|---|---|---|
| Container | A4 portrait | 1920x1080 or A4 landscape | Fixed ratio, never `100vw x 100vh` |
| Title size | 30-34pt | 48-64pt | Projection needs larger display |
| Slide padding | N/A | 72-80px top, 80px sides | Less than 72px top feels cramped |
| Eyebrow tracking | 0.5-1pt | 3px max | Print tracking looks scattered at slide scale |
| Display tracking | 0 to -0.2pt | -0.5pt | Tighten large titles to prevent letter gaps |
| Header gap | 8-14pt | 36px+ between rule and H1 | Below 36px the rule looks like an underline |
| Line-height titles | 1.1-1.3 | 1.3 minimum | CJK characters collide below 1.3 at slide sizes |
| Code blocks | Full runnable code | Pseudocode + `.hl` keywords | Slide code is for structure, not execution |
| Images | Inline with text | `width:100%; object-fit:contain; max-height:780px` | Avoid fixed height that clips different ratios |
| Footer | Per-template | Single component, CSS-injected text | Prevents drift across 50+ slides |
| Punctuation | Standard | Use ` - ` for short joins, `<ol>` for parallel items | CJK commas break visual rhythm at slide scale |

---

## Part 3 · Verify & Debug

### The three-step loop (mandatory after every change)

```bash
# 1. Generate
python3 -c "from weasyprint import HTML; HTML('doc.html').write_pdf('out.pdf')"

# 2. Page count
python3 -c "from pypdf import PdfReader; print(len(PdfReader('out.pdf').pages))"

# 3. Visual inspect (when in doubt)
pdftoppm -png -r 300 out.pdf inspect
```

**Not verified = not done.**

### Did the font actually load?

```bash
pdffonts output.pdf
```

If the output shows `DejaVuSerif` / `Bitstream Vera` - your specified font didn't load, fell through to system ultimate fallback. Expected: `Charter`, `Georgia`, `TsangerJinKai02`, or a Japanese Mincho face such as `YuMincho`, `Hiragino-Mincho`, `Noto-Serif-CJK-JP`, or `Source-Han-Serif-JP`.

### One-step build + validate

Project script `scripts/build.py` is the productized version of the three-step loop:

```bash
python3 scripts/build.py               # all 12 examples
python3 scripts/build.py resume-en     # one target + page count + fonts
python3 scripts/build.py --check       # scan for CSS rule violations
python3 scripts/build.py --check-density       # warn on pages with >25% trailing whitespace
```

### Layout stabilizer (HTML templates)

When a constrained template is near page overflow (one-pager, letter, resume, and English variants), run the stabilizer first, then verify:

```bash
python3 scripts/stabilize.py all --out-dir dist/stabilized --report
python3 scripts/stabilize.py one-pager letter resume one-pager-en letter-en resume-en --strict --report
python3 scripts/build.py --check
python3 scripts/build.py --verify
```

`scripts/stabilize.py` is intentionally independent from `scripts/build.py`. Thresholds and solver behavior are controlled by `references/stabilizer_profiles.json`.

### Hi-res visual inspection

```bash
pdftoppm -png -r 160 output.pdf preview      # standard
pdftoppm -png -r 300 output.pdf preview      # detail bugs
pdftoppm -png -r 400 output.pdf preview      # extreme detail (tag double-rect check)
```

### 5-point pre-ship review

A successful render is not enough. Scan these before delivery:

| Dimension | Pass standard |
|---|---|
| Fact accuracy | Numbers, dates, versions, funding, specs, and market facts have sources; uncertainty is written as magnitude or marked as missing |
| Content structure | Headlines read as a summary; each paragraph opens with a claim; no ceremonial filler |
| Material coverage | Branded docs include logo, product image, or UI screenshot coverage; missing materials are clearly marked |
| Typographic detail | Fonts load correctly, line-height stays in spec, emphasis only marks numbers or distinctive phrases, tag backgrounds are solid hex |
| PDF readiness | Page count fits, placeholders are replaced, visual inspection shows no overflow, overlap, or broken page breaks |

If any row fails, fix it before delivery.

---

## Part 4 · 16 known pitfalls

Every entry below came from a real failure. Check here first when something looks wrong.

Severity scale: **(P0)** render-breaking, must fix before delivery. **(P1)** breaks the design contract (rhythm, spec). **(P2)** visible to a careful reader, but not blocking. **(P3)** operational: affects workflow, not visual output.

### 1. (P0) Tag / Badge double-rectangle bug (the worst)

**Symptom**: PDFs show two concentric rectangles on tag backgrounds at zoom - an outer softer one and an inner tighter one. Especially visible on mobile PDF viewers.

**Root cause**: WeasyPrint renders `rgba(..., 0.xx)` by compositing the **padding area** and the **glyph pixel area** independently. Glyph anti-aliasing stacks alpha differently, creating the second visible edge.

**Fix**: Tag backgrounds must be solid hex. No rgba.

```css
/* avoid */ .tag { background: rgba(201, 100, 66, 0.18); }
/* use   */ .tag { background: #E4ECF5; }
```

**rgba -> solid conversion** (parchment `#f5f4ed` base + ink-blue `#1B365D`):

| rgba alpha | Solid hex |
|---|---|
| 0.08 | `#EEF2F7` |
| 0.14 | `#E4ECF5` |
| **0.18** | **`#E4ECF5`** ← default |
| 0.22 | `#D0DCE9` |
| 0.30 | `#D6E1EE` |

Formula: `solid_channel = base + (foreground - base) × alpha`. Different base colors (e.g. ivory) need re-computing.

**Want "breathing" texture?** Use `linear-gradient` - the whole tag rasterizes as one bitmap, no alpha compositing:

```css
.tag { background: linear-gradient(to right, #D6E1EE, #E4ECF5 70%, #EEF2F7); }
```

**Aesthetic warning**: gradients work engineering-wise but usually oversell the tag. Priority order: lightest solid (`#EEF2F7`) > standard solid (`#E4ECF5`) > gradient (rarely). If the reader's eye lands on the tag background shape before the text inside - you went too far.

### 2. (P0) Thin border + radius = double circle

**Symptom**: `border: 0.4pt solid ...` + `border-radius: 2pt` shows two parallel arcs on zoom.

**Root cause**: WeasyPrint strokes border inner and outer paths separately when `< 1pt` + rounded corners - at thin widths they can't overlap.

**Fix (pick one)**:
1. Use background fill instead (preferred, design-consistent)
2. Border ≥ 1pt
3. Drop `border-radius`

### 3. (P1) 2-page hard-limit overflow

For resume, one-pager, and other length-capped docs.

**Common causes**: font fallback, content added, font-size bumped by accident, line-height pushed from 1.4 to 1.6.

**Diagnose**: `pdffonts output.pdf` to verify what actually loaded.

**Fix (priority)**:
1. Cut redundant qualifiers ("deeply researched" -> "researched")
2. Merge related data points in the same section
3. Drop non-essential items whole (not piecemeal)
4. Reduce section spacing (use sparingly - affects global rhythm)
5. Last resort: shrink font by 0.1-0.2pt

**Don't**: cut cover / education / timeline structural blocks; cut emphasis (resume becomes flat).

**5-project high-density layout** (when content legitimately needs 5 projects on page 1): add `class="resume--dense"` to `<body>`. This activates a built-in CSS variant that applies the following adjustments without touching any content:

| Property | Default | Dense |
|---|---|---|
| `body font-size` | 9.2pt | 9pt |
| `.proj-text line-height` | 1.40 | 1.38 (CN) / 1.40 (EN) |
| `.tl-body font-size` | 9pt | 8.5pt (CN only) |
| `.section-title margin-top` | 5mm | 3.5mm |

Apply dense mode only when the 4-project layout already overflows. Do not use it as a default; the visual rhythm is noticeably tighter.

### 4. (P1) Font fallback causes page count inconsistency

**Symptom**: 2 pages locally, 4 pages in CI / on server.

**Root cause**: font file neither alongside HTML nor system-installed.

**Fix**:

```bash
# Preferred: multi-source download script (retries, size validation)
bash scripts/ensure-fonts.sh

# Or put .ttf alongside the HTML
cp TsangerJinKai02-W04.ttf workspace/

# macOS fallback font
brew install --cask font-source-han-serif-sc

# Linux system install
apt install fonts-noto-cjk
mkdir -p ~/.fonts && cp *.ttf ~/.fonts/ && fc-cache -f
```

### 5. (P2) CJK and Latin crowding (Chinese mode only)

**Symptom**: "125.4k GitHub Stars" - k and G feel glued.

**Wrong fixes**: hand-added `&nbsp;` / `margin-left: 2mm` (misaligns adjacent elements).

**Right fix**: separate spans with flex gap:

```html
<div class="metric">
  <span class="metric-value">125.4k</span>
  <span class="metric-label">GitHub Stars</span>
</div>
```
```css
.metric { display: flex; align-items: baseline; gap: 6pt; }
```

### 6. (P2) Full-width vs half-width spaces (Chinese mode)

- **Between Chinese characters**: U+3000 full-width space + `·` + space
- **Between Latin words**: half-width space + `·` + space
- **Mixed**: prefer flex gap, don't hand-type spaces

### 7. (P2) Thousands / percent / arrows - be consistent

| Use | Avoid |
|---|---|
| `5,000+` | `5000+` |
| `90%` | `90 %` (pre-space) |
| `->` | `->` / `-&gt;` |

Self-check:
```bash
grep -oE '->|->|⟶|⇒' doc.html | sort | uniq -c
grep -oE '[0-9]{4,}' doc.html | sort -u
```

### 8. (P2) Too much / too little emphasis

- Four or five ink-blue runs in one line -> visual fatigue, no focal point
- Entire section with none -> flat, no scan handles

**Rule**: ≤ 2 emphases per line, ≥ 1 per section, only **quantifiable numbers or distinctive phrases** get highlighted - never adjectives.

Healthy ratio: one emphasis per 80-150 words.

### 9. (P0) `height: 100vh` doesn't work

**Symptom**: full-bleed cover using `height: 100vh` renders empty.

**Root cause**: viewport units are undefined in WeasyPrint's `@page` context.

**Fix**:

```css
.cover {
  min-height: 257mm;                   /* A4 height 297 - 40mm margins */
  display: flex;
  flex-direction: column;
  justify-content: center;
}
```

### 10. (P1) `break-inside` fails inside flex

**Symptom**: `.card { break-inside: avoid }` still splits across pages.

**Root cause**: WeasyPrint's flex/grid `break-inside` support on direct children is incomplete.

**Fix**: wrap the flex item in an extra block:

```html
<div class="row">
  <div class="card-wrapper"><div class="card">...</div></div>
</div>
```
```css
.row { display: flex; }
.card-wrapper { break-inside: avoid; }
```

### 11. (P3) Hide page number on the first page

```css
@page:first {
  @top-right { content: ""; }
}
```

### 12. (P2) Printed white margin around the page

**Symptom**: printing produces a white border even though `background` is set.

**Root cause**: default `@page background` only covers the content area, not the margin.

**Fix**:

```css
@page {
  size: A4; margin: 20mm;
  background: #f5f4ed;    /* extends past margins */
}
```

### 13. (P2) Blurry images

**Symptom**: images in PDF look soft.

**Root cause**: WeasyPrint renders at source pixel density. A4 @ 300 dpi = 2480 × 3508 pixels.

**Fix**: source images at 2x or 3x.

### 14. (P3) Verification loop (catch-all)

```bash
python3 -c "from weasyprint import HTML; HTML('doc.html').write_pdf('out.pdf')"
python3 -c "from pypdf import PdfReader; print(len(PdfReader('out.pdf').pages))"
pdftoppm -png -r 300 out.pdf inspect    # when in doubt
```

**Not verified = not done.**

### 15. (P0) SVG marker `orient="auto"` ignored

**Symptom**: SVG arrows using `<marker orient="auto">` or `orient="auto-start-reverse"` all point right (the marker's default drawing direction), regardless of the path's tangent angle.

**Root cause**: WeasyPrint's SVG renderer does not support the `orient="auto"` attribute on markers. The marker is always drawn at 0°.

**Fix**: skip `<marker>` entirely. Draw each arrowhead as a manual chevron `<path>` at the endpoint, with the direction hardcoded.

```xml
<!-- Bad: marker arrow, WeasyPrint renders all pointing right -->
<defs>
  <marker id="a" orient="auto" ...>
    <path d="M2 1L8 5L2 9" .../>
  </marker>
</defs>
<path d="M 440 52 Q 568 52 568 244" marker-end="url(#a)"/>

<!-- Good: manual chevron, direction per endpoint -->
<path d="M 440 52 Q 568 52 568 244" fill="none" stroke="#504e49" stroke-width="1.5"/>
<path d="M 560 236 L 568 244 L 576 236" fill="none" stroke="#504e49" stroke-width="1.5"
      stroke-linecap="round" stroke-linejoin="round"/>
```

Chevron templates (tip at endpoint, 8px arm length):

| Direction | chevron path |
|---|---|
| down | `M (x-8) (y-8) L x y L (x+8) (y-8)` |
| left | `M (x+8) (y-8) L x y L (x+8) (y+8)` |
| up | `M (x-8) (y+8) L x y L (x+8) (y+8)` |
| right | `M (x-8) (y-8) L x y L (x-8) (y+8)` |

### 16. (P1) Slide letter-spacing must be halved

**Symptom**: Slide text looks "scattered" or over-spaced when print letter-spacing values (e.g. `letter-spacing: 8px`) are used directly.

**Root cause**: Print letter-spacing values are tuned for small sizes (8-12pt). At slide sizes (48-64px), the same absolute value gets multiplied out of control.

**Fix**: Slide letter-spacing = print value / 2. Mono fonts are exempt (fixed-width by nature, no extra tracking needed).

```css
/* Print eyebrow */
.eyebrow { letter-spacing: 6px; }

/* Slide eyebrow */
.slide .eyebrow { letter-spacing: 3px; }   /* halved */
```

### 17. (P0) Donut chart: `stroke-dasharray` on `<circle>` fails in Puppeteer PDF

**Symptom**: Donut chart renders as a solid-colored circle (no visible segments) when built with `stroke-dasharray` on `<circle>` elements and exported via Puppeteer `page.pdf()`.

**Root cause**: Puppeteer's PDF engine (Chromium's print pipeline) does not render `stroke-dasharray` on circles with large stroke-widths reliably. The dashes collapse into a solid fill.

**Fix**: Always use `<path>` arc segments for donut charts in Puppeteer-generated PDFs. Each segment is a pie-slice path with outer arc, radial line, inner arc, and close:

```xml
<!-- S1: 55% (0° to 198°), cx=120 cy=155 R=70 r=40 -->
<path d="M 120 85
  A 70 70 0 1 1 58.2 224.9
  L 91.8 191.3
  A 40 40 0 1 0 120 115
  Z" fill="#1B365D"/>
```

The donut-chart template (`assets/diagrams/donut-chart.html`) already uses this pattern. When creating a donut inline (not from template), follow the same `<path>` arc approach — never `stroke-dasharray`.

**WeasyPrint**: `stroke-dasharray` on circles works in WeasyPrint. This pitfall is Puppeteer-specific.

### 18. (P0) Do not fabricate status claims in document content

**Symptom**: Document states a deliverable was provided ("Le fichier CSV est fourni", "Data appended") when it was not actually generated or attached.

**Root cause**: Auto-completion of status language without verifying the file exists or was written.

**Fix**: Before writing any claim that a file, data set, or artifact is "provided", "attached", or "included", verify the file exists on disk or was just written. If the artifact is missing, state the gap honestly — "Le fichier CSV n'a pas été généré" — rather than inventing its presence. This applies to all document types (reports, proposals, summaries).

**Symptom**: Slide text looks "scattered" or over-spaced when print letter-spacing values (e.g. `letter-spacing: 8px`) are used directly.

**Root cause**: Print letter-spacing values are tuned for small sizes (8-12pt). At slide sizes (48-64px), the same absolute value gets multiplied out of control.

**Fix**: Slide letter-spacing = print value / 2. Mono fonts are exempt (fixed-width by nature, no extra tracking needed).

```css
/* Print eyebrow */
.eyebrow { letter-spacing: 6px; }

/* Slide eyebrow */
.slide .eyebrow { letter-spacing: 3px; }   /* halved */
```

### 19. (P0) `position: fixed` and `position: absolute` unreliable in WeasyPrint

**Symptom**: Watermarks, overlays, or floating badges using `position: fixed` or `position: absolute` disappear, render off-page, or appear in the wrong location in the PDF.

**Root cause**: WeasyPrint's positioning model differs from browsers. `position: fixed` is not supported in the `@page` context. `position: absolute` can work but is unstable — elements may be clipped by page breaks, rendered outside margins, or dropped entirely depending on the surrounding layout.

**Fix**: Use inline flow elements instead of positioned overlays.

```css
/* Bad: watermark with fixed/absolute — disappears or misplaces */
.watermark {
  position: fixed; /* or absolute */
  top: 50%; left: 50%;
  transform: translate(-50%, -50%) rotate(-30deg);
}

/* Good: watermark as inline block in document flow */
.watermark {
  display: block;
  text-align: center;
  margin: 0 0 16pt 0;
  border: 2pt solid #cc6600;
  padding: 6pt 16pt;
  font-size: 14pt;
  letter-spacing: 6pt;
}
```

For watermarks / status badges: place them at the top of `<body>` as a regular block element. Control visibility via `display: none` / `display: block` (or inline `style="display:{{PLACEHOLDER}}"`).

**Alternative for true overlays**: Use `::before` / `::after` pseudo-elements on a container — these are more reliably positioned than standalone `position: absolute` elements.

**Test**: Always verify positioned elements with `pdftoppm -png -r 300` or visual PDF inspection. A silent drop (no error, no warning) is the most common failure mode.

---

### 20. (P2) Blurry logos / images in WeasyPrint PDFs

**Symptom**: Logo or embedded image appears soft/pixelated in the generated PDF despite being a high-resolution source file.

**Root cause**: WeasyPrint does not upscale or sharpen images during CSS scaling. A 2000×2000px logo scaled down to 72pt via CSS will look soft because the browser's subpixel rendering doesn't apply the same sharpening as a screen display.

**Fix (pre-process, don't CSS-scale)**:

1. **Resize source image to 2× display size** before embedding. For a 72pt logo, use 144×144px (72pt × 2 at 72dpi).
2. **Apply sharpening + contrast boost** during resize (PIL Sharpener filter + contrast enhancement).
3. **Set CSS to exact display size** — no scaling: `height: 72pt; width: auto;`
4. **Add `image-rendering: crisp-edges`** to prevent interpolation artifacts.

```css
.logo-block img {
  height: 72pt;
  width: auto;
  image-rendering: crisp-edges;
}
```

```python
from PIL import Image, ImageFilter
img = Image.open("source-logo.png")
img = img.resize((144, 144), Image.LANCZOS)
img = img.filter(ImageFilter.SHARPEN)
img = img.point(lambda p: p * 1.1)  # contrast boost
img.save("optimized-logo.png")
```

**Avoid**: `-webkit-optimize-contrast` prefix — causes WeasyPrint warnings and has no effect on PDF output.

**Benign warnings**: `WARNING: Font-face 'JetBrains Mono' cannot be loaded` — this is expected when the font isn't installed locally; WeasyPrint falls back to system monospace automatically. No action needed.

---

## Part 5 · HTML -> DOCX (pandoc + SVG-to-PNG)

PDF is the delivery format; DOCX is the collaboration format. For proposal / report scenarios where the recipient needs to edit, comment, or forward to their team, ship a `.docx` alongside the PDF.

### Why two-step

`pandoc input.html -o output.docx` looks straightforward, but inline `<svg>` blocks fail Word's OOXML validation. Word treats them as unknown content and either drops them or refuses to open the file. The fix is to bake every SVG to PNG first, swap the `<svg>` blocks for `<img>` tags, then run pandoc.

### Install

```bash
# macOS
brew install pandoc librsvg

# Linux
apt install pandoc librsvg2-bin
```

### Workflow

```bash
# 1. Extract every SVG to its own file, ensuring xmlns is set
python3 - << 'EOF'
import re
src = open('input.html').read()
for i, m in enumerate(re.finditer(r'<svg[^>]*>.*?</svg>', src, re.DOTALL)):
    svg = m.group(0)
    if 'xmlns=' not in svg[:200]:
        svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    with open(f'/tmp/diagram-{i}.svg', 'w') as f:
        f.write('<?xml version="1.0"?>\n' + svg)
EOF

# 2. Rasterize each SVG to a wide PNG
for f in /tmp/diagram-*.svg; do
    rsvg-convert "$f" -o "${f%.svg}.png" -w 1600
done

# 3. Replace each <svg>...</svg> in the HTML with <img src="/tmp/diagram-N.png">
#    (do this with the same regex iteration that produced the indices above)

# 4. Convert
pandoc input-with-png.html -o output.docx
```

### What carries over

| Carries over | Lost |
|---|---|
| Headings, body, lists, tables | Grid layouts, custom positioning |
| Brand color on headings, `<strong>`, `.hl` | Subtle borders and shadows |
| PNG figures at full width | SVG editability |
| Page breaks via `<hr class="page-break">` | `@page` margins |

The recipient gets a Word document they can edit text in and replace figures in (drag a new PNG over the existing one). Complex layout differences are expected; the goal is editability, not visual fidelity.

### When not to ship a DOCX

Skip this for resume, one-pager, slides, and portfolio. They are visual artifacts; converting them produces a degraded version with no editability gain. Long-doc / proposal / equity-report are the document types where a DOCX companion has a real audience.
