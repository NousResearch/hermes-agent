# Markdown to PDF Pipeline (WeasyPrint)

## Overview

Convert markdown files to styled PDFs using WeasyPrint (HTML to PDF). Pipeline: MD → HTML → PDF.

## Prerequisites

WeasyPrint requires pip to be bootstrapped first (on some systems like Ubuntu/WSL):

```bash
# Bootstrap pip if missing
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --break-system-packages

# Install dependencies
pip3 install --break-system-packages weasyprint markdown
```

WeasyPrint requires system fonts (DejaVu Sans covers most Unicode needs) and Pango for layout.

## Core Pattern

```python
import markdown
import weasyprint

# markdown MUST be instantiated, not just called as a function
# WRONG: html = markdown.markdown(text, extensions=[...])
# RIGHT:
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
html = md.convert(text)

# Generate PDF from HTML
weasyprint.HTML(filename='/tmp/input.html').write_pdf('/tmp/output.pdf')
```

**Why `markdown.markdown()` fails for code blocks:** The top-level function creates a new `Markdown()` instance on each call, but extension processing for fenced code blocks requires state to be maintained across multiple calls. The `codehilite` extension especially needs a persistent instance. Always use `Markdown().convert()`.

## Full Pipeline Script

```python
import markdown
import weasyprint
import os

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #888;
    }
}
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.5; }
h1 { font-size: 20pt; color: #1a1a2e; border-bottom: 2px solid #1a1a2e; }
h2 { font-size: 14pt; color: #2d4059; margin-top: 1.5em; border-bottom: 1px solid #ccc; }
h3 { font-size: 11pt; color: #3d5a80; margin-top: 1.2em; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 8pt; }
pre { background: #f4f4f4; padding: 10px; border-radius: 6px; font-size: 8pt;
      overflow-x: auto; page-break-inside: avoid; border-left: 3px solid #2d4059; }
pre code { background: none; padding: 0; }
.codehilite { background: #f4f4f4; border-radius: 6px; padding: 10px;
              page-break-inside: avoid; border-left: 3px solid #2d4059; }
.codehilite pre { background: none; padding: 0; border: none; border-radius: 0; margin: 0; }
blockquote { border-left: 4px solid #3d5a80; padding-left: 12px; color: #555; font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th { background: #1a1a2e; color: white; padding: 8px; }
td { border: 1px solid #ddd; padding: 5px; }
tr:nth-child(even) { background: #f9f9f9; }
"""

def md_to_pdf(md_text, output_path, title="Document"):
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{md.convert(md_text)}
</body>
</html>"""

    html_file = output_path.replace('.pdf', '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_body)
    weasyprint.HTML(filename=html_file).write_pdf(output_path)
    os.remove(html_file)
    print(f"Generated: {output_path} ({os.path.getsize(output_path) // 1024} KB)")

# Usage
with open('document.md', 'r') as f:
    md_to_pdf(f.read(), 'output.pdf', 'My Document')
```

## Splitting Large Documents into Parts

When splitting a markdown file into parts (for separate PDFs), watch for anchor collision:

```python
import re

with open('bigdoc.md', 'r') as f:
    content = f.read()

# WRONG: first occurrence of "# PART 1:" may be in the TOC
idx = content.find('# PART 1:')  # Could return TOC position!

# RIGHT: anchor at line start to skip TOC entries
matches = list(re.finditer(r'^# PART \d+:', content, re.MULTILINE))
# Or: skip past TOC
toc_end = content.find('---', content.find('## Table of Contents')) + 5
part1_start = content.find('\n# PART 1:', toc_end)

# ALWAYS verify extraction
part1 = content[matches[0].start():matches[1].start()]
print(f"Code blocks: {part1.count('```python')}")  # Should match document
```

**Table of Contents entries** start at column 0 but use different markers (often `###` level). Content sections use `#` at column 0. Verify which occurrence you got by checking for code blocks — if the extracted part has 0 code blocks but the document has 45, you extracted the TOC.

## WeasyPrint Limitations

- Does NOT support JavaScript — purely static HTML/CSS
- Complex CSS (flexbox/grid) can cause layout issues — keep CSS simple
- Page breaks: `page-break-after: avoid` on headings, `page-break-inside: avoid` on code blocks
- Font loading: WeasyPrint embeds fonts — ensure fonts are available on the system. DejaVu Sans covers most Unicode including non-Latin scripts.
- Rendering time: scales with content size. A 500KB HTML file may take 5-10 seconds.

## Alternative Tools

| Tool | Pros | Cons |
|------|------|------|
| WeasyPrint | Pure Python, no external deps, good CSS support | Slow for large docs, limited JS |
| Pandoc | Best conversion quality, many formats | Requires install, not in WSL |
| wkhtmltopdf | Fast, good for simple pages | Requires X server or Xvfb |
| Chromium | Exact browser rendering | Requires Chrome, heavy |

WeasyPrint is the right choice for Python-native environments without external toolchain access (WSL default).