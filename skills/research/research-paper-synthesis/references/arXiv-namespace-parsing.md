# arXiv Atom XML Namespace-Aware Parsing

## Problem

arXiv API returns Atom XML with namespace `http://www.w3.org/2005/Atom`. Standard `find()` calls with tag names like `"a:entry"` fail silently because namespace prefixes in the XML (e.g. `a:`) are just aliases. Python's `xml.etree.ElementTree` requires the fully expanded namespace URI in the tag.

## Solution

```python
import xml.etree.ElementTree as ET

ATOM = "http://www.w3.org/2005/Atom"

# Parse the XML file
root = ET.parse("ai.xml").getroot()

# Find entries using namespace-aware tag
for entry in root.findall(f"{{{ATOM}}}entry"):
    # Extract text fields safely
    def get_text(tag):
        el = entry.find(f"{{{ATOM}}}{tag}")
        return (el.text or "").strip() if el is not None else ""
    
    title = get_text("title")
    summary = get_text("summary")
    published = get_text("published")
    
    # Extract authors defensively (some entries have empty names)
    authors = ", ".join(
        (a.find(f"{{{ATOM}}}name") or type("X", (), {"text": ""})()).text or ""
        for a in entry.findall(f"{{{ATOM}}}author")
    )
    
    # Categories
    cats = ", ".join(c.get("term") or "" for c in entry.findall(f"{{{ATOM}}}category"))
```

## Why Not grep/sed

- XML tags may span multiple lines; grep can't reliably extract structured fields.
- Namespace prefixes (`a:`, `arxiv:`) vary by document — `a:entry` might not match.
- Python stdlib `xml.etree.ElementTree` is dependency-free and handles all edge cases.

## Common Pitfall: DeprecationWarning on Element truthiness

When using `(el or fallback).text` patterns, Python 3.12+ warns: "Testing an element's truth value will raise an exception in future versions." Use explicit `None` checks (`el is not None`) or wrap in a helper function.
