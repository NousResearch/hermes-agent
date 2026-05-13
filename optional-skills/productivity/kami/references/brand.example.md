# Brand Profile (Optional)
#
# Kami treats this as the lowest-resolution context: a fallback when the current
# request is ambiguous. The current document's needs always come first. Leave
# fields blank when you don't have a strong preference — silence is fine.
#
# To activate: cp references/brand.example.md ~/.config/kami/brand.md
# Then edit the values below.

---
# ─── Identity & contact ───
name: ""                    # Your name. Used in author metadata and {{NAME}} placeholders.
role_title: ""              # e.g. "Software Engineer", "Partner", "Founder"
email: ""                   # e.g. "you@example.com"
website: ""                 # e.g. "yoursite.com" (no protocol needed)
github: ""                  # GitHub handle only, e.g. "tw93" (not the full URL)
x: ""                       # X (Twitter) handle only, e.g. "HiTw93"
city: ""                    # e.g. "San Francisco"
country: ""                 # e.g. "USA"

# ─── Brand identity ───
company: ""                 # Company or project name for headers and footers
tagline: ""                 # One-liner used in one-pager / long-doc footer
logo: ""                    # Absolute path to logo file, e.g. "~/Downloads/logo.svg"
brand_color: ""             # Hex to override --brand, e.g. "#1B365D". Warm palette still applies.

# ─── Document defaults ───
language: ""                # cn / en / ja. Used when the request language is ambiguous.
default_doc_type: ""        # one-pager / long-doc / letter / portfolio / resume / slides / equity-report / changelog
output_format: pdf          # pdf / pptx / both
page_size: a4               # a4 / letter
always_include_toc: false   # true to auto-add TOC in long-doc and portfolio

# ─── Content conventions ───
date_format: "YYYY-MM-DD"   # Date format used in content
currency_locale: ""         # USD → $, M/B notation; CNY → 元, 万/亿 notation
footer_note: ""             # Standing disclaimer appended to footers, e.g. "Confidential"
signature_block: ""         # Closing signature text for letters
tone: balanced              # formal / balanced / casual
---

# Habits

Freeform notes Kami should respect. Write in plain prose or bullets. Be specific
about document types when a preference only applies to one ("equity reports: ...").
These are advisory: Kami applies them when they fit the content, not mechanically.

Examples to replace with your own:

- Equity reports: dense, evidence-led, cite sources inline. Avoid adjective-heavy commentary.
- Letters: formal opening, casual closing. Keep to one page unless content demands two.
- Slides: prefer 5-7 slides, avoid bulleted walls, one strong assertion per slide title.
- Always include a one-line confidentiality disclaimer on internal documents.
- Long docs: open with the conclusion, then support it. Do not bury the thesis.
