# Congressional brand-doc visual remediation notes (2026-06-26)

Use this as a session-specific cautionary pattern for premium venue/club brand-design PDFs when the user is actively reviewing one specific artifact.

## Failure pattern
- The active artifact was the Congressional Country Club brand-design PDF, but adjacent True Swing recall caused a wrong-brand proof/link to be produced.
- The operator needed the existing Congressional PDF URL fixed in place, not a new unrelated bundle.
- The source HTML contained logo data URIs and swatch markup, but the rendered/served PDF showed blank/white logo placeholders and weak visual proof.

## Correct recovery pattern
1. **Anchor on the exact user URL first.** Fetch/verify the current PDF path and rendered content for that URL before following any recall trail or adjacent brand history.
2. **Remove wrong-brand public proof exports immediately** if they were created during the mistake; do not leave misleading artifacts on the served proof surface.
3. **Patch the active artifact in place** when the user supplied the exact review URL. Preserve the same operator-openable URL unless the user asks for a new versioned path.
4. **Use the actual Congressional logo raster/SVG only.** Crop away the large white canvas around the logo, make transparent/cropped variants, and place those variants into cover and identity sections.
5. **Force visible print backgrounds.** Add `-webkit-print-color-adjust: exact; print-color-adjust: exact;` to core styles and swatches.
6. **Use Playwright PDF export with no browser headers/footers.** Avoid Chrome CLI output that injects timestamp/title/file URL headers unless explicitly acceptable.
7. **Verify more than page 1.** Render a contact sheet from the exact served/exported PDF and visually check: cover logo, identity logo/icon section, color swatches, and blank/overflow pages.

## QA gates used
- Page 1: Congressional logo visible, navy/gold cover visible, no wrong-brand text.
- Identity page: Congressional logo visible, no blank placeholders.
- Color page/contact sheet: visible navy/gold/warm-gold/charcoal/cream swatches.
- Served Tailnet URL returns 200 and bytes hash matches local file.
- Scope checks `/QUEUE.md` and `/golf-darin-memory-inventory/` return 404 from proof port.

## Pitfalls
- Do not call the artifact GO just because page 1 is fixed. A later contact-sheet pass may reveal overflow/blank pages or misplaced content.
- Do not leave wording like “True Swing-style” inside another brand’s packet after using a standardized template; replace with neutral class language such as “standardized premium brand book.”
- Do not send `127.0.0.1` links to the operator for proof validation; use the verified Tailnet URL.
