# Great typography

Good typography is mostly restraint. A sensible scale, comfortable spacing and enough contrast beat any clever effect. A label, a table cell, a marketing headline and an article paragraph should not share one set of rules. Apply these principles when building or reviewing anything with text in it.

When reviewing, read the page instead of scanning the code: squint to check the hierarchy holds, read one full paragraph for comfort, and resize the viewport to catch bad wrapping, widows and truncation at real content lengths.

The words themselves (button labels, error messages, empty states) are covered by the `better-writing` skill; semantic heading structure by `better-accessibility`; spatial RTL layout and logical CSS properties by `better-layout`; rendered-pair contrast measurement and color remediation by `better-colors`. This skill owns how text renders, wraps, and behaves in mixed-direction content.

Write every fix in the project's own idiom: the styling system already in use, never a second one alongside it. The [cheat sheet](css-cheat-sheet.md) maps each declaration to its Tailwind equivalent.

## Quick Reference

| Category | When to use | Reference |
| --- | --- | --- |
| Choosing fonts | Font categories, pairing, formats, typeface anatomy | [choosing-fonts.md](choosing-fonts.md) |
| Variable fonts & OpenType | Axes, weights, tabular numbers, stylistic sets | [variable-fonts-and-opentype.md](variable-fonts-and-opentype.md) |
| Spacing & sizing | Type scale, heading hierarchy, line-height, letter-spacing, text trimming | [spacing-and-sizing.md](spacing-and-sizing.md) |
| Wrapping & punctuation | Measure, wrapping, truncation, smart punctuation, RTL | [wrapping-and-punctuation.md](wrapping-and-punctuation.md) |
| Details & accessibility | Underlines, selection, forms, decorative text, contrast | [details-and-accessibility.md](details-and-accessibility.md) |
| CSS cheat sheet | Quick lookup of every property covered, with Tailwind equivalents | [css-cheat-sheet.md](css-cheat-sheet.md) |
| Review output format | Severity scale, findings table, verification, verdict | [review-output.md](review-output.md) |

## Core Principles

### 1. Serve the Right Format

Use `.woff2` (Brotli compression, broadly supported) on the web. `.woff` is a fallback only for very old browsers; `.ttf` and `.otf` are raw desktop formats with no web compression. How the files are loaded is the project's own concern, this skill does not prescribe it.

### 2. Properties Over Raw Tags

When a CSS property exists, use it. `font-weight: 650` instead of `font-variation-settings: "wght" 650`, `font-optical-sizing: auto` instead of `"opsz"`, `font-variant-numeric: tabular-nums` instead of `font-feature-settings: "tnum" 1`. Properties keep working when a non-variable fallback renders. Reserve the raw-tag properties for custom axes (`"GRAD" 80`) and niche features (`"ss01" 1`) that have no property of their own.

### 3. Load Intended Weights and Styles

Browsers may synthesize a requested weight or style that the active family does not provide. Prefer loading the faces the design actually uses. Set `font-synthesis: none` only after verifying that every required bold, italic, small-cap, superscript, and subscript form remains visually distinct across the complete fallback stack; disabling synthesis is not a diagnostic and must not erase emphasis.

### 4. Fewer Fonts, Sizes and Weights

Rarely use more than three fonts. Weight and size define hierarchy, but overusing them hurts readability quickly. Pair for contrast, not similarity: a serif headline with a sans body reads as deliberate, two near-identical sans-serifs read as a mistake. Below `18px`, stay at weight `400`+; weights under `300` are display-only (`28px`+), they disappear at text sizes.

### 5. Use a Type Scale with Semantic Names

Define a small set of sizes and deviate from it as little as possible. Hard-coded sizes without a system break down at scale. For solo projects, default names like `text-sm` work fine as long as the usage rules are clear. On a team, name sizes by use (`text-body-sm`), not by size, so the rules stay consistent.

### 6. Heading Sizes Descend with Level

Within a coherent page hierarchy, map heading levels to descending steps of the type scale: a visually subordinate heading should not accidentally overpower its parent. Adjacent levels may share a size toward the small end of the scale as long as weight or spacing keeps them distinct. Pick semantic heading elements according to `better-accessibility`; this skill controls only their visual treatment.

### 7. Line-Height by Role

Headings tighter, around `1.1`. Body copy `1.5` to `1.6`. Prefer unitless values so line-height scales with the font size; fixed values like `24px` do not. Tight line-height is for short text: anything that wraps to three or more lines needs at least `1.4`, even in height-constrained rows.

### 8. Letter-Spacing by Size

Large headings often look better with slightly negative letter-spacing. Small uppercase labels need a little positive letter-spacing so letters do not feel crowded. Body copy at reading sizes needs neither.

### 9. Cap the Measure

Long lines make it hard for the eye to find the next line. Cap long-form text around 60–75 characters per line. Any unit works; what matters is that a cap exists and the resulting line length sits in range. [Unit choices and the pixel equivalents](wrapping-and-punctuation.md#measure-line-length).

### 10. Wrap Deliberately

`text-wrap: balance` distributes text evenly across lines: use it on headings. `text-wrap: pretty` avoids leaving a single short word on the final line: use it on descriptions. Skip both in long-form text. `overflow-wrap: break-word` where long words, links or IDs could escape the container. `white-space: nowrap` on labels and badges where a line break looks broken.

### 11. Tabular Numbers on Changing Values

Digits have different widths by default, so timers, counters and prices shift layout as they update. Apply `font-variant-numeric: tabular-nums` to any value that changes.

### 12. Truncate Without Losing Content

Single line: `text-overflow: ellipsis` with `overflow: hidden` and `white-space: nowrap`. Multiple lines: `line-clamp`. Truncation hides content, so if the missing text matters, keep the full value reachable in a tooltip or expanded view.

### 13. Write Copy Naturally, Style with CSS

Store text in natural case and control presentation with `text-transform`, so redesigns never require rewriting copy. Use smart punctuation: curly quotes in prose (straight quotes in code), an en dash for ranges like `2010–2020`, an em dash to set off a thought, the single ellipsis character, `&nbsp;` to keep values like `16 px` together and `&shy;` to control where long words may break.

### 14. Underlines from the Font

Default underlines sit wherever the browser decides. Pull position and thickness from the font's own metrics with `text-underline-position: from-font` and `text-decoration-thickness: from-font`, or tune manually with `text-decoration-thickness`, `text-underline-offset` and `text-decoration-skip-ink`. `text-decoration-style` draws the line dotted, dashed or wavy; a dotted underline is a common hint that a word carries extra information, like an abbreviation or a defined term. Unless the only thing animating is a color change, build the underline as a separate element instead of using `text-decoration`: color is the only part of a real underline that animates reliably.

### 15. Inputs at 16px on Mobile

iOS Safari zooms the whole page when an input's text is smaller than `16px`. Two fixes hold the font size at `16px` in different ways, so ask which one the design wants instead of choosing silently: size the input up on mobile (`text-base sm:text-sm`), which changes how it looks on small screens, or keep `font-size: 16px` and render the intended size with `transform: scale()`, compensating width and `line-height` so the design is identical at every viewport. [Both recipes](details-and-accessibility.md).

### 16. Size and Contrast Floors

Start long-form body text near the browser default of `16px`, then judge it in the actual typeface, measure, platform, and product density. UI text can go smaller: `14px` is a useful starting point for inputs and menus (inputs still need `16px` on mobile, see principle 15), `13px` for captions, rarely below `12px`. When text appears low-contrast, use `better-colors` to measure the rendered pair and `better-accessibility` to classify the requirement; do not change colors unless asked.

### 17. Font Smoothing on the Root

On macOS text renders heavier than intended. Apply `-webkit-font-smoothing: antialiased` and `-moz-osx-font-smoothing: grayscale` (both covered by Tailwind's `antialiased`) once on the root layout, never per component.

### 18. Language and Bidi Behavior

Set `lang` so browsers and assistive technology choose the right pronunciation, quotes, and hyphenation. Set `dir` at the document or content boundary where direction changes, preserve digit order, and use `<bdi>` for isolated mixed-direction values when needed. Spatial mirroring and logical CSS properties belong to `better-layout`.

### 19. Keep Useful Text Selectable

`::selection` can carry brand into the reading experience when the selected combination stays legible. Keep text selectable by default. Use `user-select: none` only on a specific draggable or gesture-driven surface where accidental selection demonstrably interferes with the interaction; never disable selection across the interface or merely because a button label can be highlighted.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| `.ttf`/`.otf` served on the web | Convert to `.woff2` |
| `font-variation-settings: "wght"` for weight | `font-weight` (works with non-variable fallbacks) |
| `font-feature-settings: "tnum" 1` | `font-variant-numeric: tabular-nums` |
| Synthesized face differs from the intended design | Load the required face; disable only the verified synthesis mode without erasing emphasis |
| Hard-coded one-off font sizes | Use the type scale |
| Child heading visually overpowers its parent | Map that section's hierarchy to descending scale steps |
| Heading element picked for its default size | Choose semantics with `better-accessibility`, then set the visual size in CSS |
| `line-height: 24px` on scalable text | Unitless value (`1.5`) |
| Full-width paragraphs | Cap around 60–75 characters per line |
| Orphan on the last line of a paragraph | `text-wrap: pretty` |
| Lopsided two-line heading | `text-wrap: balance` |
| Numbers cause layout shift | `tabular-nums` |
| Truncated text with no way to read it | Tooltip or expanded view for the full value |
| `UPPERCASE` typed into copy | Natural case + `text-transform` |
| Justified text in an interface | `text-align: start`; reserve justify for specific editorial layouts |
| Underline cuts through descenders | `text-decoration-skip-ink: auto`, `from-font` metrics |
| Inputs below `16px` zoom on iOS | Ask first: `text-base sm:text-sm`, or `16px` scaled down with `transform` to keep the designed size |
| Root layout omits font smoothing | Apply `antialiased` once at the root |
| Mixed-direction value renders in the wrong order | Set the correct `lang`/`dir`; isolate the value with `<bdi>` when needed |
| Selection disabled across application chrome | Restore selection; suppress it only on a specific interaction that conflicts with dragging or gestures |
| Extra-info hint with no visual cue | Dotted underline via `text-decoration-style: dotted` |
| Thin/Light weight on `14px` UI text | Weight `400`+ below `18px`; thin weights are display-only |
| `leading-none` on a three-line card description | At least `1.4` on any text that wraps to 3+ lines |

## Reporting

A standalone typography review is finished when every confirmed finding is reported in the format in [review-output.md](review-output.md), with verification and a verdict. Under `better-interface`, its format governs instead.
