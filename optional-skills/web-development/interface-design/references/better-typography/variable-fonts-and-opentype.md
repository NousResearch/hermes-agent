# Variable fonts and OpenType

What a font file can do beyond drawing letters, and how to reach those abilities from CSS.

## Static vs variable

- **Static font:** one weight and one style per file. Regular + medium + bold = three files.
- **Variable font:** an entire range in one file. Any value in the range works, e.g. `font-weight: 589`.

A variable font is not automatically better. One or two weights: static files can be smaller. Several weights, optical sizes or custom axes: a variable font usually makes more sense.

## Load intended weights and styles

When you use a weight or style the active family does not provide, the browser may synthesize it. Prefer loading the faces the design actually uses. Disable synthesis only after verifying the complete fallback stack and every semantic emphasis state; `none` disables weight, style, small-cap, superscript, and subscript synthesis together and can erase distinctions when the real face is unavailable.

```css
.brand-wordmark {
  /* Safe only after this isolated treatment is verified */
  font-synthesis: none;
}
```

For body and interface text, keep synthesis enabled unless a verified font setup supplies every requested form. If only one mode is unwanted, use the specific longhand (`font-synthesis-weight`, `font-synthesis-style`, and related properties) instead of the blanket shorthand.

## Axes

Variable-font controls, each with a four-letter tag. A font only supports the axes its designer included.

| Axis | Tag | Controls |
| --- | --- | --- |
| Weight | `wght` | Stroke thickness (like `font-weight`) |
| Optical size | `opsz` | Details and spacing tuned for the display size |
| Width | `wdth` | Glyph width |
| Slant | `slnt` | Slant angle |
| Custom | e.g. `GRAD` (Roboto Flex) | Whatever the designer built |

Inter's variable file exposes only `wght` and `opsz`.

Optical sizes predate variable fonts and many fonts still ship them as separate files: Heldane Text is sturdier and more spaced for reading sizes, Heldane Display has finer details for large sizes.

## Properties over axis tags

When a property exists, use it. `font-weight` keeps working when a non-variable fallback renders; `font-variation-settings` silently does nothing. Save the raw tags for custom axes with no property of their own:

```css
/* Good: common axes use the properties */
.heading {
  font-weight: 650;
  font-optical-sizing: auto;
}

/* Good: custom axis with no property of its own */
.heading-grade {
  font-variation-settings: "GRAD" 80;
}

/* Bad: weight via raw tag breaks on fallback fonts */
.heading {
  font-variation-settings: "wght" 650;
}
```

## OpenType features

OpenType is the standard behind almost every modern font. Features are extra built-in options and, unlike axes, they work the same on static and variable fonts. A font only ships the features its designer included.

| Tag | Feature |
| --- | --- |
| `tnum` | Tabular numbers: every digit the same width |
| `zero` | Slashed zero: `0` distinct from `O` |
| `liga` | Ligatures: joins pairs like "fi" into one shape |
| `ss01`–`ss20` | Stylistic sets (numbered slots) |
| `cv01`–`cv99` | Character variants (numbered slots) |

Same rule as axes: prefer the `font-variant-*` properties, reserve `font-feature-settings` for tags with no property:

```css
/* Good: common features use the properties */
.price {
  font-variant-numeric: tabular-nums;
}

/* Good: slashed zero via the property too */
.id {
  font-variant-numeric: slashed-zero;
}

/* Good: niche feature with no property of its own */
.logo {
  font-feature-settings: "ss01" 1;
}
```

Tabular numbers matter for changing values: without them each digit has a different width and the layout shifts as values update.

## Small caps, superscripts, subscripts

- **Small capitals:** uppercase letters drawn at a smaller size. Enable real ones with `font-variant-caps`.
- **Superscripts** sit above the normal line (the 2 in x²), **subscripts** below it (H₂O). Enable proper glyphs with `font-variant-position`.

Both require the font to include the glyphs.

## Stylistic sets and character variants

`ss01` = stylistic set, slot 01. `cv11` = character variant, slot 11. What each slot does differs font to font, which is why they are numbered, not named. Check the font's docs. In Inter, `ss01` switches to open digits and `cv11` swaps in a single-story `a`.
