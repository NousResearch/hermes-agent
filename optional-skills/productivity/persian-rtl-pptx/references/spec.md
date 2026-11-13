# Deck spec format (rtl_pptx.py)

`write_file` the spec as UTF-8 JSON, then run
`python scripts/rtl_pptx.py spec.json out.pptx`. Top-level keys:

```jsonc
{
  "font": "Vazirmatn",            // single family, used everywhere
  "persian_digits": true,         // auto-convert 0-9 in all text (default)
  "footer": "عملیات شرهانی · دفاع مقدس",  // small footer on every content slide
  "theme": {                      // all optional; defaults are the deep-green/gold system
    "deep": "0E3B2E", "gold": "C9A227", "gold_lt": "D9C998",
    "cream": "F7F3E9", "ink": "20261C", "muted": "6E705F",
    "title_size": 36, "body_size": 22
  },
  "slides": [ ... ]               // objects with "type" + per-type keys
}
```

Every slide accepts `notes` (speaker notes). Content slides auto-get a
right-aligned 36pt title, gold rule, footer line and Persian slide number.
Keep body text ≤6 bullets, ≤~12 words each, ≥18pt (cartographic map
annotations are the only exception).

## Slide types

- `title` — `kicker`, `title` (54pt), `subtitle`, `subtitle2`, `presenter`,
  `date`, `pattern` (PNG path). Dark gradient, gold frame.
- `agenda` — `items`: list of strings; numbered gold circles.
- `bullets` — `title`, optional `subtitle`, `items`: strings or
  `{"text": ..., "sub": ...}` (sub-line in muted, smaller) or
  `{"text": ..., "size": ...}`.
- `split` — image left / bullets right (RTL reading order): `title`,
  `subtitle`, `image` (`path`, `x`, `y`, `w`, `h`, `card`: true to put it on a
  bordered card, `bullets_size`), `bullets` (list), `labels`: map of
  `key -> {text, fx, fy, align, size, bold, color}` where `fx`/`fy` are
  fractions (0..1) of the image and `align` is `l` (box starts at the point)
  or `r` (box ends at the point). Box width is computed from text length;
  positions are clamped inside the image. This is how maps stay legible:
  graphics in the PNG, Persian labels as native text on top.
- `table` — `title`, `subtitle`, `headers` (list), `rows` (list of lists),
  `row_height` (default 0.86), `note` (full-width note bar below).
- `timeline` — `title`, `subtitle`, `events`: list of `[date, text]` pairs;
  vertical gold spine on the right (RTL), dates bold right of the text.
- `cards` — `title`, `subtitle`, up to 2 `cards`: `{title, color, lines: []}`,
  optional `note`.
- `sources` — `items`: list of source strings, numbered.
- `closing` — `title` (44pt), `subtitle`, `pattern` (PNG). Dark gradient.

## Map recipe

1. Draw the schematic map with matplotlib **with no text** — polygons for
   land/sea, a polyline border, triangles for heights, line for the river,
   `FancyArrowPatch` for the operation axes. Save transparent PNG at
   150-170 dpi.
2. Emit/write the label positions by hand from the data transform:
   `fx = x / x_range`, `fy = 1 - y / y_range` for north-up data.
3. In the spec, list every label under `split.labels`. Run
   `verify_deck.py` and nudge `fx`/`fy` until overlaps clear (expect
   1-2 iterations). Direct labels beat a legend.

## Rules that stay true

- Persian digits everywhere in prose (`persian_digits` handles it, but
  author prose with ۱۲۳۴۵۶۷۸۹ in the first place).
- No Latin digits/letters in body text (verify_deck flags them).
- All text shapes stay within ~0.65in margins.
