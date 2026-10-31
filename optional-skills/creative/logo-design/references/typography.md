# Typography for Logos

Typography is pictures of words. We decode letterforms the way we decode images: their shapes carry attitude,
history and culture before a single word is read. Read this when designing wordmarks, lettermarks, lockups, or
choosing brand typefaces.

## Contents
1. Choosing a direction: type study
2. Classification and what each voice says
3. Custom vs. existing type
4. Letterform craft (spacing, kerning, optical fixes)
5. Lockups: symbol + type
6. Brand typefaces beyond the logo
7. Licensing and technical checks
8. Common mistakes

---

## 1. Choosing a direction: type study

1. Set the name in 20–40 candidate typefaces across categories, in lowercase, uppercase and title case.
   Starting with type is a legitimate way to begin a logo — once the type is right, the rest can flow from it.
2. Judge by: the shape of the letters *in combination* (not individually), legibility, the rhythm of the word,
   how the letters echo the way the word sounds, and how it reads at small size.
3. Shortlist 3–5 and look for **opportunities**: interesting letter pairs, shared strokes, counters that can hold a
   shape, a letter that can become a sign, repeated letters that can create rhythm, a natural ligature.
4. Only then customise (see §3).

Legibility varies with case: some typefaces read better in upper and lower case, others in capitals. Test.

## 2. Classification and what each voice says

| Class | Character | Typical voice |
|---|---|---|
| Humanist / Venetian serif | calligraphic stress, low contrast | literary, traditional, warm |
| Old-style serif | moderate contrast, diagonal stress | classic, trustworthy, bookish |
| Transitional serif | sharper, more vertical stress | refined, institutional |
| Modern / Didone | extreme thick–thin contrast, hairlines | fashion, luxury, drama (fragile when small) |
| Slab serif | heavy, block serifs | sturdy, industrial, confident, retro |
| Grotesque / neo-grotesque sans | even, rational | neutral, corporate, modern, "Swiss" |
| Geometric sans | built on circles/squares | modern, clean, tech, optimistic |
| Humanist sans | calligraphic proportions | friendly, readable, approachable |
| Rounded sans | soft terminals | playful, gentle, youthful |
| Script / handwritten | cursive, personal | personal, crafted, elegant or casual |
| Display / custom | idiosyncratic shapes | highly specific personality (dates fastest) |
| Monospace | fixed widths | technical, code, systematic |

Tone rules of thumb:
- Heavier weights read strong/confident; lighter weights read elegant/quiet (but fragile at small sizes).
- Wide proportions feel stable and expansive; condensed feel efficient, urgent, editorial.
- Lowercase feels approachable; uppercase feels monumental. Title case is the neutral default.
- Serif ↔ sans pairing in one mark can express "heritage + modern" (e.g. a classic ampersand in a modern gothic name).

## 3. Custom vs. existing type

- Most viewers can't tell a custom face from a stock one, yet it's precisely the unique attributes of custom
  letters that give the client ownership and protect against imitation. An unmodified stock font can be typed by
  anyone — including competitors and counterfeiters.
- Levels of customisation, from light to heavy:
  1. Tight custom spacing and kerning of an existing face.
  2. Modify a few letters (a distinctive `a`, `g`, `e`, `R`, a cut terminal, a joined pair, a swapped dot).
  3. Redraw the whole word with a consistent logic (shared radius, stroke contrast, terminal angle).
  4. Draw from scratch on a system (e.g. built on a square grid, or with the symbol's curvature applied to the type).
- **Warning**: the further letterforms depart from familiar shapes, the more quickly they date and the harder they
  are to read.
- Reflect the symbol in the type subtly (a corner radius, a cut angle, a stroke ending) — echo, don't duplicate.
- Redraw and re-space existing marks for legibility when refreshing an old logo; making it impossible to reproduce
  by typing in any existing font is itself protective.

## 4. Letterform craft

- **Spacing is optical, not metric.** Equalise the *area* of space between letters, not the distance. Round letters
  (`O`, `C`) sit closer to neighbours than straight ones (`H`, `I`); diagonals (`A`, `V`, `W`) need kerning.
- **Kern for the size of use.** Spacing that works in a book heading may look loose at logo scale or on a building.
  Check key pairs (`AV`, `To`, `LT`, `r.`, `ye`).
- **Overshoot**: round letters must extend slightly above cap height/x-height and below the baseline to look equal
  in height.
- **Horizontal strokes thinner** than verticals to look equal; thin strokes where they meet at junctions to avoid dark
  spots.
- **Consistent stroke logic**: contrast axis, terminal angles, corner radii and aperture should follow one rule.
- **Tittles and dots** (on `i`, `j`) are tempting substitution points; use them only if the substitute reads at small
  size and doesn't feel like a cliché.
- **Small-size fixes**: open apertures, larger counters, slightly heavier weight and looser spacing for a
  small-use version (optical size) of the wordmark.
- In SVG, final logo type must be **outlined paths**, not `<text>` — see `svg-construction.md`.

## 5. Lockups: symbol + type

- A lockup fixes the positions and relative sizes of all elements (symbol, wordmark, tagline). Once locked, they are
  never rearranged ad hoc.
- Create several lockups for different formats: **horizontal** (most common for headers/signage), **stacked /
  vertical** (square formats, social), **symbol only** (app icon, favicon, avatar), **wordmark only**, **with
  tagline**. Simplify complexity as size shrinks — a letterhead lockup may carry more than a mobile header.
- **Balance**: an ornate or detailed symbol pairs with plain, neutral type; a bold symbol pairs with a lighter type
  weight. The goal is complement, not contrast for its own sake.
- **Alignment**: align the symbol optically — typically centred on the cap height or x-height band, or spanning
  baseline to cap height/ascender. Check by eye; geometric centring often looks low.
- **Spacing**: the gap between symbol and wordmark is usually derived from an element of the mark (e.g. the width
  of a stem, the height of the x-height, or a fraction of the symbol's width) so it scales consistently.
- **Tagline**: significantly smaller, a lighter or contrasting style, aligned to the wordmark; define the size ratio
  and specify a version without it. Taglines rarely survive at small sizes — drop them there.
- Details like "bold vs slightly more bold" or a small kerning tweak aren't noticed consciously, but they are the
  difference between good and great.

## 6. Brand typefaces beyond the logo

- Corporate typography is often a different entity from the logo's letterforms.
- **Body/text face**: simple, highly readable, ideally a neutral sans with many weights; consider system/web
  availability and cost.
- **Display/headline face**: more freedom, providing contrast with the body text.
- **Limit to two families** in most identities (weights and italics within them are fine). More families dilute
  the voice.
- Specify web fallbacks and which faces are licensed for which uses.

## 7. Licensing and technical checks

- Free fonts are frequently **personal-use only**. A logo is commercial use: confirm the licence allows logo /
  trademark use (some licences restrict it or require a separate logo licence). Open-licence families (e.g. SIL Open
  Font License) are generally safe for logos.
- Check glyph coverage: accented characters for all the brand's languages, punctuation, `@`, `&`, numerals, currency
  symbols. Incomplete fonts silently substitute or show boxes.
- Prefer professional OpenType fonts with proper kerning tables.
- Once outlined in the logo file, the font is no longer needed to display the logo — but the brand typefaces still
  need licences for everyday use.

## 8. Common mistakes

- Using the default weight and spacing of a popular geometric sans with no modification — instantly generic.
- Too many typefaces in one mark (a script + a serif + a sans).
- Tracking lowercase text very wide (lowercase is designed to be read tight).
- Effects on type (outlines, drop shadows, bevels, distortion via horizontal/vertical scaling). Never scale type
  non-uniformly to fit — choose a condensed/extended width instead.
- Overused faces chosen because they are fashionable this year.
- Thin hairlines or tiny counters that close up at small sizes.
- Letters so customised the name becomes a puzzle.
