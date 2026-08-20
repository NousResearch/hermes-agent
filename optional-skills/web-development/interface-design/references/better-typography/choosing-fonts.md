# Choosing fonts

Choosing a typeface, the right file format and understanding why fonts look the way they do.

## Choosing a typeface

Font families set the tone before the specific font does.

| Category | Traits | Use for |
| --- | --- | --- |
| Serif | Small strokes at the ends of letters guide the eye along a line | Long passages, editorial reading |
| Sans-serif | Clean, even shapes that stay crisp at small sizes | Default for most interfaces (Helvetica, Inter, Geist) |
| Monospace | Every glyph the same width so columns line up | Code, tables, tabular data |
| Display | Drawn for large headlines | Marketing headlines, hero text |
| Script | Mimics handwriting | Rare, decorative moments |

CSS exposes `cursive` and `fantasy` keywords for the last two categories.

"Display" in a font's name does not make it a display font. Fonts like SF Pro and Heldane ship a `Display` variant for large sizes and a `Text` variant for smaller sizes. Use the variant that matches the size you are setting.

### Rules

- Fewer fonts is usually better. Rarely use more than three. Marketing pages can be more expressive than apps.
- The same applies to sizes and weights. They define hierarchy, but overusing them hurts readability quickly.
- Pair for contrast, not similarity. A serif headline with a sans body looks like a deliberate display/reading split. Two near-identical sans-serifs look like a mistake.
- Thin weights are display-only. Below `18px`, stay at weight `400`+; Ultralight/Thin/Light (`100`–`300`) strokes disappear at text sizes and on low-DPI screens. Reserve them for `28px`+ display text, and even there check they hold up against the background.

## Font family scope

Applying or reviewing typography never requires a new typeface. Use the product's existing type system unless the task explicitly asks for a type change, and do not introduce a paid or proprietary face just to satisfy a review checklist. Rendering details like font smoothing, text wrapping and tabular numbers do not override the project's chosen font family.

When a type change is asked for: a system-native macOS/iOS feel comes from the system stack; a commercial face such as Helvetica Now is a brand decision and keeps a practical fallback stack.

```css
/* System-native macOS/iOS feel */
html {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Commercial brand face with safe fallbacks */
html {
  font-family: "Helvetica Now", "Helvetica Neue", Arial, sans-serif;
}
```

## Formats

| Format | Notes |
| --- | --- |
| `.woff2` | Brotli compression, broadly supported. Use this on the web. |
| `.woff` | Older compression. Fallback only for very old browsers. |
| `.ttf` / `.otf` | Raw formats, no web compression, larger files. Desktop only unless there is no other option. |

## Anatomy of a typeface

| Term | Meaning |
| --- | --- |
| x-height | Height of a lowercase `x` |
| Cap height | Height of uppercase letters |
| Baseline | The invisible line letters sit on |
| Ascender | Part of a letter rising above the x-height |
| Descender | Part dropping below the baseline |

These measurements are why two fonts at the same `font-size` can look like different sizes. A font with a large x-height looks bigger.
