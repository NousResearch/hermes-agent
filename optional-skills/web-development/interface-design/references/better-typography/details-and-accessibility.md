# Details and accessibility

Underlines, selection, forms, decorative text and the floors that keep everything readable.

## Underlines

Default underline position is browser-determined: sometimes too close, cutting through descenders or too thin. Pull position and thickness from the font's own metrics:

```css
a {
  text-underline-position: from-font;
  text-decoration-thickness: from-font;
}
```

The line does not have to be solid. `text-decoration-style` can draw it dotted, dashed or wavy. A dotted underline is a common hint that a word carries extra information, like an abbreviation or a defined term:

```css
abbr {
  text-decoration: underline dotted;
}
```

Or tune manually:

```css
a {
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
  text-decoration-skip-ink: auto;
  text-decoration-color: var(--color-gray-1000);
  transition: text-decoration-color 200ms ease-out;
}

a:hover {
  text-decoration-color: var(--color-gray-1200);
}
```

Unless the only thing animating is a color change, build the underline as a custom element instead of using `text-decoration`: color is the only part of a real underline that animates reliably. Animate the custom element however the effect requires.

## Selection

- `::selection` changes the background and color of selected text; a subtle way to embed brand. Keep the combination legible.
- Keep text selectable by default, including application chrome; users copy labels, identifiers, errors, and values in ways the designer may not predict.
- Use `user-select: none` only on a specific draggable or gesture-driven surface where accidental selection conflicts with the interaction. Do not apply it globally or solely to imitate native chrome.
- `::target-text` styles the phrase a shared link scrolls to.
- The Custom Highlight API styles ranges you pick yourself, like search matches, without extra markup.

## Forms and editable text

- `::placeholder` styles the hint in an empty field.
- `caret-color` colors the blinking insertion bar. Color is about as far as caret styling goes: a fully custom caret is very difficult to build and usually not worth it unless a very specific effect calls for it.

### iOS input zoom

Focusing an input with text smaller than `16px` zooms the whole page (an accessibility feature: `16px` is the web default and Safari treats smaller as too hard to read while typing).

Two fixes work, and they differ in what they do to the design rather than in correctness. Ask which one the user wants before changing an input; do not pick for them.

**Size up on mobile.** The input genuinely renders at `16px` on small screens and drops to the design size from the `sm` breakpoint up. Nothing to compensate, but the mobile input no longer matches the desktop one.

```tsx
<input className="text-base sm:text-sm" type="email" />
```

**Scale the text down.** Keep `font-size` at `16px` so Safari never zooms, then render it at the intended size with a transform. The design survives at every viewport, at the cost of two compensating calcs: widen the element by the inverse of the scale so it still fills its container once shrunk, and divide `line-height` by the same factor so the intended leading survives. `origin-left` pins the text to the start edge (`origin-right` under RTL). Above the breakpoint, drop the transform and set the real size.

```tsx
// 13px rendered from a 16px font-size: 13 / 16 = 0.8125
<div className="flex h-10 items-center rounded-[10px] bg-gray-300 px-2.5">
  <input
    className="h-full w-[calc(100%/0.8125)] origin-left scale-[0.8125] bg-transparent text-base leading-[calc(1.125/0.8125)] outline-none sm:w-full sm:scale-100 sm:text-[13px]"
    type="email"
  />
</div>
```

The transform shrinks the whole box, not just the glyphs, so let a wrapper draw the field's surface and keep the input itself transparent. A background, border or ring on the scaled element shrinks with the text and misses the intended hit area.

## Decorative text

| Property | Effect |
| --- | --- |
| `::first-letter` | Drop cap, widely supported |
| `::first-line` | Styles only the first line |
| `initial-letter` | Sizes the drop cap; limited support, no Firefox yet |
| `background-clip: text` | Clips a background or gradient to the letter shapes |
| `-webkit-text-stroke` | Outlines the letters; works across modern browsers despite the prefix |
| `text-shadow` | Like `box-shadow` but follows the character shapes |

If a text stroke draws lines inside the letters, that is the font: the stroke traces every contour and variable fonts usually keep overlapping shapes unmerged. Static fonts do not have this issue.

## Sizes

Typography must survive the reader changing it: zoom, a larger browser font size, overridden line height or letter spacing.

| Text | Size |
| --- | --- |
| Long-form body starting point | Around `16px`, verified in the actual typeface and measure |
| Inputs and menus starting point | Around `14px` |
| Captions | `13px` |
| Floor | Rarely below `12px` |

When text appears low-contrast, use `better-colors` to measure the rendered foreground/background pair and `better-accessibility` to classify the applicable requirement. Changing the project's colors remains a design decision unless the user asks for remediation.

## Font smoothing

On macOS text renders heavier than intended. Apply font smoothing once on the root layout so it covers all text. Tailwind's `antialiased` sets both properties:

```css
html {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

```tsx
<html lang="en">
  <body class="font-sans antialiased">
    <main>{children}</main>
  </body>
</html>
```
