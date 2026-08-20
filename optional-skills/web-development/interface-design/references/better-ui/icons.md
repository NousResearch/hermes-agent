# Icons

Icon weight, states, sizing, and direction: the details that make icons sit naturally in an interface.

## Match Icon Stroke to Text Weight

An icon next to text should carry the same optical weight as the text, or the pair looks mismatched: a hairline icon beside semibold text reads as broken, a heavy icon beside regular text shouts.

| Adjacent text | Icon stroke width (24px grid) |
| --- | --- |
| Regular (400), 14–16px | `1.5px` |
| Medium/Semibold (500–600) | `2px` |
| Bold (700), or emphasized standalone | `2.5px` |

```html
<!-- Good: stroke tuned to the label weight -->
<button class="flex items-center gap-2 font-semibold">
  <PlusIcon stroke-width="2" class="size-4" />
  New project
</button>

<!-- Bad: default 1.5px stroke against a bold label -->
<button class="flex items-center gap-2 font-bold">
  <PlusIcon stroke-width="1.5" class="size-4" />
  New project
</button>
```

Two related consistency rules:

- **One optical strategy per surface.** Do not mix icon libraries with incompatible stroke conventions on one toolbar. If the chosen library intentionally supports stroke variants, match them to adjacent text as above; otherwise preserve the set's native stroke and use size or color for emphasis.
- **Size icons relative to the text's cap height**, typically `1em`–`1.25em` when inline with text, so the pair scales together.

## One SVG, Recolored per State

Never ship separate icon assets for default/hover/selected/disabled states. Use a single SVG drawn with `currentColor` and let CSS state drive the color:

```html
<!-- Good: one asset, states are CSS -->
<svg fill="none" stroke="currentColor" stroke-width="2">…</svg>
```

```css
.icon-button { color: oklch(0.552 0.016 285.938); }
.icon-button:hover { color: oklch(0.21 0.006 285.885); }
.icon-button[aria-pressed="true"] { color: oklch(0.623 0.188 259.815); }
.icon-button:disabled { opacity: 0.4; }
```

```html
<!-- Tailwind -->
<button class="text-zinc-500 hover:text-zinc-900 aria-pressed:text-blue-600 disabled:opacity-40">
  <BookmarkIcon />
</button>
```

Hardcoded fills inside the SVG (`fill="#666"`) break this; strip them to `currentColor` when importing icons.

## Outline Default, Fill Active

When an icon set offers outline and filled variants, use them as a state pair, not interchangeably:

| Variant | Use for |
| --- | --- |
| Outline | Default state: toolbars, list rows, inline with text |
| Fill | Selected/active state: the active tab, a toggled bookmark, a liked heart |

```tsx
// Good: variant communicates state
<TabIcon variant={isActive ? "solid" : "outline"} />

// Bad: filled icons everywhere, so the active tab has no state signal
<TabIcon variant="solid" />
```

The swap between variants is a contextual icon animation; use the exact cross-fade values in [icon-transitions.md](icon-transitions.md).

## Design at Render Size

An icon that looks great at 48px can collapse into mush at 16px. Details that read at large sizes (thin interior lines, tight counters, fine texture) blur or alias when small.

- Test every icon at the smallest size it will render (often `16px`); it must stay recognizable there.
- Prefer simplified glyphs for small contexts over scaling down detailed artwork.
- Keep icons on the pixel grid at their render size: a 16px icon drawn on a 24px grid with fractional scaling renders soft. Use the icon set's native grid sizes (`16`, `20`, `24`) rather than arbitrary scales.
- Always SVG, never raster, so the same asset stays crisp at every density.

## Icons in RTL

Under `dir="rtl"`, flip icons whose meaning is tied to reading direction, and leave the rest alone:

| Flip | Don't flip |
| --- | --- |
| Back/forward arrows, chevrons in navigation | Logos and brand marks |
| Text-block glyphs (alignment, lists, indent) | Checkmarks |
| Speaker/volume waves (emanate in reading direction) | Physical objects: clocks, cups, pencils |
| "Send" style directional glyphs | Media playback (play/rewind refer to tape direction, convention keeps them LTR) |

```css
/* Good: mirror only direction-dependent icons */
[dir="rtl"] .icon-directional {
  scale: -1 1;
}
```

```html
<!-- Tailwind -->
<ChevronRightIcon class="icon-directional rtl:-scale-x-100" />
```

Analyze composite icons part by part: a badge or slash overlay may keep its position even when the base glyph flips. Accessible names for icon-only buttons are covered by the `better-accessibility` skill.
