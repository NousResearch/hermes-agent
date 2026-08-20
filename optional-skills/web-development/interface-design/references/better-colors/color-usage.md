# Color Usage

How to deploy color in an interface: semantic tokens, meaning, emphasis, and appearance variants. For picking the values themselves, see [palette-generation.md](palette-generation.md); for checking pairs, see [accessibility-contrast.md](accessibility-contrast.md).

## One color, one meaning

Use a color consistently for one purpose (interactive, destructive, featured) across the interface. If the brand color signals that text is interactive, that hue (anywhere within ±15°) on non-interactive text tells users to click something that isn't clickable.

```css
/* Bad: brand blue means both "link" and "decorative heading" */
a { color: oklch(0.623 0.188 259.815); }
.section-title { color: oklch(0.65 0.17 259.815); }

/* Good: interactive elements own the brand hue; headings stay neutral */
a { color: oklch(0.623 0.188 259.815); }
.section-title { color: oklch(0.279 0.041 260.031); }
```

## Semantic tokens over raw values

Name colors by role, not by appearance, and apply them only in that role: `--color-text-secondary` is muted foreground text, and using it as a background breaks every future theme change that assumes the role.

```css
/* Good: tokens named by role, used in that role */
:root {
  --color-text-primary: oklch(0.21 0.006 285.885);
  --color-text-secondary: oklch(0.552 0.016 285.938);
  --color-separator: oklch(0.92 0.004 286.32);
  --color-surface: oklch(1 0 0);
}

/* Bad: separator token repurposed as text color because it "looked right" */
.caption { color: var(--color-separator); }

/* Bad: secondary-text token repurposed as a background */
.tag { background: var(--color-text-secondary); }
```

If a role has no token yet, add the token; don't borrow one that happens to have the right value today. In Tailwind projects, this is the `@theme` block; see [gamut-and-tailwind.md](gamut-and-tailwind.md).

## One colored action per view

When the product uses filled color to encode primary emphasis, give that treatment to one primary action in the current decision context and leave peer actions neutral. Preserve an established component hierarchy that communicates emphasis another way; do not recolor controls merely to impose this recipe. Multiple colored backgrounds are acceptable when they encode distinct states or categories and do not compete as peer actions.

```html
<!-- Good: one filled primary action, neutral secondaries -->
<button class="bg-blue-600 text-white">Save</button>
<button class="text-zinc-700">Cancel</button>

<!-- Bad: every action colored, so nothing is primary -->
<button class="bg-blue-600 text-white">Save</button>
<button class="bg-blue-600 text-white">Duplicate</button>
<button class="bg-blue-600 text-white">Export</button>
```

Put the color on the background, not the label: a filled `bg-blue-600 text-white` button reads as primary from across the room; blue label text on a neutral button reads as a link. Selected states (an active tab, a checked segment) may use the accent color on the glyph and label: that is state, not emphasis.

## Color across cultures

Color meaning is not universal. If a color is load-bearing (finance, status, alerts), verify the meaning holds in every locale you ship to.

| Color | Common Western reading | Elsewhere |
| --- | --- | --- |
| Red | Danger, loss, errors | Luck, prosperity; **gains** in Chinese financial UIs |
| Green | Success, gains, go | Losses in Chinese financial UIs |
| White | Purity, cleanliness | Mourning in parts of East Asia |
| Gold | Premium, luxury | Religious significance in some regions |

The classic case: stock tickers show gains in green for English locales and in red for Chinese locales. If your product localizes into such markets, make the gain/loss colors a per-locale token, not a hardcoded value.

## Light, dark, and increased contrast

Every custom color needs a light and a dark variant; dark mode derivation is covered in [palette-generation.md](palette-generation.md). Beyond that, users who enable increased contrast expect visibly stronger differentiation; supply it with `prefers-contrast`:

```css
:root {
  --color-accent: oklch(0.623 0.188 259.815);
}

@media (prefers-color-scheme: dark) {
  :root { --color-accent: oklch(0.707 0.165 254.624); }
}

@media (prefers-contrast: more) {
  :root { --color-accent: oklch(0.488 0.243 264.376); }
}
```

The increased-contrast variant widens the foreground/background lightness gap by at least `0.15` L over the default variant. Then re-verify the pair against APCA's preferred thresholds (Lc 90 body, Lc 75 non-body).

Two testing rules:

- **Recheck every foreground/background pair in both appearances.** A pair that passes in light mode can fail in dark mode; the palettes aren't mirror images.
- **Account for translucency.** A color on a translucent surface (`backdrop-filter` header, overlay) shifts with whatever scrolls behind it. Test it over the lightest and darkest content it can sit on, or make the surface opaque enough that the shift can't break contrast.
