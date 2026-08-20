# Gamut Awareness & Tailwind v4

## sRGB vs Display P3

Every sRGB color exists within Display P3, but not every P3 color exists within sRGB. Display P3 covers ~50% more colors.

## Max chroma varies by lightness and hue

The gamut boundary is irregular. At L=0.5 in sRGB:

- Highest chroma: purple (H ≈ 285) at C ≈ 0.29
- Red-orange (H ≈ 0-30): C ≈ 0.20
- Lowest chroma: cyan (H ≈ 195) at C ≈ 0.09

The peak hue shifts with lightness. At L=0.7 magenta peaks, at L=0.9 green peaks. Cyan consistently has the lowest max chroma.

## Gamut checking

If a color's chroma exceeds the maximum for its L/H/space, it clips. The fix: reduce chroma while keeping L and H constant.

```css
/* Out of sRGB gamut */
oklch(0.7 0.35 150)

/* Clamped to max chroma */
oklch(0.7 0.22 150)
```

## CSS fallback patterns

```css
/* sRGB fallback for all browsers */
.accent {
  color: oklch(0.7 0.2 150);
}

/* P3 enhancement for wider gamut displays */
@media (color-gamut: p3) {
  .accent {
    color: oklch(0.7 0.3 150);
  }
}
```

For browsers without oklch support:

```css
.accent {
  color: #4ade80;
}

@supports (color: oklch(0 0 0)) {
  .accent {
    color: oklch(0.7 0.2 150);
  }

  @media (color-gamut: p3) {
    .accent {
      color: oklch(0.7 0.3 150);
    }
  }
}
```

## Tailwind v4

Tailwind CSS v4 defines its default palette in oklch. Custom themes should follow the same convention.

### Custom color scale with @theme

```css
@theme {
  --color-brand-50: oklch(0.971 0.012 250);
  --color-brand-100: oklch(0.932 0.028 250);
  --color-brand-200: oklch(0.882 0.048 250);
  --color-brand-300: oklch(0.812 0.078 250);
  --color-brand-400: oklch(0.722 0.148 250);
  --color-brand-500: oklch(0.623 0.188 250);
  --color-brand-600: oklch(0.535 0.168 250);
  --color-brand-700: oklch(0.445 0.138 250);
  --color-brand-800: oklch(0.362 0.108 250);
  --color-brand-900: oklch(0.289 0.078 250);
  --color-brand-950: oklch(0.215 0.048 250);
}
```

This gives you `bg-brand-500`, `text-brand-200`, etc. automatically.

### Opacity modifiers

Tailwind's opacity modifier syntax works with oklch:

```html
<div class="bg-brand-500/50"></div>
<!-- Compiles to: oklch(0.623 0.188 250 / 0.5) -->
```

### Migrating existing themes

1. Convert all hex values in `@theme` to oklch
2. Replace any `theme()` references that used hex
3. Test dark mode: oklch values may look slightly different due to perceptual accuracy
4. Check for hardcoded hex in component code and convert those too
