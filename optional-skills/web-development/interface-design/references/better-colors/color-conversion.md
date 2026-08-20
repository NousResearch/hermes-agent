# Color Conversion

Use this reference only when the user asks for conversion, the project is already standardizing on OKLCH, or an agreed color-system migration requires it. Do not convert an isolated value in a project that intentionally uses another notation. When conversion is in scope, convert the color values but leave everything else unchanged: don't change gradient interpolation or restructure the CSS.

## Supported input formats

| Format | Examples |
| --- | --- |
| Hex (3/6/8-digit) | `#f00`, `#ff0000`, `#ff000080` |
| `rgb()` / `rgba()` | `rgb(255, 0, 0)`, `rgba(255, 0, 0, 0.5)` |
| `hsl()` / `hsla()` | `hsl(0, 100%, 50%)`, `hsla(0, 100%, 50%, 0.5)` |

## Conversion examples

```css
/* Before */
color: #3b82f6;
background: #1e293b;
border-color: #e2e8f0;

/* After */
color: oklch(0.623 0.188 259.815);
background: oklch(0.279 0.037 260.031);
border-color: oklch(0.929 0.013 255.508);
```

```css
/* Before */
color: rgb(59, 130, 246);
border: 1px solid rgba(0, 0, 0, 0.1);

/* After */
color: oklch(0.623 0.188 259.815);
border: 1px solid oklch(0 0 0 / 0.1);
```

Alpha uses the forward-slash syntax. Omit alpha when it's 1.

## What to leave alone

- CSS keywords: `currentColor`, `inherit`, `initial`, `unset`, `transparent`
- Gradient interpolation methods: only convert the color stops, not the function itself
- Colors in third-party library configs that expect hex input

## Bulk conversion

Bulk conversion is a migration task, not routine cleanup. When an entire file is explicitly in scope:

1. Replace all hex colors with their oklch equivalents
2. Replace all `rgb()`, `rgba()`, `hsl()`, `hsla()` function calls
3. Leave gradient functions unchanged; only convert the color stops within them
4. Leave `currentColor`, `inherit`, `transparent`, and CSS keywords as-is
5. Preserve comments and formatting
