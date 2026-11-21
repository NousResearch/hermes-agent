# Design Tokens

Design tokens are the shared source of truth for visual values. They keep the design and code synchronized across frameworks.

## Token Layers

| Tier | Example | Description |
|------|---------|-------------|
| primitive | `--color-blue-500` | raw value, no semantic meaning |
| semantic | `--color-brand-primary` | role in the system |
| component | `--button-primary-background` | specific component use |

## Core Token Categories

- **Color:** brand, neutrals, semantic (success/warning/error).
- **Typography:** font families, sizes, weights, line heights.
- **Spacing:** 4px or 8px scale.
- **Sizing:** touch targets, icon sizes, border-radius.
- **Elevation:** shadows, z-index, layer definitions.
- **Motion:** durations, easings.
- **Breakpoints:** the named viewport ranges.

## Framework Implementation

- **Angular:** CSS custom properties in `styles.scss` or token files.
- **React:** CSS variables or a theme object in a provider.
- **Blazor:** `:root` custom properties in `site.css`.

## Best Practices

- Tokens live in one file or one set of files.
- Use semantic names, not raw values, in component code.
- Document the meaning of each token, not just the value.
- Support light and dark mode from the start.
