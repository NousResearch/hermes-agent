# DTCG Design Token Pipeline Example

This folder contains a small Design Tokens Community Group 2025.10 style token pipeline for two product profiles:

- `front` for public websites
- `admin` for dense admin interfaces

The token layers are:

1. Primitive tokens in `core.tokens.json`
2. Semantic tokens in `front.tokens.json` and `admin.tokens.json`
3. Component tokens in `front.tokens.json` and `admin.tokens.json`

All color primitives use OKLCH token values:

```json
{
  "$type": "color",
  "$value": {
    "colorSpace": "oklch",
    "components": [0.6, 0.19, 257]
  }
}
```

Profile files use DTCG aliases such as `{color.brand.600}` and `{semantic.color.action.primary.background}`.

## Files

- `core.tokens.json` - primitive tokens for color, dimension, font family, font weight, duration, and cubic Bezier easing
- `front.tokens.json` - public website semantic and component tokens with open spacing, larger radius, and more expressive motion
- `admin.tokens.json` - admin semantic and component tokens with 0.75x density, smaller radius, and 36px table rows
- `build-tokens.mjs` - Node build script with no external package dependency
- `dist/` - generated output after running the script

## Run

Generate both profiles:

```bash
node build-tokens.mjs
```

Generate only one profile:

```bash
node build-tokens.mjs front
node build-tokens.mjs admin
```

## Output

The script writes:

- `dist/front.css`
- `dist/admin.css`
- `dist/front.tokens.ts`
- `dist/admin.tokens.ts`
- `dist/tokens.ts`

Each CSS file contains light mode in `:root` and dark mode overrides in `[data-theme="dark"]`.

Example usage:

```html
<html data-theme="dark">
  <button class="button">Save</button>
</html>
```

```css
@import "./dist/front.css";

.button {
  background: var(--component-button-primary-background);
  color: var(--component-button-primary-text);
  height: var(--component-button-primary-height);
  border-radius: var(--component-button-primary-radius);
  transition-duration: var(--component-button-primary-motion-duration);
  transition-timing-function: var(--component-button-primary-motion-easing);
}
```

## Notes

- The script resolves aliases before writing CSS and TypeScript.
- The script throws if a resolved color is not OKLCH.
- The dark theme is built by applying `theme.dark` over the base profile, so component tokens can keep pointing to semantic tokens.
