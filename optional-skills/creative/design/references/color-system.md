# Color System

A clear color system encodes meaning, reinforces hierarchy, and supports accessibility. Start with 4–6 named roles before adding accents.

## Core Palette Roles

| Role | Purpose |
|------|---------|
| background | page surface color |
| surface | cards, panels, dialogs |
| text-primary | main text on background |
| text-secondary | muted text, captions |
| border | dividers, card outlines |
| accent | primary CTA, links, active states |
| success/warning/error | semantic states |

## Contrast Requirements

- Text on background/surface: WCAG AA minimum (4.5:1 for body, 3:1 for large text).
- Interactive elements: 3:1 against adjacent colors.
- Never rely on color alone for meaning. Pair with icons, text, or shape.

## Dark Mode

- Define tokens for both light and dark up front.
- Invert `background` and `surface` carefully. Pure black is usually too harsh.
- Check contrast in both modes.

## Color in UI

- Use color sparingly. One accent is usually enough.
- Semantic colors should be consistent: red for error, amber for warning, green for success.
- Avoid gradients as decoration unless they serve the brand.
