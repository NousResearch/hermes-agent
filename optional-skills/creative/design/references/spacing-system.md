# Spacing System

A predictable spacing system creates rhythm and reduces arbitrary margin decisions.

## Base Unit

Use 4px or 8px as the base unit. 8px is the most common because it scales well and aligns with platform conventions.

## Scale

| Token | Value | Common use |
|-------|-------|------------|
| xs | 4px | tight icon/text alignment |
| sm | 8px | component internal padding |
| md | 16px | card padding, default gap |
| lg | 24px | section gaps, form blocks |
| xl | 32px | major section separation |
| 2xl | 48px | page-level sections |
| 3xl | 64px | hero or landing sections |

## Margins and Padding

- Mobile margins: 16px.
- Tablet margins: 24px.
- Desktop margins: 24–48px, or a max-width container with auto margins.
- Component internal padding: 16px mobile, 24px desktop.

## Vertical Rhythm

- Stack spacing: 1.5x the line-height is a safe default between paragraphs.
- Section gaps: use the scale, not arbitrary values.
- Group related items closer; separate unrelated items more.

## Best Practices

- Never use arbitrary pixel values. Map every spacing decision to the scale.
- Use CSS custom properties to expose the scale to the team.
- Keep `margin` and `padding` aligned to the same grid.
