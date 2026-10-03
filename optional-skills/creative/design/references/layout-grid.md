# Layout Grid

A responsive grid keeps the layout predictable across screens and helps align components without arbitrary positioning.

## Default Grid

| Viewport | Columns | Gutter | Margin |
|----------|---------|--------|--------|
| small | 4 | 16px | 16px |
| medium | 8 | 24px | 24px |
| large | 12 | 24px | 24–48px |
| xlarge | 12 | 32px | centered max-width |

## Grid Types

- **Column grid:** equal columns for page and section layout.
- **Baseline grid:** vertical rhythm in 4px or 8px increments.
- **Modular grid:** columns + rows for complex dashboards.
- **Compound grid:** overlapping grids for editorial layouts.

## Common Patterns

- **Contained:** max-width container with side margins.
- **Full-bleed:** content spans the full viewport, often for heroes.
- **Sidebar:** a fixed or collapsible side panel on large screens.
- **Card grid:** auto-fill responsive cards that share one column on mobile.

## Best Practices

- Align content to the grid, not arbitrarily.
- Use CSS Grid for page layout and Flexbox for component alignment.
- Allow intentional grid-breaking for emphasis, but only once per page.
- Document grid specs for developers.
