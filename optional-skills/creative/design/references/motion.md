# Motion and Micro-Interactions

Motion should clarify, not decorate. It answers a user's action or signals a state change.

## When to Use Motion

- A state changes and the user needs to understand where it went.
- A long process needs feedback.
- A modal or panel appears and the user needs context.
- A list reorders and the change must be visible.

## When to Avoid Motion

- Decorative entrance animations on every section.
- Hover transitions on every card.
- Anything that competes with the primary action.
- Motion that ignores `prefers-reduced-motion`.

## Common Patterns

- **Fade/slide for modals:** fast (150–250ms), with a clear entry point.
- **Skeleton loading:** matches the shape of the final content.
- **Button press:** subtle scale or color shift on active.
- **Page transitions:** 200–300ms, natural easing.

## Timing

| Type | Duration | Easing |
|------|----------|--------|
| micro | 100–150ms | ease-out |
| UI | 200–300ms | ease-in-out |
| page | 300–500ms | ease-in-out |
| emphasis | 400–600ms | spring or custom |

## Best Practices

- Use CSS transitions for simple state changes.
- Use CSS animations for loops or complex sequences.
- Prefer `transform` and `opacity` for performance.
- Always provide a reduced-motion fallback.
