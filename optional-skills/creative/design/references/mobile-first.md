# Mobile-First Design

Mobile-first is the default starting point for every interface. It does not mean the design is only for phones; it means the smallest, most constrained viewport forces the hard decisions about priority, hierarchy, and content order.

## Why Mobile-First

- Forces the one thing: if only one piece of content fits first, which is it?
- Surfaces real constraints: touch, short attention, slow connections, small screens.
- Prevents desktop sprawl: features that do not fit on mobile must justify themselves.
- Makes progressive enhancement natural: add columns, sidebars, and effects as the screen grows, never subtract.

## Workflow

1. **Frame at 375px.** Sketch or wireframe the single-column stack. Decide what lives above the fold, the primary CTA, and the reading order.
2. **Size the type for thumbs, not cursors.** Minimum 16px body. Buttons and links are easy to tap without zoom.
3. **Design touch.** Targets are 44x44px minimum. Keep enough space between tappable items.
4. **Order by user intent.** The most important action is first in the vertical stack. Secondary content can move lower or behind a toggle.
5. **Scale up.** Only after mobile is resolved, add breakpoints for wider screens.

## Common Mobile Patterns

- **Priority+ navigation:** show the most important 2-4 links, collapse the rest.
- **Off-canvas:** hide filters, settings, or secondary navigation behind a slide-in panel.
- **Stack and spread:** start stacked, then spread items into columns at medium/large.
- **Full-bleed cards on small, contained cards on large:** use the whole width on mobile, cap width on desktop for readability.

## Touch Target Checklist

- [ ] Every tap target ≥ 44x44px.
- [ ] No adjacent tappable elements closer than 8px.
- [ ] Hover-dependent interactions have a tap equivalent.
- [ ] Swipe/gesture is not the only path to the action.
