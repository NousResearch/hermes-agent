# Craft Floor

The craft floor is the minimum quality the interface must meet before it ships. Load this reference before any UI edit and verify every item with rendered evidence, not intent.

## Verify

Run these as one batched inspection, sharing a single render.

- **Contrast:** body and placeholder text ≥ 4.5:1, large text and controls ≥ 3:1. Tint secondary text from the surface hue; never default to gray.
- **Depth:** shadows carry an offset and a soft blur. A zero-offset colored halo is decoration.
- **Spacing:** tight groups, generous separation, more space above a heading than below it. Read computed values.
- **Type:** body measure 45–75ch, display max 6rem, tracking floor -0.04em, balanced headings, obvious scale and weight steps.
- **Motion:** one authored moment, not scattered effects and not the same entrance on every section. Exponential ease-out from an already-visible default.
- **States:** hover, focus, active, disabled, loading, error, empty, and success are all designed.
- **Browser surfaces:** text selection, caret, custom scrollbars, focus rings, underline offset, and tabular numerals still carry the design.
- **Copy:** the product's own language. Controls name their action; errors name the problem and recovery.
- **Coverage:** every brief requirement is present and findable within seconds.
- **Mobile-first:** the 375px view remains the strongest story before scaling up.

## Refuse

These are the category's lazy defaults. Do not use them unless the brief explicitly earns them.

### Layout

- Same-size cards of icon + heading + text as the page structure. Cards are lazy containers; nested cards are always wrong.
- The hero-metric template: big number, small label, supporting stats, accent.
- A kicker or eyebrow above a heading. The heading carries its own weight; delete the label.
- Section numbers (`01 / 02 / 03`) unless the sequence itself carries needed information.
- A modal for a task that needs neither interruption nor protected focus.

### Surface habits

- Gradient text or gradient washes used only as decoration.
- Glass and blur as decoration rather than a specific effect.
- A colored `border-left` or `border-right` above 1px on cards, list items, callouts, or alerts.
- Hard offset shadows outside an actual neobrutalist world.
- Sparklines, progress rings, and soft-shadowed rounded rectangles standing in for content.
- Monospace as a costume for "technical."
- System display faces (Impact, Arial Black, platform sans) as the display voice of an owned world.
- Unicode glyphs or emoji standing in for an icon system.
- Geometric masks approximating a photographic subject. Use a real alpha matte or omit the effect.
- Light or dark picked by category. Pick it from the use scene: who, where, under what light.

## Handoff

When every check is green, the design is ready for the final `polish` pass.
