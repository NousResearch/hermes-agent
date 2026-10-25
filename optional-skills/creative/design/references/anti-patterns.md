# Anti-Patterns and Generated-Design Tells

Some patterns look AI-generated because they appear in every generic output. Avoid them unless the brief explicitly earns them.

## Color

- Warm cream background with terracotta or clay accent.
- Near-black background with single acid-green accent.
- Tinted near-black (`#0B0B0B`, `#111`) as the only dark.
- Gradient text or gradient washes used only as decoration.
- Single word in a headline highlighted with a different color.
- Glass and blur as decoration rather than a specific effect.
- Light or dark picked by category instead of the use scene.

## Typography

- All-caps labels.
- One word in a headline styled bold/italic differently from the rest.
- Unnecessary eyebrow labels or kickers above every heading.
- Tracking out uppercase labels.
- Monospace for small data labels just because it looks "tech."
- Arrow (`→`) appended to every link and button.
- System display faces (Impact, Arial Black, platform sans) as the display voice.

## Layout

- Identical rounded cards with the same soft shadow.
- One border-radius on everything, regardless of hierarchy.
- Numbered markers (`01 / 02 / 03`) for content that is not a sequence.
- Meta strings joined with middle dots: `A · B · C`.
- Labels built as `WORD — fragment` with a spaced em dash.
- The hero-metric template: big number, small label, supporting stats, accent.
- Cards of icon + heading + text as the page structure.
- A colored `border-left` or `border-right` above 1px on cards, list items, or callouts.
- Hard offset shadows outside an actual neobrutalist world.
- Sparklines, progress rings, and soft-shadowed rounded rectangles standing in for content.
- Geometric masks approximating a photographic subject's edge.

## Motion

- Fade-and-slide-up entrance on every section.
- Hover transitions on every card.
- Decorative parallax on every scroll.
- Confetti or bouncing elements for routine success states.
- The same animation on every repeated element.

## Browser and content

- Unicode glyphs or emoji standing in for a real icon system.
- Hover-only interactions with no touch equivalent.
- Placeholder copy or lorem ipsum left in the final surface.
- Perfect demo data as the only designed state.

## What to Do Instead

- Start from the subject matter and the brief.
- Make one bold, deliberate choice per surface.
- Use restraint. Remove one accessory before finishing.
- Design every state: hover, focus, active, disabled, loading, empty, error, success.
- Test the result against the brief: would another project look the same? If yes, revise.
