# Hit Areas

Target sizes, expanding hit areas without changing visual size, and collision rules.

## Target sizes

Separate the conformance baseline from larger usability targets:

| Standard | Minimum |
| --- | --- |
| WCAG 2.5.8 (AA) | 24×24px, the hard floor |
| WCAG 2.5.5 (AAA) | 44×44px |
| Apple HIG | 44×44pt |
| Material Design | 48×48dp |

WCAG 2.5.8 Level AA requires a 24×24 CSS-pixel target or one of its defined exceptions. Treat 44px as a recommended touch target for primary controls and 40px as a useful desktop target when the product's density permits. Smaller controls are not automatically failures: check the spacing, equivalent-control, inline, user-agent, and essential exceptions before reporting one.

Under the spacing exception, an undersized target passes if a 24px circle centered on its bounding box does not intersect another target or another undersized target's circle; in the simple case, 20px targets need at least a 4px gap.

The visible element can stay small; the hit area is what must be big. If it looks clickable, it must be clickable across its whole visual extent: no dead zones (a checkbox and its label share one hit target).

## Expanding the hit area

If the visible element is smaller (e.g., a 20×20 checkbox), extend the hit area with a pseudo-element. Put the pseudo-element on the wrapping `<label>` or `<button>`, not on the `<input>` itself; replaced elements don't render `::before`/`::after` reliably.

### CSS Example

```css
/* Small checkbox with expanded 44px hit area, on the wrapping label */
.checkbox-label {
  position: relative;
  width: 20px;
  height: 20px;
}

.checkbox-label::after {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%; /* physical centering: direction-independent */
  transform: translate(-50%, -50%);
  width: 44px;
  height: 44px;
}
```

### Tailwind Example

```tsx
<button className="relative size-5 after:absolute after:top-1/2 after:left-1/2 after:size-11 after:-translate-1/2">
  <CheckIcon />
</button>
```

### Layout alternative

When the element can afford real box size, skip the pseudo-element and let the box itself be the target; this also gives the browser the real geometry for scrolling and gestures:

```css
.icon-button {
  min-width: 44px;
  min-height: 44px;
  display: inline-grid;
  place-items: center;
}
```

## Collision Rule

If the extended hit area overlaps another interactive element, shrink the pseudo-element, but make it as large as possible without colliding. Two interactive elements should never have overlapping hit areas.

## Touch behavior

- Add `touch-action: manipulation` to interactive elements to remove the double-tap-to-zoom delay on mobile.
- Set `-webkit-tap-highlight-color` to match the design instead of the default gray flash.
- Prefer generous targets and clear affordances over finicky interactions (tiny drag handles, precise hover zones).
