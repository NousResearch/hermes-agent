# Polish — Final Quality Pass

`polish` is the last review before the interface is considered ready. It is refinement, not concealed redesign. Preserve the incumbent visual world, content, behavior, and everything outside scope.

## Triage

Fix in this order:

1. Broken or blocked tasks, data loss, misleading state, and inaccessible paths.
2. Missing loading, empty, error, success, disabled, and permission states.
3. Flow, hierarchy, responsive, and design-system drift.
4. Visual and motion inconsistencies.
5. Code and asset cleanup.

## Checklist

### Layout and type

- [ ] The squint test still reveals the primary, secondary, and major groups.
- [ ] Reading and task path are clear at every supported size.
- [ ] Related content groups naturally; unrelated content does not blur together.
- [ ] Type follows a deliberate role scale; repeated roles stay identical across screens.
- [ ] Body measure is 45–75ch; line height is tuned to the face and width.
- [ ] Loading fonts does not cause disruptive reflow or invisible text.

### Color, imagery, and icons

- [ ] Semantic tokens and stable color meanings across themes.
- [ ] Text, control, and focus contrast pass in every state.
- [ ] Icon families, stroke, weight, and optical alignment are coherent.
- [ ] Images use correct aspect ratios, responsive sources, and useful alt text.

### Interaction and state

- [ ] Every control has default, hover, focus, active, disabled, loading, error, and success states.
- [ ] Keyboard focus is visible and the tab order is logical.
- [ ] Touch targets are ≥ 44x44px and not too close to each other.
- [ ] Motion is coherent, interruptible, and reduced-motion safe.
- [ ] Long, missing, localized, offline, slow, and permission-limited content still works.

### Responsive

- [ ] Mobile view at 375px is the strongest story.
- [ ] Layout shifts are meaningful at each breakpoint.
- [ ] Images and media do not overflow or distort.
- [ ] Zoom to 200% keeps the interface usable.

### Framework and delivery

- [ ] Component boundaries match the target framework's conventions.
- [ ] Design tokens are implemented consistently.
- [ ] No framework defaults are passed off as brand design.
- [ ] No dead code, unused imports, or orphaned styles were introduced.

## Stop condition

Stop after the second full pass. Additional rounds usually introduce noise, not quality.
