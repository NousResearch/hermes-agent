/** Validate the raw payload before locating, focusing, or acquiring native input. */
export function previewDragError(action: Record<string, unknown>, verbKey: 'kind' | 'action' = 'kind'): string | undefined {
  if (action[verbKey] !== 'drag') {
    return 'dx' in action || 'dy' in action ? 'dx and dy are only valid for drag.' : undefined
  }

  if (Object.keys(action).some(key => ![verbKey, 'ref', 'selector', 'dx', 'dy'].includes(key))) {
    return 'drag accepts only kind, exactly one ref or selector, dx and dy.'
  }

  const targets = ['ref', 'selector'].filter(key => key in action)
  const target = action[targets[0]]

  if (targets.length !== 1 || typeof target !== 'string' || !target.trim()) {
    return 'drag needs exactly one non-empty ref or selector.'
  }

  for (const delta of [action.dx, action.dy]) {
    if (typeof delta !== 'number' || !Number.isFinite(delta) || Math.abs(delta) > 2000) {
      return 'drag dx and dy must be finite numbers within -2000 to 2000 CSS pixels.'
    }
  }

  return action.dx === 0 && action.dy === 0 ? 'drag needs a non-zero movement.' : undefined
}
