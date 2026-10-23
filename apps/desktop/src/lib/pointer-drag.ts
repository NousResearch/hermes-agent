/** Run a pointer drag until pointerup, then unbind both listeners.
 *
 *  Capture-phase, on `window`, so the drag keeps tracking when the pointer
 *  leaves the element it started on. Forgetting one half of the teardown leaks
 *  a live pointermove handler for the rest of the session, which is why the
 *  unbinding lives here rather than at each call site. */
export function startPointerDrag(
  onMove: (event: PointerEvent) => void,
  onEnd?: (event: PointerEvent) => void
): () => void {
  const move = (event: PointerEvent) => onMove(event)

  const up = (event: PointerEvent) => {
    window.removeEventListener('pointermove', move, true)
    window.removeEventListener('pointerup', up, true)
    onEnd?.(event)
  }

  window.addEventListener('pointermove', move, true)
  window.addEventListener('pointerup', up, true)

  return () => {
    window.removeEventListener('pointermove', move, true)
    window.removeEventListener('pointerup', up, true)
  }
}
