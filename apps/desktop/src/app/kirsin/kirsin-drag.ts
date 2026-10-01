import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from 'react'

/**
 * Kirsin header drag — move the persistent panel by grabbing its header.
 *
 * Deliberately simpler than the HUD's {@link useHudComposerDrag}. The HUD is a
 * CLICK-THROUGH transparent band hovering over other apps, so a press must not
 * immediately start a drag (hence its long-press hold, the X11 Ctrl-grab, and
 * the workspace-transfer escape hatch). The Kirsin window is a SOLID,
 * non-click-through panel whose header carries no editable text, so a plain
 * primary-button press-and-drag is the correct, expected title-bar behavior —
 * a click on the (non-button) header is a grab, and the three header buttons
 * stopPropagation on pointerdown so they never start one.
 *
 * It drives the same IPC the HUD uses (begin-move / move-by / end-move, see
 * electron/kirsin-ipc.ts): the renderer snapshots the window size at press and
 * re-pins it on every move (a transparent frameless window drifts wider on
 * Windows otherwise), and main samples the native cursor to park the window at
 * cursor-minus-grab-offset (see electron/hud-drag.ts).
 */
export function useKirsinHeaderDrag() {
  const [dragging, setDragging] = useState(false)
  const stateRef = useRef<{ height: number; width: number; pointerId: number } | null>(null)
  const targetRef = useRef<HTMLElement | null>(null)

  const end = useCallback((sendEndMove: boolean) => {
    const state = stateRef.current

    if (!state) {
      return
    }

    if (sendEndMove) {
      window.hermesDesktop?.kirsin?.endMove?.()
    }

    try {
      if (targetRef.current?.hasPointerCapture?.(state.pointerId)) {
        targetRef.current.releasePointerCapture?.(state.pointerId)
      }
    } catch {
      // Pointer cancellation may invalidate the id before React cleans up.
    }

    stateRef.current = null
    targetRef.current = null
    setDragging(false)
  }, [])

  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) {
      return
    }

    // The header buttons stopPropagation on pointerdown, so a press on one
    // never starts a drag; the closest() guard is belt-and-suspenders. A
    // gesture already in flight (the orb owns its own pointer capture) must
    // not open a second grab session either.
    if ((event.target as HTMLElement).closest?.('[data-kirsin-ctl]') || stateRef.current) {
      return
    }

    const state = { width: window.outerWidth, height: window.outerHeight, pointerId: event.pointerId }

    stateRef.current = state
    targetRef.current = event.currentTarget

    try {
      // Capture so the moves keep arriving once the cursor outruns the header.
      event.currentTarget.setPointerCapture?.(state.pointerId)
    } catch {
      // A renderer can reject capture; the window-capture listeners below keep
      // the gesture alive, so a failed capture must not abort the drag.
    }

    window.hermesDesktop?.kirsin?.beginMove?.()
    setDragging(true)
  }, [])

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      event.preventDefault()
      window.hermesDesktop?.kirsin?.moveBy?.({ width: state.width, height: state.height })
    }

    const onUp = (event: PointerEvent | MouseEvent) => {
      const state = stateRef.current
      const pointerId = 'pointerId' in event ? event.pointerId : state?.pointerId

      if (!state || pointerId !== state.pointerId) {
        return
      }

      end(true)
    }

    const onCancel = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      // Crossing a display often fires pointercancel without a matching up;
      // ending the grab there parks the window on the first monitor. Re-pin the
      // size and keep the hold so the next move (or the mouseup) finishes the
      // drag on the other display.
      event.preventDefault()
      window.hermesDesktop?.kirsin?.moveBy?.({ width: state.width, height: state.height })

      try {
        targetRef.current?.setPointerCapture?.(state.pointerId)
      } catch {
        // ignore — the listeners keep the gesture alive regardless
      }
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
    window.addEventListener('mouseup', onUp, true)
    window.addEventListener('pointercancel', onCancel, true)

    return () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
      window.removeEventListener('mouseup', onUp, true)
      window.removeEventListener('pointercancel', onCancel, true)
    }
  }, [end])

  // A drag in flight at unmount must not leave main's grab session open.
  useEffect(() => () => end(false), [end])

  return { dragging, onPointerDown }
}
