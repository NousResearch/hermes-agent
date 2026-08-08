import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from 'react'

import { triggerHaptic } from '@/lib/haptics'

/** Hold anywhere on the HUD composer before the bar becomes draggable. Short
 *  enough to feel like grabbing it, long enough that a click still clicks. */
const LONG_PRESS_MS = 140
/** Slop before the hold is read as a text selection instead of a grab. */
const MOVE_TOLERANCE = 8

interface PressState {
  armed: boolean
  lastX: number
  lastY: number
  pointerId: number
  startX: number
  startY: number
  target: HTMLElement
}

/**
 * HUD-only: press and hold the composer, then drag to move the window.
 *
 * The only way to move the HUD. An `-webkit-app-region: drag` handle is the
 * obvious alternative and cannot work here: the window manager takes a drag
 * region's mouse input whole, which starves `useHudClickThrough` of the moves
 * it decides solidity from, so the window is already transparent by the time
 * the press lands and it falls through to the app behind. See click-through.ts.
 *
 * Deltas are read in SCREEN coordinates. Client coordinates are relative to the
 * window we are moving, so a window that keeps up with the cursor reports the
 * same clientX every frame — zero delta, and the drag dies one pixel in.
 */
export function useHudComposerDrag(enabled: boolean) {
  const [grabbing, setGrabbing] = useState(false)
  const stateRef = useRef<PressState | null>(null)
  const timerRef = useRef<number | null>(null)

  const reset = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }

    const state = stateRef.current

    if (state?.target.hasPointerCapture(state.pointerId)) {
      state.target.releasePointerCapture(state.pointerId)
    }

    stateRef.current = null
    setGrabbing(false)
  }, [])

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (!enabled || event.button !== 0) {
        return
      }

      const target = event.currentTarget

      stateRef.current = {
        armed: false,
        lastX: event.screenX,
        lastY: event.screenY,
        pointerId: event.pointerId,
        startX: event.screenX,
        startY: event.screenY,
        target
      }

      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }

      timerRef.current = window.setTimeout(() => {
        const state = stateRef.current

        if (!state || state.armed) {
          return
        }

        state.armed = true
        setGrabbing(true)
        triggerHaptic('selection')

        // Capture so the moves keep arriving once the cursor outruns the bar,
        // and drop the caret so the drag isn't also extending a selection.
        state.target.setPointerCapture(state.pointerId)

        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur()
        }
      }, LONG_PRESS_MS)
    },
    [enabled]
  )

  useEffect(() => {
    if (!enabled) {
      return
    }

    const onMove = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      if (!state.armed) {
        if (
          Math.abs(event.screenX - state.startX) > MOVE_TOLERANCE ||
          Math.abs(event.screenY - state.startY) > MOVE_TOLERANCE
        ) {
          reset()
        }

        return
      }

      event.preventDefault()

      const dx = event.screenX - state.lastX
      const dy = event.screenY - state.lastY

      state.lastX = event.screenX
      state.lastY = event.screenY

      window.hermesDesktop?.hud?.moveBy?.({ x: dx, y: dy })
    }

    const onUp = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      // A completed hold is a grab, not a click on whatever was underneath it.
      if (state.armed) {
        window.addEventListener('click', click => click.stopPropagation(), { capture: true, once: true })
      }

      reset()
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)

    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [enabled, reset])

  useEffect(() => reset, [reset])

  return { grabbing, onPointerDown }
}
