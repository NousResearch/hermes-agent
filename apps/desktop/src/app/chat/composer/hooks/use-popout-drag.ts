import { type PointerEvent as ReactPointerEvent, type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import {
  POPOUT_ESTIMATED_HEIGHT,
  POPOUT_WIDTH_REM,
  type PopoutPosition,
  type PopoutSize,
  readPopoutBounds,
  setComposerPopoutPosition
} from '@/store/composer-popout'

// Floating surface long-press before it becomes draggable (the 5px platform drags
// instantly; this only covers grabbing the composer body itself).
const LONG_PRESS_MS = 360
const LONG_PRESS_MOVE_TOLERANCE = 10
// Upward drag distance from the docked composer that peels it off into a float.
const PEEL_OUT_PX = 16
const DOCK_ZONE_BOTTOM_PX = 72
// How close the composer's center must be to the viewport center (px) to count as
// "over the dock". Kept tight so the bottom-left/right corners stay free.
const DOCK_ZONE_CENTER_TOLERANCE_PX = 150
// Falloff distances over which dock proximity ramps from 1 (in-zone) down to 0.
const DOCK_VERTICAL_FALLOFF_PX = 260
const DOCK_HORIZONTAL_FALLOFF_PX = 220

interface PressState {
  armed: boolean
  mode: 'dock' | 'float'
  pointerId: number
  startBottom: number
  startRight: number
  startX: number
  startY: number
}

interface ComposerPopoutGesturesOptions {
  composerRef: RefObject<HTMLFormElement | null>
  onDock: () => void
  onPopOut: () => void
  poppedOut: boolean
  position: PopoutPosition
}

function gestureTargetOk(target: EventTarget | null) {
  if (!(target instanceof Element)) {
    return false
  }

  return !target.closest('button, a, input, textarea, select, [role="menuitem"], [data-radix-popper-content-wrapper]')
}

/** Floating composer's 5px outer frame — grab here to drag without long-press. */
function isFloatDragPlatform(target: EventTarget | null) {
  if (!(target instanceof Element)) {
    return false
  }

  if (!target.closest('[data-slot="composer-root"][data-popped-out]')) {
    return false
  }

  if (target.closest('[data-slot="composer-surface"], [data-slot="composer-rich-input"]')) {
    return false
  }

  return gestureTargetOk(target)
}

/** 0 (far) → 1 (inside the dock zone). Drives both the dock glow and the
 *  release-to-dock test (which fires at proximity 1). */
function dockProximityOf(rect: DOMRect) {
  const horizontalDist = Math.abs(rect.left + rect.width / 2 - window.innerWidth / 2)
  const verticalGap = window.innerHeight - DOCK_ZONE_BOTTOM_PX - rect.bottom

  const v = verticalGap <= 0 ? 1 : Math.max(0, 1 - verticalGap / DOCK_VERTICAL_FALLOFF_PX)

  const h =
    horizontalDist <= DOCK_ZONE_CENTER_TOLERANCE_PX
      ? 1
      : Math.max(0, 1 - (horizontalDist - DOCK_ZONE_CENTER_TOLERANCE_PX) / DOCK_HORIZONTAL_FALLOFF_PX)

  return v * h
}

const clampOffset = (value: number, max: number) => Math.min(Math.max(0, value), max)

/** Fixed-position composer uses bottom/right insets; keep the grab point under the pointer. */
function popoutPositionUnderPointer(
  clientX: number,
  clientY: number,
  grabX: number,
  grabY: number,
  boxWidth: number,
  boxHeight: number
): PopoutPosition {
  return {
    bottom: window.innerHeight - clientY + grabY - boxHeight,
    right: window.innerWidth - clientX + grabX - boxWidth
  }
}

/**
 * Gesture pop-out / dock for the composer — fully gestural, no hold-to-toggle.
 *
 * Docked: drag the composer upward (off the dock) to peel it out into a float,
 * then keep dragging in the same motion.
 * Floating: drag the 5px frame to move instantly, or long-press the body then
 * drag; release over the bottom-center dock band to snap back in.
 */
export function useComposerPopoutGestures({
  composerRef,
  onDock,
  onPopOut,
  poppedOut,
  position
}: ComposerPopoutGesturesOptions) {
  const [dragging, setDragging] = useState(false)
  const [dockProximity, setDockProximity] = useState(0)

  const stateRef = useRef<PressState | null>(null)
  const timerRef = useRef<number | null>(null)
  const liveRef = useRef(position)
  liveRef.current = position

  const onPopOutRef = useRef(onPopOut)
  onPopOutRef.current = onPopOut

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const resetGesture = useCallback(() => {
    clearTimer()
    stateRef.current = null
    setDragging(false)
    setDockProximity(0)
  }, [clearTimer])

  const beginFloatDrag = useCallback(
    (state: PressState, clientX: number, clientY: number, next: PopoutPosition, size?: PopoutSize) => {
      clearTimer()
      const clamped = setComposerPopoutPosition(next, { area: readPopoutBounds(composerRef.current), size })
      liveRef.current = clamped

      state.mode = 'float'
      state.armed = true
      state.startBottom = clamped.bottom
      state.startRight = clamped.right
      state.startX = clientX
      state.startY = clientY

      setDragging(true)
    },
    [clearTimer, composerRef]
  )

  const peelOffFromDock = useCallback(
    (state: PressState, clientX: number, clientY: number) => {
      const composer = composerRef.current

      if (!composer) {
        return
      }

      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
      const rect = composer.getBoundingClientRect()
      const boxWidth = POPOUT_WIDTH_REM * rem
      const boxHeight = POPOUT_ESTIMATED_HEIGHT
      const grabX = clampOffset(state.startX - rect.left, boxWidth)
      const grabY = clampOffset(state.startY - rect.top, boxHeight)
      const next = popoutPositionUnderPointer(clientX, clientY, grabX, grabY, boxWidth, boxHeight)

      beginFloatDrag(state, clientX, clientY, next, { height: boxHeight, width: boxWidth })
      onPopOutRef.current()
    },
    [beginFloatDrag, composerRef]
  )

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (event.button !== 0 || !gestureTargetOk(event.target)) {
        return
      }

      // Floating: grabbing the 5px platform drags immediately.
      if (poppedOut && isFloatDragPlatform(event.target)) {
        stateRef.current = {
          armed: true,
          mode: 'float',
          pointerId: event.pointerId,
          startBottom: liveRef.current.bottom,
          startRight: liveRef.current.right,
          startX: event.clientX,
          startY: event.clientY
        }
        setDragging(true)

        return
      }

      stateRef.current = {
        armed: false,
        mode: poppedOut ? 'float' : 'dock',
        pointerId: event.pointerId,
        startBottom: liveRef.current.bottom,
        startRight: liveRef.current.right,
        startX: event.clientX,
        startY: event.clientY
      }

      clearTimer()

      // Docked has NO timer — pop-out is purely the upward peel gesture (handled
      // in pointermove). Floating arms a long-press to drag the body.
      if (poppedOut) {
        timerRef.current = window.setTimeout(() => {
          const state = stateRef.current

          if (!state || state.armed) {
            return
          }

          state.armed = true
          setDragging(true)
        }, LONG_PRESS_MS)
      }
    },
    [clearTimer, poppedOut]
  )

  useEffect(() => {
    // Coalesce drag updates to one per frame — pointermove can fire several times
    // between paints on high-Hz mice, and each update re-renders + clamps.
    let raf: number | null = null
    let pending: { x: number; y: number } | null = null

    const cancelRaf = () => {
      if (raf !== null) {
        cancelAnimationFrame(raf)
        raf = null
      }
    }

    const flush = () => {
      raf = null
      const state = stateRef.current

      if (!state?.armed || state.mode !== 'float' || !pending) {
        return
      }

      const composer = composerRef.current
      const size = composer ? { height: composer.offsetHeight, width: composer.offsetWidth } : undefined

      liveRef.current = setComposerPopoutPosition(
        {
          bottom: state.startBottom - (pending.y - state.startY),
          right: state.startRight - (pending.x - state.startX)
        },
        { area: readPopoutBounds(composer), size }
      )

      if (composer) {
        setDockProximity(dockProximityOf(composer.getBoundingClientRect()))
      }
    }

    const handleMove = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      // Pre-arm: cheap threshold checks run inline (no per-frame work yet).
      if (!state.armed) {
        const deltaX = event.clientX - state.startX
        const deltaY = event.clientY - state.startY

        if (state.mode === 'dock') {
          // Peel off only on a clear upward drag — not a sideways/down wiggle.
          if (-deltaY > PEEL_OUT_PX && -deltaY > Math.abs(deltaX)) {
            peelOffFromDock(state, event.clientX, event.clientY)
          } else if (Math.abs(deltaX) > PEEL_OUT_PX || deltaY > LONG_PRESS_MOVE_TOLERANCE) {
            resetGesture()
          }
        } else if (Math.abs(deltaX) > LONG_PRESS_MOVE_TOLERANCE || Math.abs(deltaY) > LONG_PRESS_MOVE_TOLERANCE) {
          // Float body long-press pending: movement cancels the hold.
          resetGesture()
        }

        return
      }

      if (state.mode !== 'float') {
        return
      }

      event.preventDefault()
      pending = { x: event.clientX, y: event.clientY }
      raf ??= requestAnimationFrame(flush)
    }

    const handleUp = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      cancelRaf()

      if (state.armed && state.mode === 'float') {
        const composer = composerRef.current
        const rect = composer?.getBoundingClientRect()

        if (rect && dockProximityOf(rect) >= 1) {
          onDock()
        } else {
          // Persist the resting position once, on release — never per move.
          const size = composer ? { height: composer.offsetHeight, width: composer.offsetWidth } : undefined
          setComposerPopoutPosition(liveRef.current, { area: readPopoutBounds(composer), persist: true, size })
        }
      }

      resetGesture()
    }

    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', handleUp)
    window.addEventListener('pointercancel', handleUp)

    return () => {
      cancelRaf()
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', handleUp)
      window.removeEventListener('pointercancel', handleUp)
    }
  }, [composerRef, onDock, peelOffFromDock, resetGesture])

  useEffect(() => clearTimer, [clearTimer])

  return { dockProximity, dragging, onPointerDown }
}
