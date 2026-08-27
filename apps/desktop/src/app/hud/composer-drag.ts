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
  mode: 'control' | 'hold'
  originH: number
  originW: number
  pointerId: number
  startX: number
  startY: number
  target: HTMLElement
  workspaceTransfer: boolean
}

interface HudComposerDragOptions {
  /** X11 escape hatch: Ctrl+primary-button grabs immediately instead of
   *  competing with Chromium's text-selection drag until the hold timer. */
  controlDrag?: boolean
  /** X11/KWin only: keep the grabbed window visible while the user changes
   *  virtual desktops, then pin it to the destination desktop on release. */
  workspaceTransfer?: boolean
}

function capturePointer(state: PressState): void {
  try {
    state.target.setPointerCapture?.(state.pointerId)
  } catch {
    // A renderer can reject capture after selection/native-drag bookkeeping.
    // Window capture-phase listeners below still keep the in-window gesture
    // alive, so a failed capture must not abort the drag altogether.
  }
}

function releasePointer(state: PressState): void {
  try {
    if (state.target.hasPointerCapture?.(state.pointerId)) {
      state.target.releasePointerCapture?.(state.pointerId)
    }
  } catch {
    // Pointer cancellation may invalidate the id before React cleans up.
  }
}

function setWorkspaceTransfer(transferring: boolean): void {
  window.hermesDesktop?.hud?.setWorkspaceTransfer?.(transferring)
}

/**
 * HUD-only: press and hold the composer, then drag to move the window. On X11,
 * Ctrl+primary-button is an immediate grab that also works over selected text.
 *
 * The way to move the HUD on macOS/Windows. An `-webkit-app-region: drag`
 * handle is the obvious alternative and cannot work THERE: the window manager
 * takes a drag region's mouse input whole, which starves `useHudClickThrough`
 * of the moves it decides solidity from, so the window is already transparent
 * by the time the press lands and it falls through to the app behind. See
 * click-through.ts. Native Wayland uses a compositor drag region instead,
 * because apps cannot place their own top-level surfaces there. X11 stays on
 * this renderer path and additionally supports an immediate Ctrl-drag.
 *
 * Deltas are read in SCREEN coordinates. Client coordinates are relative to the
 * window we are moving, so a window that keeps up with the cursor reports the
 * same clientX every frame — zero delta, and the drag dies one pixel in.
 *
 * The size is snapshotted at press and sent with every move, so main can pin it
 * (see hermes:hud:move-by — a transparent frameless window drifts wider on
 * Windows otherwise). Same shape as the pet overlay's drag.
 */
export function useHudComposerDrag(
  enabled: boolean,
  { controlDrag = false, workspaceTransfer = false }: HudComposerDragOptions = {}
) {
  const [grabbing, setGrabbing] = useState(false)
  const stateRef = useRef<PressState | null>(null)
  const timerRef = useRef<number | null>(null)

  const reset = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
      timerRef.current = null
    }

    const state = stateRef.current

    if (state) {
      if (state.workspaceTransfer) {
        setWorkspaceTransfer(false)
      }

      releasePointer(state)
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
      const immediate = controlDrag && event.ctrlKey

      // A press over an existing contentEditable selection otherwise starts
      // Chromium's native text drag, which cancels our pointer stream. Cancel
      // that default action before it is chosen; do not blur or rewrite the
      // Selection, so the user's selected text survives moving the window.
      if (immediate) {
        event.preventDefault()
      }

      const state: PressState = {
        armed: immediate,
        lastX: event.screenX,
        lastY: event.screenY,
        mode: immediate ? 'control' : 'hold',
        originH: window.outerHeight,
        originW: window.outerWidth,
        pointerId: event.pointerId,
        startX: event.screenX,
        startY: event.screenY,
        target,
        workspaceTransfer: false
      }

      stateRef.current = state

      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }

      if (immediate) {
        state.workspaceTransfer = workspaceTransfer

        if (workspaceTransfer) {
          setWorkspaceTransfer(true)
        }

        setGrabbing(true)
        triggerHaptic('selection')
        capturePointer(state)

        return
      }

      timerRef.current = window.setTimeout(() => {
        const state = stateRef.current

        if (!state || state.armed) {
          return
        }

        state.armed = true
        state.workspaceTransfer = workspaceTransfer

        if (workspaceTransfer) {
          setWorkspaceTransfer(true)
        }

        setGrabbing(true)
        triggerHaptic('selection')

        // Capture so the moves keep arriving once the cursor outruns the bar,
        // and drop the caret so the drag isn't also extending a selection.
        capturePointer(state)

        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur()
        }
      }, LONG_PRESS_MS)
    },
    [controlDrag, enabled, workspaceTransfer]
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

      window.hermesDesktop?.hud?.moveBy?.({
        x: dx,
        y: dy,
        width: state.originW,
        height: state.originH
      })
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

    const preventEditorGesture = (event: Event) => {
      if (stateRef.current?.mode === 'control') {
        event.preventDefault()
      }
    }

    // Capture phase wins the race against contentEditable selection/drag
    // handlers. `dragstart` is a second guard for an already-selected range;
    // `selectstart` stops the same press from replacing that range.
    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
    window.addEventListener('pointercancel', onUp, true)
    window.addEventListener('dragstart', preventEditorGesture, true)
    window.addEventListener('selectstart', preventEditorGesture, true)

    return () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
      window.removeEventListener('pointercancel', onUp, true)
      window.removeEventListener('dragstart', preventEditorGesture, true)
      window.removeEventListener('selectstart', preventEditorGesture, true)
    }
  }, [enabled, reset])

  useEffect(() => reset, [reset])

  return { grabbing, onPointerDown }
}
