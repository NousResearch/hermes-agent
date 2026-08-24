import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useRef, useState } from 'react'

/** Clamp to the same mins the window was created with (spawnHudWindow). */
const HUD_MIN_WIDTH = 380
const HUD_MIN_HEIGHT = 160

export const HUD_RESIZE_DIRECTIONS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'] as const

export type HudResizeDirection = (typeof HUD_RESIZE_DIRECTIONS)[number]

const HUD_WAYLAND_RESIZE_DIRECTIONS = ['e', 'se', 's'] as const satisfies readonly HudResizeDirection[]

/** Native Wayland cannot read or change a top-level window's global x/y, so
 *  only edges that preserve the existing top-left position are truthful. */
export function hudResizeDirections(nativeDrag: boolean): readonly HudResizeDirection[] {
  return nativeDrag ? HUD_WAYLAND_RESIZE_DIRECTIONS : HUD_RESIZE_DIRECTIONS
}

export interface HudResizeBounds {
  height: number
  width: number
  x: number
  y: number
}

interface ResizeState {
  direction: HudResizeDirection
  startX: number
  startY: number
  originX: number
  originY: number
  originW: number
  originH: number
  pointerId: number
  target: HTMLElement
}

/** Resize one or two anchored edges while preserving every opposite edge. */
export function hudResizeBounds(
  origin: HudResizeBounds,
  direction: HudResizeDirection,
  dx: number,
  dy: number
): HudResizeBounds {
  const right = origin.x + origin.width
  const bottom = origin.y + origin.height
  const west = direction.includes('w')
  const east = direction.includes('e')
  const north = direction.includes('n')
  const south = direction.includes('s')

  const width = west
    ? Math.max(HUD_MIN_WIDTH, origin.width - dx)
    : east
      ? Math.max(HUD_MIN_WIDTH, origin.width + dx)
      : origin.width

  const height = north
    ? Math.max(HUD_MIN_HEIGHT, origin.height - dy)
    : south
      ? Math.max(HUD_MIN_HEIGHT, origin.height + dy)
      : origin.height

  return {
    x: west ? right - width : origin.x,
    y: north ? bottom - height : origin.y,
    width,
    height
  }
}

function capturePointer(state: ResizeState): void {
  try {
    state.target.setPointerCapture?.(state.pointerId)
  } catch {
    // Window capture-phase listeners below keep the in-window gesture alive.
  }
}

function releasePointer(state: ResizeState): void {
  try {
    if (state.target.hasPointerCapture?.(state.pointerId)) {
      state.target.releasePointerCapture?.(state.pointerId)
    }
  } catch {
    // Pointer cancellation may invalidate the id before cleanup.
  }
}

/**
 * HUD-only: drag any edge or corner handle to resize the window.
 *
 * The window is created `resizable: false` (see spawnHudWindow — a transparent
 * frameless window must not expose a system resize hot-zone, or every drag
 * grows it), so resizing has to be programmatic: the handle reports absolute
 * screen bounds and main flips resizable on for the setBounds call. Same
 * pattern as the pet overlay's wheel-scale (`hermes:pet-overlay:set-bounds`).
 *
 * Each handle preserves its opposite edge, matching ordinary desktop windows
 * and CanvasTTY cards. Deltas are read in SCREEN coordinates, like the composer
 * drag: client coordinates are relative to a window that is changing position
 * and size, so they cannot be trusted mid-resize.
 */
export function useHudResizeHandle(): {
  resizing: boolean
  onPointerDown: (event: ReactPointerEvent<HTMLElement>, direction: HudResizeDirection) => void
} {
  const [resizing, setResizing] = useState(false)
  const stateRef = useRef<ResizeState | null>(null)

  const reset = useCallback(() => {
    const state = stateRef.current

    if (state) {
      releasePointer(state)
    }

    stateRef.current = null
    setResizing(false)
  }, [])

  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLElement>, direction: HudResizeDirection) => {
    if (event.button !== 0) {
      return
    }

    stateRef.current = {
      direction,
      startX: event.screenX,
      startY: event.screenY,
      originX: window.screenX,
      originY: window.screenY,
      originW: window.outerWidth,
      originH: window.outerHeight,
      pointerId: event.pointerId,
      target: event.currentTarget
    }

    setResizing(true)
    capturePointer(stateRef.current)
    event.preventDefault()
    event.stopPropagation()
  }, [])

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      event.preventDefault()

      const dx = event.screenX - state.startX
      const dy = event.screenY - state.startY

      window.hermesDesktop?.hud?.setBounds?.(
        hudResizeBounds(
          { x: state.originX, y: state.originY, width: state.originW, height: state.originH },
          state.direction,
          dx,
          dy
        )
      )
    }

    const onUp = (event: PointerEvent) => {
      const state = stateRef.current

      if (!state || event.pointerId !== state.pointerId) {
        return
      }

      reset()
    }

    window.addEventListener('pointermove', onMove, true)
    window.addEventListener('pointerup', onUp, true)
    window.addEventListener('pointercancel', onUp, true)

    return () => {
      window.removeEventListener('pointermove', onMove, true)
      window.removeEventListener('pointerup', onUp, true)
      window.removeEventListener('pointercancel', onUp, true)
    }
  }, [reset])

  // A resize interrupted by an unmount must not leave the state dangling.
  useEffect(() => reset, [reset])

  return { resizing, onPointerDown }
}
