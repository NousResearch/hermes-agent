import type * as React from 'react'

/** `MouseEvent.button` for the middle (wheel) button. */
const MIDDLE_BUTTON = 1

/** Where the current middle press started. One pointer holds one button, so a
 *  single slot is the whole state, and it's only ever compared by identity in
 *  the pointerup right after — a value left behind by a press released
 *  elsewhere is inert, not stale. */
let pressedOn: EventTarget | null = null

/**
 * Middle-click as a gesture that survives a real three-button mouse.
 *
 * `auxclick` is the obvious event and the wrong one to build on. Windows and
 * Linux Chromium answer a middle press inside a scroller by starting the
 * AUTOSCROLL pan, and the mouseup that ends the pan is spent stopping it
 * instead of completing a click — so `auxclick` never arrives. Every surface
 * carrying this gesture (tab strips, the session list, the terminal rail) is a
 * scroller, which is why it only ever worked on macOS, where autoscroll
 * doesn't exist.
 *
 * Pointer events fire either way, so the gesture arms on pointerdown and is
 * spent on the pointerup over the SAME element — press one tab, release on
 * another and nothing happens (Chrome / VS Code semantics). mousedown's default
 * dies on every middle press, action or not, so the pan widget can't appear on
 * a surface that owns the button.
 *
 * A plain factory, not a hook: tab strips call it inside `map()`.
 */
export function middleClickHandlers(action: (() => void) | undefined) {
  return {
    onMouseDown: (event: React.MouseEvent) => {
      if (event.button === MIDDLE_BUTTON) {
        event.preventDefault()
      }
    },

    onPointerDown: (event: React.PointerEvent) => {
      if (event.button === MIDDLE_BUTTON) {
        pressedOn = action ? event.currentTarget : null
      }
    },

    onPointerUp: (event: React.PointerEvent) => {
      if (event.button !== MIDDLE_BUTTON) {
        return
      }

      const armed = pressedOn === event.currentTarget
      pressedOn = null

      if (armed) {
        action?.()
      }
    }
  }
}
