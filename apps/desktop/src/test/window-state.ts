// Stand-ins for the two window-visibility signals the renderer reacts to:
// Electron's `hermesDesktop` bridge — both the change push and the
// `getWindowState()` pull that seeds a controller in an already-hidden window
// — and the DOM's own `document.hidden`. Both are read-only under jsdom, and
// anything that pauses work while the window is hidden — the pet, the terminal
// pane, budgeted loops, pulse animations — needs to drive them.

import { vi } from 'vitest'

export interface WindowStatePayload {
  isMinimized?: boolean
  isVisible?: boolean
}

export interface WindowStateBridge {
  /** Push a state change to whoever subscribed through the bridge. */
  emit: (payload: WindowStatePayload) => void
  /** The unsubscribe handed back to the subscriber, for teardown assertions. */
  off: ReturnType<typeof vi.fn>
  /** Resolve the pending `getWindowState()` pull(s) with a snapshot. Until
   *  called, the pull hangs on purpose: that models a controller whose seed
   *  never landed, so tests that do not know about the pull keep the old
   *  push-only behaviour. */
  resolvePull: (payload: WindowStatePayload) => void
}

/** Install a fake `window.hermesDesktop` window-state channel. */
export function installWindowStateBridge(): WindowStateBridge {
  let callback: ((payload: WindowStatePayload) => void) | null = null
  let pendingPulls: Array<(payload: WindowStatePayload) => void> = []

  const off = vi.fn(() => {
    callback = null
  })

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      onWindowStateChanged: vi.fn((next: (payload: WindowStatePayload) => void) => {
        callback = next

        return off
      }),
      getWindowState: vi.fn(() => new Promise<WindowStatePayload>(resolve => pendingPulls.push(resolve)))
    }
  })

  return {
    emit: payload => callback?.(payload),
    off,
    resolvePull: payload => {
      const pending = pendingPulls
      pendingPulls = []

      for (const resolve of pending) {
        resolve(payload)
      }
    }
  }
}

/** Drive `document.hidden` / `document.visibilityState` together. */
export function setDocumentHidden(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: hidden ? 'hidden' : 'visible' })
}
