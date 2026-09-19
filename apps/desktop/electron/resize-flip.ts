// Shared "briefly flip resizable for setBounds" dance (Vicky review: third
// hand-rolled copy in plugin-overlay-ipc.ts, and it lacked the try/finally
// the HUD version has). Transparent frameless windows are created
// non-resizable (no stray edge-drag hot-zone), which on Windows/Linux also
// blocks programmatic setBounds SIZING — so the window is flipped resizable
// for the duration of the call. The try/finally guarantees the resize lock
// comes back even when setBounds throws (window destroyed between
// validation and the native call).

export interface ResizeFlipWindow {
  getSize(): number[]
  isDestroyed(): boolean
  isResizable(): boolean
  setBounds(bounds: { x: number; y: number; width: number; height: number }): void
  setResizable(resizable: boolean): void
}

export interface ResizeFlipBounds {
  x: number
  y: number
  width: number
  height: number
}

/** Apply bounds with the resize flip. The flip only happens when the SIZE
 *  actually changes (a pure move never touches resizable state — same
 *  semantics as the pet/hud call sites). Returns false when the window
 *  vanished or setBounds threw. */
export function applyBoundsWithResizeFlip(win: ResizeFlipWindow, bounds: ResizeFlipBounds): boolean {
  if (win.isDestroyed()) {
    return false
  }

  try {
    const [curW, curH] = win.getSize()
    const resizing = Math.round(bounds.width) !== curW || Math.round(bounds.height) !== curH
    const restoreLock = resizing && !win.isResizable()

    if (restoreLock) {
      win.setResizable(true)
    }

    try {
      win.setBounds(bounds)
    } finally {
      if (restoreLock && !win.isDestroyed()) {
        win.setResizable(false)
      }
    }

    return true
  } catch {
    return false
  }
}
