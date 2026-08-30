type MainWindowLike = {
  isDestroyed: () => boolean
}

/**
 * Stop-gap lifecycle policy: closing the final Desktop window means quitting
 * the app, including on macOS. Quitting enters the existing backend shutdown
 * coordinator, which owns the app-spawned CLI/serve trees and cron scheduler.
 */
export function shouldQuitAfterWindowAllClosed(): true {
  return true
}

type EnsureMainWindowOptions<T extends MainWindowLike> = {
  isReady: boolean
  createWindow: () => unknown
  focusWindow: (window: T) => unknown
  focusExisting?: boolean
}

export function ensureMainWindow<T extends MainWindowLike>(
  window: T | null | undefined,
  { isReady, createWindow, focusWindow, focusExisting = true }: EnsureMainWindowOptions<T>
) {
  if (!window || window.isDestroyed()) {
    // a closed electron window stays truthy, so replace it before invoking native methods.
    if (isReady) {
      createWindow()
    }

    return
  }

  if (focusExisting) {
    focusWindow(window)
  }
}
