type MainWindowLike = {
  isDestroyed: () => boolean
}

type DrainWindowLike = MainWindowLike & {
  destroy: () => unknown
}

/** Destroy live renderer windows before an asynchronous backend shutdown drain. */
export function closeWindowsForDrain(windows: DrainWindowLike[]): number {
  let closed = 0
  for (const window of windows) {
    if (window.isDestroyed()) {
      continue
    }
    window.destroy()
    closed += 1
  }
  return closed
}

/** Closing the final desktop window enters the normal app shutdown coordinator. */
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
