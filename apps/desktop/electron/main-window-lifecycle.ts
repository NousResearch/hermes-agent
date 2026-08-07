type MainWindowLike = {
  isDestroyed: () => boolean
}

/**
 * What an activation actually did. Returned so the caller can tell an
 * *effective* activation from one that could not do anything: a relaunch during
 * a slow first start has no window to focus and must not create a second one, so
 * without a signal the click is swallowed and the user clicks again (and again).
 */
export type EnsureMainWindowOutcome = 'focused' | 'created' | 'starting' | 'not-ready' | 'deep-link'

type EnsureMainWindowOptions<T extends MainWindowLike> = {
  isReady: boolean
  createWindow: () => unknown
  focusWindow: (window: T) => unknown
  focusExisting?: boolean
  /** First launch is still bringing its backend up; no window exists yet. */
  starting?: boolean
}

export function ensureMainWindow<T extends MainWindowLike>(
  window: T | null | undefined,
  { isReady, createWindow, focusWindow, focusExisting = true, starting = false }: EnsureMainWindowOptions<T>
): EnsureMainWindowOutcome {
  if (!window || window.isDestroyed()) {
    // a closed electron window stays truthy, so replace it before invoking native methods.
    if (!isReady) {
      return 'not-ready'
    }

    // A launch already in flight owns window creation. Duplicating it here would
    // race two boot sequences against the same backend port.
    if (starting) {
      return 'starting'
    }

    createWindow()

    return 'created'
  }

  if (!focusExisting) {
    return 'deep-link'
  }

  focusWindow(window)

  return 'focused'
}
