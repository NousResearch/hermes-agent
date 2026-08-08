type MainWindowLike = {
  isDestroyed: () => boolean
}

type FocusableWindowLike = MainWindowLike & {
  focus: () => void
  isMinimized: () => boolean
  isVisible: () => boolean
  restore: () => void
  show: () => void
}

type DeepLinkWindowLike = FocusableWindowLike & {
  webContents: {
    send: (channel: string, payload: unknown) => void
  }
}

type EnsureMainWindowOptions<T extends MainWindowLike> = {
  isReady: boolean
  createWindow: () => unknown
  focusWindow: (window: T) => unknown
  focusExisting?: boolean
}

export function focusMainWindow(window: FocusableWindowLike | null | undefined) {
  if (!window || window.isDestroyed()) {
    return false
  }

  if (window.isMinimized()) {
    window.restore()
  }

  if (!window.isVisible()) {
    window.show()
  }

  window.focus()

  return true
}

export function deliverDeepLink(window: DeepLinkWindowLike | null | undefined, payload: unknown) {
  if (!focusMainWindow(window)) {
    return false
  }

  window.webContents.send('hermes:deep-link', payload)

  return true
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
