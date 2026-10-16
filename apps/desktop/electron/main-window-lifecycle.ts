import { shouldFocusToTakeKeyboard } from './window-focus-policy'

type MainWindowLike = {
  isDestroyed: () => boolean
}

type ActivatableWindow = MainWindowLike & {
  isMinimized: () => boolean
  isVisible: () => boolean
  isFocused: () => boolean
  restore: () => void
  show: () => void
  focus: () => void
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

/**
 * Whether the last-chat-window `closed` fallback must quit the app (#130810).
 *
 * Hidden helpers (Quick Entry, HUD, pet overlay) are still BrowserWindows,
 * so `window-all-closed` never fires while one lingers. This fallback quits
 * when no chat surface remains. It is keyed on whether a quit is actually in
 * progress (`quitInProgress`), never on the overlay-suppression latch
 * (`appQuitting`): on Windows/Linux an ordinary primary-window close sets
 * `appQuitting = true` in its `close` handler (for #55920 pop-in
 * suppression) before `closed` fires, so gating on that latch would suppress
 * the very quit the fallback exists for and strand a windowless
 * single-instance lock holder. A tray-absorbed close never reaches the
 * fallback (preventDefault) and a multi-window close still has peers left.
 */
export function shouldQuitOnLastChatClosed({
  platform,
  isQuittingForHandoff,
  remainingChatWindows,
  quitInProgress
}: {
  platform: NodeJS.Platform | string
  isQuittingForHandoff: boolean
  remainingChatWindows: number
  quitInProgress: boolean
}): boolean {
  return platform !== 'darwin' && !isQuittingForHandoff && !quitInProgress && remainingChatWindows === 0
}

/**
 * Explicit-activation restore for a user-invoked relaunch (#130810).
 *
 * Unlike the ambient `focusWindow` in main.ts (showInactive, never steal
 * foreground per #83998), a second-instance / dock-activate / tray-click is
 * the user asking for Hermes back, so the window must come back activatable:
 * restore a minimized window, show a tray-hidden one with `show()` (not
 * `showInactive()` — the latter leaves a Windows hidden-minimized window
 * painted-but-dead, #119252), then take keyboard focus when unfocused.
 * Showing emits `show`, which lets minimize-to-tray release the window from
 * its hidden set and clear `skipTaskbar` on Windows.
 */
export function activateWindow(window: ActivatableWindow | null | undefined): void {
  if (!window || window.isDestroyed()) {
    return
  }

  if (window.isMinimized()) {
    window.restore()
  }

  // Re-read visibility after the restore: a tray-hidden window reports
  // invisible until it is shown.
  if (!window.isVisible()) {
    window.show()
  }

  if (shouldFocusToTakeKeyboard(window)) {
    window.focus()
  }
}
