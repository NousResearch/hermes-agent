import { extractDeepLink } from './deep-link-protocol'
import { type MacApplicationsGuardResult, macGuardAllowsWindows } from './mac-applications-guard'
import { hasClosePreviewFlag } from './preview-guest-escape'
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

interface SecondInstanceDeps<T extends MainWindowLike> {
  mainWindow: T | null | undefined
  /** The Applications move prompt's answer; null while it still waits. */
  launchGuard: MacApplicationsGuardResult | null
  deepLinkSchemes: readonly string[]
  handleDeepLink: (url: string) => void
  createWindow: () => unknown
  focusWindow: (window: T) => unknown
  log: (message: string) => void
}

/** The primary instance's `second-instance` handler: another launch's argv arrives here. */
export function handleSecondInstance<T extends MainWindowLike>(
  argv: readonly string[],
  deps: SecondInstanceDeps<T>
): void {
  // --close-preview: the same escape hatch as hermes://close-preview, for
  // environments where spawning a URL is harder than a flag (kiosk launchers,
  // SSH-started sessions). Checked before deep links so a carried `hermes://`
  // URL still routes normally when no flag is present.
  if (hasClosePreviewFlag(argv)) {
    deps.handleDeepLink('hermes://close-preview')
  }

  const url = extractDeepLink(argv, deps.deepLinkSchemes)

  if (url) {
    deps.handleDeepLink(url)
  }

  // #130810: a second Start-menu / shortcut / Hermes.exe launch must never
  // silently exit. Log it so a future silent exit stays diagnosable, then
  // restore a live primary with activation (restore + show + focus), or
  // re-create it when it was destroyed.
  deps.log(`[second-instance] relaunch (deepLink=${url ? 'yes' : 'no'})`)
  ensureMainWindow(deps.mainWindow, {
    // app.isReady() is already true while the move prompt waits; the first window waits for its answer.
    isReady: macGuardAllowsWindows(deps.launchGuard),
    createWindow: deps.createWindow,
    focusWindow: deps.focusWindow,
    // deep-link delivery focuses a live window after its renderer is ready.
    focusExisting: !url
  })
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
