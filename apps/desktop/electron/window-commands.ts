/**
 * Window-scoped commands: ⌘W / ⌘R, the native back/forward gestures, the
 * Ctrl/Cmd zoom chords, and the application-menu items that mirror them
 * (File → Open Folder… / Close, View → Reload / Actual Size / Zoom In / Zoom Out).
 *
 * Each one acts on the window it came from: the window whose input hook caught
 * it, or the `browserWindow` Electron hands a menu click. Several full app
 * windows can be open at once (session, instance, HUD, popped-out Browser), and
 * sending these to the primary made ⌘R in a second window reload the first one
 * and ⌘W close a tab in a window the user wasn't looking at.
 *
 * The primary stays the fallback for a command with no live app window behind
 * it: a menu click with nothing focused, or from a window none of these hooks
 * were installed on (an OAuth sign-in page must never receive app IPC).
 */

import type { BaseWindow, BrowserWindow, WebContents } from 'electron'

import { commandFocusedGuest } from './preview-guest-offscreen'
import { windowAcceleratorAction } from './window-accelerator'
import { DEFAULT_ZOOM_LEVEL, ZOOM_STEP } from './zoom'

type PreviewNavCommand = 'back' | 'forward' | 'reload'
type ZoomCommand = 'zoom-in' | 'zoom-out' | 'zoom-reset'

/** Whatever a hook or a menu click names as the window a command came from. */
type CommandSource = BaseWindow | null | undefined

export interface WindowCommandsDeps {
  isMac: boolean
  /** Electron's focused webContents: a focused preview guest takes a nav command itself. */
  focusedWebContents: () => null | undefined | WebContents
  /** Where a command with no live source window goes. */
  primaryWindow: () => BrowserWindow | null | undefined
  /** True for a HUD window, including one still inside its close grace. */
  isHud: (window: BrowserWindow) => boolean
  /** Leave HUD mode from `window` (a HUD that is already closing ignores it). */
  leaveHud: (window: BrowserWindow) => void
  /** Apply and persist one window's zoom level. */
  setZoomLevel: (window: BrowserWindow, level: number) => void
}

const ZOOM_COMMANDS = new Set<string>(['zoom-in', 'zoom-out', 'zoom-reset'])

const isZoomCommand = (action: string): action is ZoomCommand => ZOOM_COMMANDS.has(action)

function live(window: BrowserWindow | null | undefined): BrowserWindow | null {
  return window && !window.isDestroyed() && !window.webContents.isDestroyed() ? window : null
}

function nextZoomLevel(window: BrowserWindow, command: ZoomCommand): number {
  if (command === 'zoom-reset') {
    return DEFAULT_ZOOM_LEVEL
  }

  return window.webContents.getZoomLevel() + (command === 'zoom-in' ? ZOOM_STEP : -ZOOM_STEP)
}

export function createWindowCommands(deps: WindowCommandsDeps) {
  // The app windows these hooks were installed on, and the subset that zooms
  // itself. Only those can be a command's own target.
  const commandWindows = new WeakSet<BaseWindow>()
  const zoomWindows = new WeakSet<BaseWindow>()

  const isWired = (source: CommandSource, wired: WeakSet<BaseWindow>): source is BrowserWindow =>
    Boolean(source) && wired.has(source)

  const target = (source: CommandSource, wired: WeakSet<BaseWindow>): BrowserWindow | null =>
    live(isWired(source, wired) ? source : null) ?? live(deps.primaryWindow())

  /**
   * A browser-navigation gesture: the focused page guest first, else the
   * source window's renderer, which routes it to its focused preview pane.
   * `reload` also has an app-level fallback there (reload that window); `back`
   * and `forward` mean nothing outside the browser, so the renderer ignores them.
   */
  function previewNav(source: CommandSource, command: PreviewNavCommand) {
    // The user is inside the page itself: main is the only party that can see
    // that, so act here and never round-trip.
    if (commandFocusedGuest(command, deps.focusedWebContents())) {
      return
    }

    target(source, commandWindows)?.webContents.send('hermes:preview-nav', command)
  }

  /** Ask a window's renderer to close its preview pane (no HUD meaning). */
  function closePreview(source: CommandSource) {
    target(source, commandWindows)?.webContents.send('hermes:close-preview-requested')
  }

  /** ⌘W and File → Close. */
  function closeTab(source: CommandSource) {
    const window = target(source, commandWindows)

    // ⌘W in the HUD is "leave HUD mode", not "close a tab in the app window".
    // Routing it to the main renderer closed the app's tab out from under the
    // user while the HUD stayed put. A closing (or superseded) HUD swallows it,
    // so it can't reach the hidden main window during its 1.5s close grace.
    if (window && deps.isHud(window)) {
      deps.leaveHud(window)

      return
    }

    window?.webContents.send('hermes:close-preview-requested')
  }

  /** File → Open Folder…: the same flow as the window's own ⌘O keybind. */
  function openFolder(source: CommandSource) {
    target(source, commandWindows)?.webContents.send('hermes:open-folder-requested')
  }

  /**
   * The window ⌘+ / ⌘- / ⌘0 would zoom. A window without zoom wiring (pet
   * overlay, quick entry, wake indicator) never claims those keys, so on macOS
   * the menu accelerator answers them, and that has always meant the primary.
   */
  function zoom(source: CommandSource, command: ZoomCommand) {
    const window = target(source, zoomWindows)

    if (window) {
      deps.setZoomLevel(window, nextZoomLevel(window, command))
    }
  }

  function installPreviewShortcut(window: BrowserWindow) {
    commandWindows.add(window)
    let focusedAt = 0

    window.on('focus', () => {
      focusedAt = Date.now()
    })

    window.webContents.on('before-input-event', (event, input) => {
      const action = windowAcceleratorAction(input, deps.isMac, Date.now() - focusedAt)

      // A ⌘W/⌘R that auto-repeats or lands right as focus arrives belongs to
      // the app that just lost focus (#105498). Claim it so the renderer's
      // keybind doesn't act on it either, and do nothing.
      if (action === 'swallow') {
        event.preventDefault()

        return
      }

      // Always claim ⌘W here (the File>Close item deliberately has no
      // accelerator, so nothing else does). The renderer decides tab-vs-window
      // — no `previewShortcutActive` gate, so it works for every closeable tab.
      // keyUp is not a claim: a chord that started in another app can deliver
      // its leftover keyup when this window inherits focus (#105498).
      if (action === 'close-tab') {
        event.preventDefault()
        closeTab(window)

        return
      }

      // ⌘R rides here rather than on the View menu item for the same reason:
      // the application menu only exists on macOS (it is set to null elsewhere,
      // see #77845), so a menu accelerator would leave Windows and Linux with no
      // way to reload a page at all. ⇧⌘R is left alone — that is `forceReload`,
      // the unconditional whole-window escape hatch.
      if (action === 'reload') {
        event.preventDefault()
        previewNav(window, 'reload')
      }
    })
  }

  /**
   * The native back/forward gestures, which never reach the renderer on their own.
   *
   * - macOS: a two/three-finger swipe. Chromium's own overscroll navigation is
   *   off in an Electron window, so the OS gesture surfaces as this event and
   *   nothing consumes it. Requires "Swipe between pages" in System Settings.
   * - Windows/Linux: the dedicated back/forward buttons on a mouse, delivered as
   *   `WM_APPCOMMAND`.
   */
  function installBrowserNavGestures(window: BrowserWindow) {
    commandWindows.add(window)

    window.on('swipe', (_event, direction) => {
      if (direction === 'left' || direction === 'right') {
        // Swipe LEFT moves the page left, revealing what's behind it — that's
        // back. Matches Safari, Chrome, and Finder.
        previewNav(window, direction === 'left' ? 'back' : 'forward')
      }
    })

    window.on('app-command', (event, command) => {
      if (command !== 'browser-backward' && command !== 'browser-forward') {
        return
      }

      // Claim it either way: unhandled, Chromium walks the HOST document's
      // history, which would navigate the app shell itself.
      event.preventDefault()
      previewNav(window, command === 'browser-backward' ? 'back' : 'forward')
    })
  }

  /**
   * Override Ctrl/Cmd + +/-/0 with half Chromium's default zoom step (ZOOM_STEP
   * is 0.1 vs Chromium's 0.2). The menu items handle this on macOS (where the
   * menu is always present), but on Linux/Windows the menu is null and
   * Chromium's default handler would use the full 0.2 step, so we intercept
   * here for consistency. Ctrl/Cmd+0 resets to DEFAULT_ZOOM_LEVEL, not Chromium 0.
   */
  function installZoomShortcuts(window: BrowserWindow) {
    zoomWindows.add(window)

    window.webContents.on('before-input-event', (event, input) => {
      const action = windowAcceleratorAction(input, deps.isMac)

      if (isZoomCommand(action)) {
        event.preventDefault()
        zoom(window, action)
      }
    })

    // Ctrl/Cmd + mouse wheel — the standard desktop/browser zoom gesture
    // (#40295). Chromium surfaces it as the main-process 'zoom-changed' event
    // (wheel events are DOM-side, so before-input-event never sees them).
    // Route through the same persist+notify funnel as the keyboard shortcuts
    // so wheel zoom survives restarts and the settings Scale control stays in
    // sync, and use the same half step for consistency.
    window.webContents.on('zoom-changed', (event, zoomDirection) => {
      event.preventDefault()
      zoom(window, zoomDirection === 'in' ? 'zoom-in' : 'zoom-out')
    })
  }

  return {
    closePreview,
    closeTab,
    installBrowserNavGestures,
    installPreviewShortcut,
    installZoomShortcuts,
    openFolder,
    previewNav,
    zoom
  }
}
