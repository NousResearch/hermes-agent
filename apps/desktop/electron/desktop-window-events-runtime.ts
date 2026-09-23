// Native window state and guest-page navigation stay coupled to the currently
// focused Electron window. Main injects the live primary window accessor.
export function createDesktopWindowEventsRuntime(deps: {
  DARWIN_MAJOR: number
  IS_MAC: boolean
  IS_WINDOWS: boolean
  IS_WSL: boolean
  WINDOW_BUTTON_POSITION: Electron.Point
  computeNativeOverlayWidth: (...args: any[]) => any
  electronWebContents: any
  getMainWindow: () => Electron.BrowserWindow | null
  windowControlState: (...args: any[]) => any
}) {
  const {
    DARWIN_MAJOR,
    IS_MAC,
    IS_WINDOWS,
    IS_WSL,
    WINDOW_BUTTON_POSITION,
    computeNativeOverlayWidth,
    electronWebContents,
    getMainWindow,
    windowControlState
  } = deps

  function getWindowButtonPosition(win = getMainWindow()) {
    if (!IS_MAC) {
      return null
    }

    // Fullscreen hides the traffic lights — treat as no left-side controls so the
    // renderer drops the traffic-light dodge inset and Y nudge.
    if (win?.isFullScreen?.()) {
      return null
    }

    return win?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION
  }

  function getNativeOverlayWidth() {
    return computeNativeOverlayWidth({ isWindows: IS_WINDOWS, isWsl: IS_WSL, isMac: IS_MAC })
  }

  function getWindowState(win = getMainWindow()) {
    return {
      isFullscreen: Boolean(win?.isFullScreen?.()),
      isMinimized: Boolean(win?.isMinimized?.()),
      isVisible: Boolean(win?.isVisible?.()),
      nativeOverlayWidth: getNativeOverlayWidth(),
      windowButtonPosition: getWindowButtonPosition(win),
      darwinMajor: IS_MAC ? DARWIN_MAJOR : 0,
      ...windowControlState(win, !IS_WINDOWS && IS_WSL)
    }
  }

  function sendClosePreviewRequested() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:close-preview-requested')
  }

  /**
   * Run a browser gesture on the guest page the user is actually in, if any.
   *
   * A `<webview>` guest is its own out-of-process webContents: pointer and focus
   * events inside the page never reach the host document, so NOTHING in the
   * renderer — not `document.activeElement`, not the layout tree's hover/focus
   * ladder — can see that the user is in there. Main can: Electron tracks the
   * focused webContents across processes, which is the definition of a runtime
   * fact it owns.
   *
   * Returns false when focus is in the app's own chrome, where the renderer is
   * the one that knows which pane is active.
   */
  function commandFocusedGuest(command: 'back' | 'forward' | 'reload'): boolean {
    const focused = electronWebContents.getFocusedWebContents()

    if (!focused || focused.isDestroyed() || focused.getType() !== 'webview') {
      return false
    }

    const history = focused.navigationHistory

    if (command === 'reload') {
      focused.reload()
    } else if (command === 'back') {
      if (!history.canGoBack()) {
        return true
      }

      history.goBack()
    } else {
      if (!history.canGoForward()) {
        return true
      }

      history.goForward()
    }

    return true
  }

  /**
   * Ask the renderer to run a browser-navigation gesture on its focused preview
   * pane. `reload` also has an app-level fallback (reload the window); `back` and
   * `forward` mean nothing outside the browser, so the renderer just ignores them.
   */
  function sendPreviewNavCommand(command: 'back' | 'forward' | 'reload') {
    const mainWindow = getMainWindow()

    // The user is inside the page itself — main is the only party that can see
    // that, so act here and never round-trip.
    if (commandFocusedGuest(command)) {
      return
    }

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:preview-nav', command)
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
  function installBrowserNavGestures(window) {
    window.on('swipe', (_event, direction) => {
      if (direction === 'left' || direction === 'right') {
        // Swipe LEFT moves the page left, revealing what's behind it — that's
        // back. Matches Safari, Chrome, and Finder.
        sendPreviewNavCommand(direction === 'left' ? 'back' : 'forward')
      }
    })

    window.on('app-command', (event, command) => {
      if (command !== 'browser-backward' && command !== 'browser-forward') {
        return
      }

      // Claim it either way: unhandled, Chromium walks the HOST document's
      // history, which would navigate the app shell itself.
      event.preventDefault()
      sendPreviewNavCommand(command === 'browser-backward' ? 'back' : 'forward')
    })
  }

  function sendOpenFolderRequested() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const webContents = mainWindow.webContents

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:open-folder-requested')
  }

  // Push titlebar/fullscreen chrome state to a window's renderer. Defaults to the
  // primary, but any full chat window (primary or a secondary "instance" peer)
  // passes itself so its own fullscreen toggle drives its own traffic-light inset.
  function sendWindowStateChanged(nextIsFullscreen?: boolean, target = getMainWindow()) {
    if (!target || target.isDestroyed()) {
      return
    }

    const { webContents } = target

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    const state = getWindowState(target)

    if (typeof nextIsFullscreen === 'boolean') {
      state.isFullscreen = nextIsFullscreen
    }

    webContents.send('hermes:window-state-changed', state)
  }

  return {
    getWindowState,
    sendClosePreviewRequested,
    sendPreviewNavCommand,
    installBrowserNavGestures,
    sendOpenFolderRequested,
    sendWindowStateChanged
  }
}
