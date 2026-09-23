// Window security, native shortcuts, and reveal share one registration point.
// Keep callback ownership in main so every late-created window uses its live
// Electron and external-open policy functions.
export function createDesktopWindowWiringRuntime(deps: {
  DEV_SERVER: string | undefined
  PREVIEW_GUEST_PRELOAD_PATH: string
  app: any
  createWindowOpenHandler: (...args: any[]) => any
  createWindowRevealController: (...args: any[]) => any
  installBrowserNavGestures: (...args: any[]) => any
  installContextMenuBridge: (...args: any[]) => any
  installDevToolsShortcut: (...args: any[]) => any
  installFindShortcut: (...args: any[]) => any
  installPreviewShortcut: (...args: any[]) => any
  installZoomReassertOnNavigation: (...args: any[]) => any
  installZoomReassertOnWindowEvents: (...args: any[]) => any
  installZoomShortcuts: (...args: any[]) => any
  openExternalUrl: (...args: any[]) => any
  rememberLog: (message: string) => void
  restorePersistedZoomLevel: (...args: any[]) => any
}) {
  const {
    DEV_SERVER,
    PREVIEW_GUEST_PRELOAD_PATH,
    app,
    createWindowOpenHandler,
    createWindowRevealController,
    installBrowserNavGestures,
    installContextMenuBridge,
    installDevToolsShortcut,
    installFindShortcut,
    installPreviewShortcut,
    installZoomReassertOnNavigation,
    installZoomReassertOnWindowEvents,
    installZoomShortcuts,
    openExternalUrl,
    rememberLog,
    restorePersistedZoomLevel
  } = deps

  // Shared navigation guards + window chrome wiring applied to every window
  // (the primary plus any secondary session windows). Factored out of
  // createWindow() so secondary windows can't drift from the main window's
  // security posture: external links open in the OS browser, in-app navigation
  // stays confined to the dev server / packaged file URL, and the preview /
  // devtools / zoom / context-menu affordances behave identically everywhere.
  //
  // `zoom` is opt-out for the pet overlay: it sizes its own OS window to fit the
  // sprite in unzoomed CSS px (overlayWindowSize -> setBounds) and has its own
  // Alt+wheel scale, so inheriting the global UI zoom would render the mascot
  // larger than its window and crop it. Chat windows keep zoom on.
  function wireCommonWindowHandlers(win, { zoom = true }: { zoom?: boolean } = {}) {
    installPreviewShortcut(win)
    installDevToolsShortcut(win)
    installBrowserNavGestures(win)

    // Claim Ctrl/Cmd+F in the main process — on Pop!_OS / GNOME-based Linux
    // distros the Ctrl+F keydown does not reach the renderer's `view.findInPage`
    // binding (#81727). Routing it through `before-input-event` forwards the
    // intent at the earliest observable point. macOS / Windows keep the
    // renderer's own rebindable keybind, so the hook is Linux-only: installing
    // it elsewhere would make Ctrl/Cmd+F un-rebindable and double-open.
    if (process.platform === 'linux') {
      installFindShortcut(win)
    }

    if (zoom) {
      installZoomShortcuts(win)
      // Re-apply persisted zoom on show/restore/resize/cross-display move
      // (Chromium can drop webContents zoom after these window transitions), on
      // EVERY full load — not once, since crash recovery reloads and would
      // outlive a spent `once` listener (#46429) — and after in-page navigation,
      // where Chromium applies the target hash route's own per-URL zoom record
      // (see installZoomReassertOnNavigation; #48658, #38854, #79863).
      const reassertZoom = () => restorePersistedZoomLevel(win)

      installZoomReassertOnWindowEvents(win, reassertZoom)
      installZoomReassertOnNavigation(win.webContents, reassertZoom)
    }

    installContextMenuBridge(win)
    // Always deny, never open as a side effect: GHSA-9f4c-93c8-jc8g. Trusted
    // links arrive via `hermes:openExternal`, not here. See window-open-policy.ts.
    win.webContents.setWindowOpenHandler(
      createWindowOpenHandler(origin => rememberLog(`[window-open] denied: ${origin}`))
    )
    win.webContents.on('will-navigate', (event, url) => {
      if ((DEV_SERVER && url.startsWith(DEV_SERVER)) || (!DEV_SERVER && url.startsWith('file:'))) {
        return
      }

      event.preventDefault()
      openExternalUrl(url)
    })
  }

  /**
   * Give the preview pane's `<webview>` guests a preload — and ONLY those
   * guests. The pane's webview is the one `webview` tag in the app and it
   * always carries the `persist:hermes-preview` partition, so the partition is
   * the ownership key: any future webview that does not opt into that partition
   * inherits nothing from this mechanism.
   *
   * The preload (preview-guest-preload-entry.ts) never opens anything itself.
   * It forwards a clicked `_blank` anchor to the host renderer via
   * `sendToHost`, and the pane admits the scheme and routes the URL through the
   * audited `hermes:openExternal` channel. Popup requests themselves stay
   * denied-by-omission: the webview has no `allowpopups`, and the
   * `setWindowOpenHandler` contract (GHSA-9f4c-93c8-jc8g) stays side-effect
   * free.
   */
  function installPreviewGuestPreload() {
    app.on('web-contents-created', (_event, contents) => {
      if (contents.getType() !== 'window') {
        return
      }

      contents.on('will-attach-webview', (_attachEvent, webPreferences, params) => {
        if (params.partition !== 'persist:hermes-preview') {
          return
        }

        webPreferences.preload = PREVIEW_GUEST_PRELOAD_PATH
      })
    })
  }

  // Every window we open starts with `show: false` so the renderer's first themed
  // paint lands before it appears, and `ready-to-show` is what reveals it.
  // Electron 40 can drop that event entirely (electron/electron#51972) on
  // Linux/Wayland, remote displays and VMs, leaving the window hidden forever even
  // though the renderer finished loading. Keep the themed path as the preferred
  // reveal, then fall back a few seconds after the renderer loads. `show` and
  // `onRevealed` carry the caller's reveal action and post-visible work; whichever
  // path wins runs them exactly once.
  function wireWindowReveal(win, { show, onRevealed }: { show?: () => void; onRevealed?: () => void } = {}) {
    const controller = createWindowRevealController(
      {
        isDestroyed: () => win.isDestroyed(),
        isVisible: () => win.isVisible(),
        show: show ?? (() => win.show())
      },
      { onRevealed }
    )

    win.once('ready-to-show', controller.reveal)
    win.webContents.once('did-finish-load', controller.scheduleFallback)
    win.on('closed', controller.dispose)

    return controller
  }

  return { wireCommonWindowHandlers, installPreviewGuestPreload, wireWindowReveal }
}
