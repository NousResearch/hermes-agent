// External URLs, find-in-page, and preview reachability are main-process
// navigation channels. Keep sender-scoped find listeners and their cleanup here.
export function registerDesktopPageInteractionIpc(deps: {
  BrowserWindow: any
  installFoundInPageForwarder: (...args: any[]) => () => void
  ipcMain: any
  openExternalUrl: (url: string) => boolean
  openPreviewInBrowser: (url: string) => Promise<boolean>
  performFindAfterIndexingStarted: (...args: any[]) => Promise<any>
  reachablePreviewUrl: (...args: any[]) => Promise<any>
  stopFind: (...args: any[]) => any
}) {
  const {
    BrowserWindow,
    installFoundInPageForwarder,
    ipcMain,
    openExternalUrl,
    openPreviewInBrowser,
    performFindAfterIndexingStarted,
    reachablePreviewUrl,
    stopFind
  } = deps

  ipcMain.handle('hermes:openExternal', (_event, url) => {
    if (!openExternalUrl(url)) {
      throw new Error('Invalid external URL')
    }
  })

  // ── Find-in-page (Ctrl/Cmd+F) ─────────────────────────────────────────────
  // The desktop supports multiple BrowserWindows (one primary plus any
  // per-session secondary windows spawned via `hermes:window:openSession`).
  // Find must run against the requesting window, not a global — otherwise
  // Cmd+F pressed in a secondary session window would search the primary
  // and the match counter would report matches the user can't see. Resolve
  // the sender through `BrowserWindow.fromWebContents(event.sender)` and
  // forward `found-in-page` results back to that same sender.

  // Lazily-installed forwarder per sender webContents. We track one
  // uninstall fn per webContents id and prune entries when the sender goes
  // away — Electron does not auto-detach webContents listeners on close,
  // so the map is the cleanup path.
  const foundInPageForwarders = new Map<number, () => void>()

  function ensureFoundInPageForwarder(sender: Electron.WebContents): void {
    if (foundInPageForwarders.has(sender.id)) {
      return
    }

    const uninstall = installFoundInPageForwarder(sender)
    foundInPageForwarders.set(sender.id, uninstall)

    sender.once('destroyed', () => {
      foundInPageForwarders.get(sender.id)?.()
      foundInPageForwarders.delete(sender.id)
    })
  }

  ipcMain.handle('hermes:find-in-page', async (event, query, options) => {
    const win = BrowserWindow.fromWebContents(event.sender)

    if (!win || win.isDestroyed()) {
      return { count: 0 }
    }

    ensureFoundInPageForwarder(event.sender)
    await performFindAfterIndexingStarted(win.webContents, query, options)

    // The match count still arrives asynchronously via `found-in-page`; this
    // reply only acknowledges that Chromium has begun returning this request.
    return { count: 0 }
  })

  ipcMain.handle('hermes:stop-find-in-page', event => {
    const win = BrowserWindow.fromWebContents(event.sender)

    if (!win || win.isDestroyed()) {
      return
    }

    stopFind(win.webContents)
  })

  // The renderer can't know whether a loopback URL is reachable — only main
  // knows which transport backs this gateway. Ask before loading one.
  ipcMain.handle('hermes:preview:reach', async (event, url) => reachablePreviewUrl(event.sender.id, String(url || '')))

  ipcMain.handle('hermes:openPreviewInBrowser', async (_event, url) => {
    if (!(await openPreviewInBrowser(url))) {
      throw new Error('Invalid preview URL')
    }
  })

}
