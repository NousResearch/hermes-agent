// The renderer mounts its open-updates listener in the same effect pass that
// signals deep-link readiness. Preserve a pending request until it can hear it.
export function createDesktopUpdateWindowRequest(deps: { getMainWindow: () => any }) {
  let rendererReadyForDeepLink = false
  let pendingOpenUpdates = false

  function sendOpenUpdatesRequested() {
    const mainWindow = deps.getMainWindow()

    if (!rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
      pendingOpenUpdates = true

      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:open-updates')

    if (!mainWindow.isVisible()) {
      mainWindow.show()
    }

    mainWindow.focus()
  }

  return {
    sendOpenUpdatesRequested,
    getPendingOpenUpdates: () => pendingOpenUpdates,
    setPendingOpenUpdates: (pending: boolean) => { pendingOpenUpdates = pending },
    getRendererReadyForDeepLink: () => rendererReadyForDeepLink,
    setRendererReadyForDeepLink: (ready: boolean) => { rendererReadyForDeepLink = ready }
  }
}
