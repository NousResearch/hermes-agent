import { type ActiveWork, mergeActiveWork, quitPromptFor } from './quit-guard'

// A quit intercepted for active work remains held until the same prompt is
// answered. The confirmation latch permits exactly one re-entry into Electron's
// before-quit listeners, where the ordinary teardown can then proceed.
export function createDesktopHeldQuitRuntime(deps: {
  app: { quit: () => void }
  BrowserWindow: {
    getFocusedWindow: () => any
    getAllWindows: () => any[]
  }
  dialog: { showMessageBox: (parent: any, options: any) => Promise<{ response: number }> }
  activeWorkByWebContents: Map<number, ActiveWork>
  minimizeToTray: { status: () => { available: boolean }; restore: () => void }
  getIsQuittingForHandoff: () => boolean
  skipQuitConfirm: boolean
}) {
  const { app, BrowserWindow, dialog, activeWorkByWebContents, minimizeToTray, getIsQuittingForHandoff,
    skipQuitConfirm } = deps

  let quitPromptOpen = false
  let quitConfirmedWithActiveWork = false

  return function heldQuitForActiveWork(event: Electron.Event): boolean {
    const isQuittingForHandoff = getIsQuittingForHandoff()

    if (skipQuitConfirm || quitConfirmedWithActiveWork || isQuittingForHandoff) {
      return false
    }

    if (quitPromptOpen) {
      event.preventDefault()

      return true
    }

    const prompt = quitPromptFor(mergeActiveWork(activeWorkByWebContents.values()), isQuittingForHandoff)

    // A tray quit with live work still needs the ordinary visible confirmation.
    if (prompt && minimizeToTray.status().available) {
      minimizeToTray.restore()
    }

    // A hidden aux window must never parent the quit prompt: the dialog would
    // be invisible and the held quit unanswerable (#116376 §E).
    const parent = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows().find(window => window.isVisible())

    if (!prompt || !parent || parent.isDestroyed()) {
      return false
    }

    event.preventDefault()
    quitPromptOpen = true

    void dialog
      .showMessageBox(parent, {
        buttons: ['Keep Running', 'Quit Anyway'],
        cancelId: 0,
        defaultId: 0,
        detail: prompt.detail,
        message: prompt.message,
        type: 'question'
      })
      .then(({ response }) => {
        quitPromptOpen = false

        if (response === 1) {
          quitConfirmedWithActiveWork = true
          app.quit()
        }
      })
      .catch(() => {
        // A dialog we can't show must not become a quit we can't perform.
        quitPromptOpen = false
        quitConfirmedWithActiveWork = true
        app.quit()
      })

    return true
  }
}
