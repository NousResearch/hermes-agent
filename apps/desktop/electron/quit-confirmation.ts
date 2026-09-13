import { app, BrowserWindow, dialog } from 'electron'

import { type ActiveWork, quitPromptFor } from './quit-guard'

interface QuitConfirmationOptions {
  getWork: () => ActiveWork
  isQuittingForHandoff: () => boolean
  skipConfirmation: boolean
}

export function createQuitConfirmation(options: QuitConfirmationOptions) {
  let quitPromptOpen = false
  let quitConfirmedWithActiveWork = false

  function holdQuit(event: Electron.Event): boolean {
    if (options.skipConfirmation || quitConfirmedWithActiveWork || quitPromptOpen) {
      return false
    }

    const prompt = quitPromptFor(options.getWork(), options.isQuittingForHandoff())
    const parent = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0]

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

  return { holdQuit }
}
