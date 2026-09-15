import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, dialog, ipcMain } from 'electron'

import { type ActiveWork, isQuitConfirmationMode, type QuitConfirmationMode, quitPromptFor } from './quit-guard'

interface QuitConfirmationOptions {
  getWork: () => ActiveWork
  isQuittingForHandoff: () => boolean
  skipConfirmation: boolean
  writeFile: (target: string, data: string, encoding: BufferEncoding) => void
}

export function createQuitConfirmation(options: QuitConfirmationOptions) {
  const configPath = path.join(app.getPath('userData'), 'quit-confirmation.json')
  const windows = new Set<BrowserWindow>()
  const closingWindows = new Set<BrowserWindow>()
  let mode: QuitConfirmationMode = 'while-working'
  let quitPromptOpen = false
  let quitConfirmed = false

  try {
    const saved = JSON.parse(fs.readFileSync(configPath, 'utf8'))?.mode

    if (isQuitConfirmationMode(saved)) {
      mode = saved
    }
  } catch {
    // Missing or malformed preferences keep the existing active-work guard.
  }

  ipcMain.handle('hermes:setting:quitConfirmation:get', () => mode)
  ipcMain.handle('hermes:setting:quitConfirmation:set', (_event, value: unknown) => {
    if (!isQuitConfirmationMode(value)) {
      throw new Error('Invalid quit confirmation preference')
    }

    fs.mkdirSync(path.dirname(configPath), { recursive: true })
    options.writeFile(configPath, JSON.stringify({ mode: value }, null, 2), 'utf8')
    mode = value

    return mode
  })

  function holdQuit(event: Electron.Event): boolean {
    if (options.skipConfirmation || quitConfirmed || options.isQuittingForHandoff()) {
      return false
    }

    // Repeated quit/close gestures must not reach teardown while unanswered.
    if (quitPromptOpen) {
      event.preventDefault()

      return true
    }

    const work = options.getWork()
    const prompt = quitPromptFor(work, false, mode)

    if (!prompt) {
      return false
    }

    event.preventDefault()
    quitPromptOpen = true

    const parent = [BrowserWindow.getFocusedWindow(), ...BrowserWindow.getAllWindows()].find(
      win => win && !win.isDestroyed() && !closingWindows.has(win)
    )

    const messageBoxOptions: Electron.MessageBoxOptions = {
      buttons: ['Keep Running', work.count > 0 ? 'Quit Anyway' : 'Quit'],
      cancelId: 0,
      defaultId: 0,
      detail: prompt.detail,
      message: prompt.message,
      type: 'question'
    }

    // Explicit macOS Dock quits can arrive after every window has closed.
    const response = parent
      ? dialog.showMessageBox(parent, messageBoxOptions)
      : dialog.showMessageBox(messageBoxOptions)

    void response
      .then(({ response }) => {
        quitPromptOpen = false

        if (response === 1) {
          quitConfirmed = true
          app.quit()
        }
      })
      .catch(() => {
        // A dialog we can't show must not become a quit we can't perform.
        quitPromptOpen = false
        quitConfirmed = true
        app.quit()
      })

    return true
  }

  function trackWindow(win: BrowserWindow) {
    windows.add(win)
    win.on('closed', () => {
      windows.delete(win)
      closingWindows.delete(win)
    })
    win.webContents.on('will-prevent-unload', event => {
      queueMicrotask(() => {
        // Here preventDefault overrides the renderer's veto and allows closing.
        if (!event.defaultPrevented) {
          closingWindows.delete(win)
        }
      })
    })
    win.on('close', event => {
      // Only real app windows are tracked: dismissing a pet, HUD, or Quick
      // Entry must never quit the app. macOS window-close keeps the Dock app.
      const lastWindow = ![...windows].some(other => other !== win && !closingWindows.has(other))

      if ((quitPromptOpen || (process.platform !== 'darwin' && lastWindow)) && holdQuit(event)) {
        return
      }

      if (event.defaultPrevented) {
        return
      }

      // Closing waits for the renderer. Two close requests can arrive before
      // either window emits 'closed'; the second must still protect a window.
      closingWindows.add(win)
      queueMicrotask(() => {
        if (event.defaultPrevented) {
          closingWindows.delete(win)
        }
      })
    })
  }

  return { holdQuit, trackWindow }
}
