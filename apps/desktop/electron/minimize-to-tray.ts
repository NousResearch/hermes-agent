import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, ipcMain, Menu, nativeImage, Tray } from 'electron'

export interface MinimizeToTrayStatus {
  enabled: boolean
  available: boolean
}

interface Options {
  preferencesPath: string
  getIconPath: () => string | undefined
  restoreMainWindow: () => void
  isQuittingForHandoff: () => boolean
  log: (message: string) => void
}

/** Device-local native preference; renderer windows only cache its status. */
export function createMinimizeToTray(options: Options) {
  let enabled = false
  let quitting = false
  let tray: Tray | null = null
  let stopWatchingHost: (() => void) | undefined
  let dockHidden = false
  let hostGeneration = 0
  let pending = Promise.resolve()
  const windows = new Set<BrowserWindow>()
  const hidden = new Set<BrowserWindow>()

  const status = (): MinimizeToTrayStatus => ({ enabled, available: !!tray && !tray.isDestroyed() })

  const broadcast = () => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:minimize-to-tray:changed', status())
      }
    }
  }

  const showDock = () => {
    if (dockHidden) {
      dockHidden = false
      void app.dock?.show()
    }
  }

  const syncDock = () => {
    if (process.platform !== 'darwin') {
      return
    }

    // A hidden primary must not remove a visible peer from Cmd-Tab/the Dock.
    const foreground = [...windows].some(win => !win.isDestroyed() && win.isVisible() && !win.isMinimized())

    if (status().available && hidden.size > 0 && !foreground && !quitting) {
      dockHidden = true
      app.dock?.hide()
    } else {
      showDock()
    }
  }

  const released = (win: BrowserWindow) => {
    if (!hidden.delete(win)) {
      return
    }

    if (process.platform === 'win32') {
      win.setSkipTaskbar(false)
    }

    showDock()
  }

  const restoreHidden = () => {
    showDock()

    for (const win of [...hidden]) {
      if (win.isDestroyed()) {
        hidden.delete(win)

        continue
      }

      released(win)

      if (win.isMinimized()) {
        win.restore()
      }

      win.showInactive()
    }
  }

  const restore = () => {
    restoreHidden()
    options.restoreMainWindow()
  }

  const destroyTray = () => {
    stopWatchingHost?.()
    stopWatchingHost = undefined
    tray?.destroy()
    tray = null
  }

  const hostLost = () => {
    // Losing the shell/tray must never strand an invisible app.
    hostGeneration += 1
    restoreHidden()
    destroyTray()
    broadcast()
  }

  const apply = async (on: boolean) => {
    enabled = on

    if (!on) {
      restoreHidden()
      destroyTray()
    } else if (!status().available && !quitting) {
      try {
        if (process.platform === 'linux') {
          const { watchLinuxTrayHost } = await import('./tray-host')
          const generation = hostGeneration
          stopWatchingHost = await watchLinuxTrayHost(hostLost)

          if (generation !== hostGeneration) {
            throw new Error('System tray host disappeared')
          }
        }

        if (quitting) {
          destroyTray()

          return status()
        }

        const iconPath = options.getIconPath()
        const icon = iconPath ? nativeImage.createFromPath(iconPath) : nativeImage.createEmpty()

        if (icon.isEmpty()) {
          throw new Error('No usable tray icon')
        }

        tray = new Tray(
          icon.resize({
            width: process.platform === 'darwin' ? 18 : 24,
            height: process.platform === 'darwin' ? 18 : 24
          })
        )
        tray.setToolTip('Hermes')
        tray.setContextMenu(
          Menu.buildFromTemplate([
            { label: 'Show Hermes', click: restore },
            { type: 'separator' },
            // Do not bypass the ordinary active-work confirmation or teardown.
            { label: 'Quit Hermes', click: () => app.quit() }
          ])
        )

        // macOS single-click opens the native menu, not the window behind it.
        if (process.platform !== 'darwin') {
          tray.on('click', restore)
        }

        tray.on('double-click', restore)
      } catch (error) {
        restoreHidden()
        destroyTray()
        options.log(`[tray] unavailable; ordinary window behavior retained: ${error}`)
      }
    }

    broadcast()

    return status()
  }

  function registerWindow(win: BrowserWindow, { closeToTray = false } = {}) {
    windows.add(win)

    const hide = () => {
      if (!enabled || !status().available || quitting || options.isQuittingForHandoff()) {
        return false
      }

      hidden.add(win)

      if (process.platform === 'win32') {
        win.setSkipTaskbar(true)
      }

      win.hide()
      syncDock()
      return true
    }

    win.on('minimize', hide)
    if (closeToTray) {
      win.on('close', event => {
        if (hide()) {
          event.preventDefault()
        }
      })
    }

    // Windows session ending need not emit app.before-quit. Never hold it open.
    win.on('query-session-end', () => {
      quitting = true
    })
    win.on('show', () => {
      released(win)
      syncDock()
    })
    win.on('restore', () => {
      released(win)
      syncDock()
    })
    win.on('closed', () => {
      windows.delete(win)
      hidden.delete(win)
      syncDock()
    })
  }

  async function start() {
    let on = false

    try {
      on = JSON.parse(fs.readFileSync(options.preferencesPath, 'utf8')).enabled === true
    } catch {
      // Missing or malformed preference preserves ordinary minimize/close.
    }

    const operation = apply(on)
    pending = operation.then(
      () => undefined,
      () => undefined
    )

    return operation
  }

  function setEnabled(on: boolean): Promise<MinimizeToTrayStatus> {
    // Serialize writes from peer windows so an older native apply cannot win.
    const operation = pending.then(async () => {
      fs.mkdirSync(path.dirname(options.preferencesPath), { recursive: true })
      fs.writeFileSync(`${options.preferencesPath}.tmp`, JSON.stringify({ enabled: on === true }), 'utf8')
      fs.renameSync(`${options.preferencesPath}.tmp`, options.preferencesPath)

      return apply(on === true)
    })

    pending = operation.then(
      () => undefined,
      () => undefined
    )

    return operation
  }

  ipcMain.handle('hermes:minimize-to-tray:get', status)
  ipcMain.handle('hermes:minimize-to-tray:set', (_event, on) => setEnabled(on === true))
  app.on('will-quit', destroyTray)

  return {
    start,
    status,
    setEnabled,
    registerWindow,
    restore,
    // Call only AFTER the active-work guard accepts the quit. Cancelling the
    // prompt must leave hiding and its recovery affordance intact.
    beginQuit: () => {
      quitting = true
    }
  }
}
