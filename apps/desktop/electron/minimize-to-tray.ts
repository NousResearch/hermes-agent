import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, ipcMain, Menu, type MenuItemConstructorOptions, nativeImage, Tray } from 'electron'

export interface MinimizeToTrayStatus {
  enabled: boolean
  available: boolean
}

interface Options {
  preferencesPath: string
  getIconPath: () => string | undefined
  restoreMainWindow: () => void
  /** Left-click / double-click-adjacent action. Defaults to restoreMainWindow. */
  onClick?: () => void
  /** The tray's context menu as an Electron template. Defaults to the
   *  Show/Quit pair — a caller that only wants hide-on-close need not know
   *  anything about menu items. */
  buildMenu?: () => MenuItemConstructorOptions[]
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

  const released = (win: BrowserWindow): boolean => {
    if (!hidden.delete(win)) {
      return false
    }

    if (process.platform === 'win32') {
      win.setSkipTaskbar(false)
    }

    showDock()

    return true
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

      if (process.platform === 'win32') {
        // showInactive() never activates the window. A restored-but-inactive
        // window can come back painted yet dead to input (AppHangB1, #119252),
        // so genuinely activate it like focusWindow in main.ts does.
        win.show()
        win.focus()
      } else {
        win.showInactive()
      }
    }
  }

  const restore = () => {
    restoreHidden()
    options.restoreMainWindow()
  }

  const destroyTray = (force = false) => {
    stopWatchingHost?.()
    stopWatchingHost = undefined

    // Linux StatusNotifierItem stays exported after Tray.destroy(), so a later
    // `new Tray()` in this process cannot re-export and the panel icon dies
    // (#126353). Park the instance until the app actually quits or the host
    // disappears.
    if (process.platform === 'linux' && !quitting && !force) {
      return
    }

    tray?.destroy()
    tray = null
  }

  const hostLost = () => {
    // Losing the shell/tray must never strand an invisible app.
    hostGeneration += 1
    restoreHidden()
    destroyTray(true)
    broadcast()
  }

  const apply = async (on: boolean) => {
    enabled = on

    if (!on) {
      restoreHidden()
      destroyTray()
    } else if (!quitting && status().available && process.platform === 'linux' && !stopWatchingHost) {
      try {
        const { watchLinuxTrayHost } = await import('./tray-host')
        const generation = hostGeneration
        stopWatchingHost = await watchLinuxTrayHost(hostLost)

        if (generation !== hostGeneration) {
          throw new Error('System tray host disappeared')
        }
      } catch (error) {
        restoreHidden()
        destroyTray(true)
        options.log(`[tray] unavailable; ordinary window behavior retained: ${error}`)
      }
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
        // A caller-supplied menu is rebuilt from scratch on every (re)create,
        // so its labels can never go stale, and each click re-enters the
        // caller's action router rather than closing over this module's view
        // of the app.
        tray.setContextMenu(
          Menu.buildFromTemplate(
            options.buildMenu?.() ?? [
              { label: 'Show Hermes', click: restore },
              { type: 'separator' },
              // Do not bypass the ordinary active-work confirmation or teardown.
              { label: 'Quit Hermes', click: () => app.quit() }
            ]
          )
        )

        // macOS single-click opens the native menu, not the window behind it.
        if (process.platform !== 'darwin') {
          tray.on('click', () => (options.onClick ?? restore)())
        }

        // A double click is always "show me the app": whatever the single
        // click is configured to do, two clicks in a row are the user asking
        // for the window, not for a second fresh conversation.
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
      if (win.isDestroyed()) {
        return false
      }

      if (!enabled || !status().available || quitting || options.isQuittingForHandoff()) {
        return false
      }

      hidden.add(win)

      if (process.platform === 'win32') {
        win.setSkipTaskbar(true)
        // A hidden Chromium window on Windows neither emits blur nor releases
        // the UI thread's keyboard focus: keys keep going to the invisible
        // page, trapping keyboard navigation and screen readers in it. Release
        // focus before hiding -- once hidden it no longer takes (#126570).
        win.blur()
      }

      win.hide()
      syncDock()

      return true
    }

    // Hide past the native minimize dispatch, not inside it: hiding
    // synchronously here re-enters window-state changes mid-flight and on
    // Windows wedges isMinimized(), so the later restore takes the
    // restore-on-hidden path back to a painted-but-dead window (#119252).
    // Guards are re-evaluated at fire time inside hide(); the close handler
    // below keeps its synchronous hide so preventDefault still works.
    win.on('minimize', () => {
      setImmediate(() => {
        // The user may have restored the window in the meantime (taskbar or
        // shortcut); a stale hide must not snatch it back.
        if (!win.isDestroyed() && !win.isMinimized() && win.isVisible()) {
          return
        }

        hide()
      })
    })

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

    // A relaunch, a deep link or a notification click can restore a tray-hidden
    // window without the tray. restore() alone paints the window on Windows but
    // does not make it the foreground window, so it drops all input (#127349).
    // Finish the activation like restoreHidden() does. Defer it, because the
    // native restore must finish first (same reason as the deferred hide).
    const release = () => {
      if (released(win) && process.platform === 'win32') {
        setImmediate(() => {
          if (!win.isDestroyed() && !win.isMinimized() && win.isVisible()) {
            win.show()
            win.focus()
          }
        })
      }

      syncDock()
    }

    win.on('show', release)
    win.on('restore', release)
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
  app.on('will-quit', () => {
    quitting = true
    destroyTray()
  })

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
