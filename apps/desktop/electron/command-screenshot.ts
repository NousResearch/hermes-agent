import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, desktopCapturer, ipcMain, shell, systemPreferences } from 'electron'
import type { IpcMainEvent, IpcMainInvokeEvent } from 'electron'

import { createScreenshotCapture } from './command-screenshot-capture'
import { CommandScreenshotMonitor } from './command-screenshot-monitor'
import type { ScreenshotStatus } from './command-screenshot-types'

/** Device-local, opt-in native gesture. No renderer can request an arbitrary screenshot. */
export function installCommandScreenshot({ rendererUrl }: { rendererUrl: string }): () => void {
  if (process.platform !== 'darwin') {
    return () => undefined
  }

  const configPath = path.join(app.getPath('userData'), 'screenshot.json')
  const expectedUrl = new URL(rendererUrl)
  const monitor = new CommandScreenshotMonitor({ appPath: app.getAppPath() })
  const hasScreenPermission = () => systemPreferences.getMediaAccessStatus('screen') === 'granted'
  const capture = createScreenshotCapture({ hasScreenPermission, getSources: options => desktopCapturer.getSources(options) })
  const recipients = new Map<number, () => void>()
  let lastRecipient: BrowserWindow | null = null
  let enabled = false
  let monitorState: ScreenshotStatus['state'] = 'disabled'
  let disposed = false
  let generation = 0

  try {
    enabled = JSON.parse(readFileSync(configPath, 'utf8')).enabled === true
  } catch {
    // Missing or malformed device preference must never opt the user in.
  }

  const status = (): ScreenshotStatus => ({
    enabled,
    state: !enabled ? 'disabled' : monitorState === 'ready' && !hasScreenPermission() ? 'screen-permission' : monitorState
  })

  const trustedWindow = (event: IpcMainEvent | IpcMainInvokeEvent): BrowserWindow | null => {
    const win = BrowserWindow.fromWebContents(event.sender)

    if (!win || win.isDestroyed() || event.senderFrame !== event.sender.mainFrame) {
      return null
    }

    try {
      const url = new URL(event.senderFrame.url)

      return url.protocol === expectedUrl.protocol && url.host === expectedUrl.host && url.pathname === expectedUrl.pathname ? win : null
    } catch {
      return null
    }
  }

  const publish = () => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:screenshot:status', status())
      }
    }
  }

  const start = (requestPermission = false) => {
    const current = ++generation
    monitor.start(window => {
      if (disposed || !enabled || current !== generation) {
        return
      }

      // Retain the last Hermes chat window when another application takes focus.
      // Closing it cancels this destination; never silently pick another chat.
      const focused = BrowserWindow.getFocusedWindow()
      const recipient = focused && recipients.has(focused.webContents.id) ? focused : lastRecipient

      if (!recipient || recipient.isDestroyed() || !recipients.has(recipient.webContents.id)) {
        return
      }

      const requestId = capture.request(recipient.webContents.id, window)

      if (requestId) {
        recipient.webContents.send('hermes:screenshot:request', requestId)
      }
    }, result => {
      if (disposed || current !== generation) {
        return
      }

      monitorState = result.type === 'error'
        ? result.code === 'permission-required' ? 'input-permission' : 'unavailable'
        : result.type === 'stopped' ? 'disabled' : result.type
      publish()
    }, requestPermission)
  }

  const onFocus = (_event: unknown, win: BrowserWindow) => {
    if (recipients.has(win.webContents.id)) {
      lastRecipient = win
    }
  }

  const onSubscribe = (event: IpcMainEvent, subscribed: unknown) => {
    const win = trustedWindow(event)

    if (!win) {
      return
    }

    if (subscribed === true) {
      if (!recipients.has(event.sender.id)) {
        const id = event.sender.id
        const onDestroyed = () => {
          recipients.delete(id)
          if (lastRecipient === win) {
            lastRecipient = null
          }
        }
        event.sender.once('destroyed', onDestroyed)
        recipients.set(id, () => event.sender.removeListener('destroyed', onDestroyed))
      }

      if (!lastRecipient || BrowserWindow.getFocusedWindow() === win) {
        lastRecipient = win
      }
    } else if (subscribed === false) {
      recipients.get(event.sender.id)?.()
      recipients.delete(event.sender.id)
      if (lastRecipient === win) {
        lastRecipient = null
      }
    }
  }

  const channels: string[] = []
  const handle = (name: string, callback: (event: IpcMainInvokeEvent, value: unknown) => unknown) => {
    const channel = `hermes:screenshot:${name}`
    channels.push(channel)
    ipcMain.handle(channel, async (event, value) => {
      if (!trustedWindow(event)) {
        throw new Error('Screenshot request from an untrusted frame')
      }

      return callback(event, value)
    })
  }

  handle('settings:get', () => status())
  handle('settings:set', async (_event, value) => {
    if (typeof value !== 'boolean') {
      throw new Error('Screenshot setting must be a boolean')
    }

    // Persist first: a failed authoritative write must not leave a live monitor.
    mkdirSync(path.dirname(configPath), { recursive: true })
    writeFileSync(`${configPath}.tmp`, JSON.stringify({ enabled: value }), { mode: 0o600 })
    renameSync(`${configPath}.tmp`, configPath)
    enabled = value
    generation += 1
    capture.clear()
    monitor.stop()

    if (enabled) {
      start(true)
      if (!hasScreenPermission()) {
        // Zero-size thumbnails skip content capture and may never request TCC
        // consent. Request the smallest thumbnail only on explicit opt-in;
        // discard it rather than retaining or attaching permission-probe pixels.
        await desktopCapturer.getSources({ types: ['window'], thumbnailSize: { width: 1, height: 1 }, fetchWindowIcons: false }).catch(() => undefined)
      }
    } else {
      monitorState = 'disabled'
    }

    publish()

    return status()
  })
  handle('capture', async (event, requestId) => {
    const result = await capture.take(event.sender.id, requestId)
    if (result.ok === false && result.reason === 'screen-permission') {
      publish()
    }

    return result
  })
  handle('permission', async (_event, kind) => {
    const pane = kind === 'input' ? 'Privacy_ListenEvent' : kind === 'screen' ? 'Privacy_ScreenCapture' : null
    if (!pane) {
      throw new Error('Unknown screenshot permission')
    }
    await shell.openExternal(`x-apple.systempreferences:com.apple.preference.security?${pane}`)
  })
  ipcMain.on('hermes:screenshot:subscribe', onSubscribe)
  app.on('browser-window-focus', onFocus)

  const dispose = () => {
    disposed = true
    capture.clear()
    monitor.stop()
    recipients.forEach(unsubscribe => unsubscribe())
    recipients.clear()
    lastRecipient = null
    channels.forEach(channel => ipcMain.removeHandler(channel))
    ipcMain.removeListener('hermes:screenshot:subscribe', onSubscribe)
    app.removeListener('browser-window-focus', onFocus)
    app.removeListener('will-quit', dispose)
  }

  app.once('will-quit', dispose)
  if (enabled) {
    start()
  }

  return dispose
}
