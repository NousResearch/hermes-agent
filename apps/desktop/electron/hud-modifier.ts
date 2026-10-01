import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, ipcMain, powerMonitor, shell } from 'electron'
import type { IpcMainInvokeEvent } from 'electron'

import { HudModifierMonitor } from './hud-modifier-monitor'
import type { HudModifierStatus } from './hud-modifier-types'

/** Opt-in device preference: native input never crosses into a renderer. */
export function installHudModifierTap({
  rendererUrl,
  summon
}: {
  rendererUrl: string
  summon: () => void
}): () => void {
  const configPath = path.join(app.getPath('userData'), 'hud-modifier.json')
  const expectedUrl = new URL(rendererUrl)
  const monitor = new HudModifierMonitor({ appPath: app.getAppPath() })
  let enabled = false
  let state: HudModifierStatus['state'] = 'disabled'
  let reason: HudModifierStatus['reason']
  let disposed = false
  let generation = 0

  try {
    enabled = JSON.parse(readFileSync(configPath, 'utf8')).enabled === true
  } catch {
    // An absent or unreadable preference never grants input monitoring.
  }

  const status = (): HudModifierStatus => ({ enabled, state, ...(reason ? { reason } : {}) })

  const publish = () => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:hud-modifier:status', status())
      }
    }
  }

  const stop = () => {
    generation += 1
    monitor.stop()
    state = 'disabled'
    reason = undefined
  }

  const start = (requestPermission = false) => {
    const current = ++generation
    state = 'starting'
    reason = undefined
    monitor.start(
      () => {
        if (!disposed && enabled && state === 'ready' && current === generation) {
          summon()
        }
      },
      result => {
        if (disposed || current !== generation) {
          return
        }

        reason = result.type === 'error' ? result.reason : undefined
        state =
          result.type === 'error'
            ? result.code === 'permission-required'
              ? 'input-permission'
              : 'unavailable'
            : result.type === 'stopped'
              ? 'disabled'
              : result.type
        publish()
      },
      requestPermission
    )
  }

  const trusted = (event: IpcMainInvokeEvent) => {
    const win = BrowserWindow.fromWebContents(event.sender)

    if (!win || win.isDestroyed() || event.senderFrame !== event.sender.mainFrame) {
      return false
    }

    try {
      const url = new URL(event.senderFrame.url)

      return (
        url.protocol === expectedUrl.protocol && url.host === expectedUrl.host && url.pathname === expectedUrl.pathname
      )
    } catch {
      return false
    }
  }

  const handlers: Record<string, (value: unknown) => unknown> = {
    'settings:get': status,
    'settings:set': value => {
      if (typeof value !== 'boolean') {
        throw new Error('HUD gesture setting must be a boolean')
      }

      // Persist before changing the listener; a failed write leaves the old setting live.
      mkdirSync(path.dirname(configPath), { recursive: true })
      writeFileSync(`${configPath}.tmp`, JSON.stringify({ enabled: value }), { mode: 0o600 })
      renameSync(`${configPath}.tmp`, configPath)
      enabled = value
      stop()

      if (enabled) {
        start(true)
      }

      publish()

      return status()
    },
    permission: async () => {
      if (process.platform === 'darwin') {
        await shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent')
      }
    }
  }

  for (const [name, handler] of Object.entries(handlers)) {
    ipcMain.handle(`hermes:hud-modifier:${name}`, async (event, value) => {
      if (!trusted(event)) {
        throw new Error('HUD gesture request from an untrusted frame')
      }

      return handler(value)
    })
  }

  // Never complete a chord across a sleep/lock boundary.
  const suspend = () => {
    stop()
    publish()
  }

  const resume = () => {
    if (enabled && !disposed) {
      start()
    }
  }

  powerMonitor.on('suspend', suspend)
  powerMonitor.on('lock-screen', suspend)
  powerMonitor.on('resume', resume)
  powerMonitor.on('unlock-screen', resume)

  const dispose = () => {
    disposed = true
    stop()
    Object.keys(handlers).forEach(name => ipcMain.removeHandler(`hermes:hud-modifier:${name}`))
    powerMonitor.removeListener('suspend', suspend)
    powerMonitor.removeListener('lock-screen', suspend)
    powerMonitor.removeListener('resume', resume)
    powerMonitor.removeListener('unlock-screen', resume)
    app.removeListener('will-quit', dispose)
  }

  app.once('will-quit', dispose)

  if (enabled) {
    start()
  }

  return dispose
}
