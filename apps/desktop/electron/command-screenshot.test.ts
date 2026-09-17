import { EventEmitter } from 'node:events'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({ start: vi.fn(), stop: vi.fn() }))
const electron = vi.hoisted(() => ({ handlers: new Map(), windows: [] as any[], focused: null as any, screenPermission: 'granted', directory: '' }))
vi.mock('./command-screenshot-monitor', () => ({ CommandScreenshotMonitor: class { start = native.start; stop = native.stop } }))
vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events')
  const ipcMain = new EventEmitter() as any
  ipcMain.handle = (channel: string, handler: unknown) => electron.handlers.set(channel, handler)
  ipcMain.removeHandler = (channel: string) => electron.handlers.delete(channel)

  return {
    app: Object.assign(new EventEmitter(), { getPath: () => electron.directory, getAppPath: () => '/app' }),
    ipcMain,
    BrowserWindow: {
      fromWebContents: (wc: unknown) => electron.windows.find(win => win.webContents === wc),
      getFocusedWindow: () => electron.focused,
      getAllWindows: () => electron.windows
    },
    desktopCapturer: { getSources: vi.fn(async () => [{ id: 'window:42:0', thumbnail: { isEmpty: () => false, toPNG: () => new Uint8Array([1]) } }]) },
    systemPreferences: { getMediaAccessStatus: () => electron.screenPermission },
    shell: { openExternal: vi.fn() }
  }
})

import { app, desktopCapturer, ipcMain } from 'electron'

import { installCommandScreenshot } from './command-screenshot'

const cleanups: (() => void)[] = []
afterEach(async () => {
  cleanups.splice(0).forEach(fn => fn())
  await rm(electron.directory, { recursive: true, force: true })
  electron.windows = []
  electron.focused = null
  electron.screenPermission = 'granted'
  vi.clearAllMocks()
})

function window(id: number, url = 'http://127.0.0.1:5174/') {
  const frame = { url }
  const wc = Object.assign(new EventEmitter(), { id, mainFrame: frame, getURL: () => url, isDestroyed: () => false, send: vi.fn() })
  const win = { webContents: wc, isDestroyed: () => false }
  electron.windows.push(win)

  return { win, event: { sender: wc, senderFrame: frame }, wc }
}

async function setup() {
  electron.directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-screenshot-'))
  cleanups.push(installCommandScreenshot({ rendererUrl: 'http://127.0.0.1:5174/' }))
}

const call = (channel: string, event: unknown, ...args: unknown[]) => electron.handlers.get(`hermes:screenshot:${channel}`)(event, ...args)

describe.skipIf(process.platform !== 'darwin')('Command screenshot native bridge', () => {
  it('persists opt-in, routes to the last focused subscribed window while backgrounded, and revokes on disable', async () => {
    await setup()
    const first = window(1)
    const second = window(2)
    expect(await call('settings:get', first.event)).toEqual({ enabled: false, state: 'disabled' })
    expect(native.start).not.toHaveBeenCalled()
    ipcMain.emit('subscribe', first.event) // unrelated IPC grants nothing
    ipcMain.emit('hermes:screenshot:subscribe', first.event, true)
    ipcMain.emit('hermes:screenshot:subscribe', second.event, true)
    app.emit('browser-window-focus', {}, second.win)
    electron.focused = null
    await call('settings:set', first.event, true)
    expect(JSON.parse(await readFile(path.join(electron.directory, 'screenshot.json'), 'utf8'))).toEqual({ enabled: true })
    const [capture, status] = native.start.mock.calls.at(-1)!
    status({ type: 'ready' })
    capture({ type: 'capture', windowId: 42, width: 600, height: 400 })
    const requests = second.wc.send.mock.calls.filter(([channel]) => channel === 'hermes:screenshot:request')
    expect(requests).toHaveLength(1)
    expect(first.wc.send.mock.calls.some(([channel]) => channel === 'hermes:screenshot:request')).toBe(false)
    expect(await call('capture', first.event, requests[0]![1])).toEqual({ ok: false, reason: 'expired' })
    await call('settings:set', second.event, false)
    expect(await call('capture', second.event, requests[0]![1])).toEqual({ ok: false, reason: 'expired' })
    expect(native.stop).toHaveBeenCalled()
    for (let i = 0; i < 3; i += 1) {
      ipcMain.emit('hermes:screenshot:subscribe', second.event, false)
      ipcMain.emit('hermes:screenshot:subscribe', second.event, true)
    }
    expect(second.wc.listenerCount('destroyed')).toBe(1)
    ipcMain.emit('hermes:screenshot:subscribe', second.event, false)
    expect(second.wc.listenerCount('destroyed')).toBe(0)
  })

  it('rejects guest/foreign frames and reports denied screen access rather than ready', async () => {
    await setup()
    const trusted = window(3)
    const foreign = window(4, 'https://example.org/')
    await expect(call('settings:set', foreign.event, true)).rejects.toThrow()
    await expect(call('settings:set', { ...trusted.event, senderFrame: { url: 'https://example.org/' } }, true)).rejects.toThrow()
    expect(native.start).not.toHaveBeenCalled()
    electron.screenPermission = 'denied'
    await call('settings:set', trusted.event, true)
    native.start.mock.calls.at(-1)![1]({ type: 'ready' })
    expect(desktopCapturer.getSources).toHaveBeenCalledWith({
      types: ['window'], thumbnailSize: { width: 1, height: 1 }, fetchWindowIcons: false
    })
    expect(await call('settings:get', trusted.event)).toEqual({ enabled: true, state: 'screen-permission' })
  })
})
