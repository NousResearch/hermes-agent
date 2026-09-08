import { EventEmitter } from 'node:events'

import type { BrowserWindow, BrowserWindowConstructorOptions } from 'electron'
import { expect, it, vi } from 'vitest'

import * as api from './browser-windows'

function fakeWindow() {
  const webContents = Object.assign(new EventEmitter(), {
    setWindowOpenHandler: vi.fn(),
    session: { setPermissionRequestHandler: vi.fn(), setPermissionCheckHandler: vi.fn(), on: vi.fn() }
  })

  const win = Object.assign(new EventEmitter(), {
    webContents,
    isDestroyed: () => false,
    loadURL: vi.fn(async () => {}),
    showInactive: vi.fn(),
    focus: vi.fn(),
    setTitle: vi.fn(),
    setMenu: vi.fn(),
    destroy: vi.fn()
  })

  win.destroy.mockImplementation(() => {
    win.emit('closed')
  })

  return win
}

it('permits only the main app frame over IPC and tears down viewers on owner reload', async () => {
  expect(api.registerPluginViewerIpc).toBeTypeOf('function')
  const handlers = new Map<string, (...args: any[]) => any>()
  const ipc = { handle: (name: string, handler: (...args: any[]) => any) => handlers.set(name, handler) }
  const win = fakeWindow()
  api.registerPluginViewerIpc(
    ipc,
    () => win as unknown as BrowserWindow,
    () => 'http://127.0.0.1:5174/'
  )
  const mainFrame = { url: 'http://127.0.0.1:5174/?win=secondary#/chat' }
  const sender = Object.assign(new EventEmitter(), { id: 4, mainFrame })
  const event = { sender, senderFrame: mainFrame }
  const open = handlers.get('hermes:window:openPluginViewer')!
  expect(
    await open({ ...event, senderFrame: { url: 'https://evil.example/' } }, 'demo', {
      id: 'watch',
      url: 'http://127.0.0.1:9876/viewer',
      title: 'view'
    })
  ).toBe(false)
  expect(await open(event, 'demo', { id: 'watch', url: 'http://127.0.0.1:9876/viewer', title: 'view' })).toBe(true)
  const fileHandlers = new Map<string, (...args: any[]) => any>()
  api.registerPluginViewerIpc(
    {
      handle: (name, handler) => {
        fileHandlers.set(name, handler)
      }
    },
    () => win as unknown as BrowserWindow,
    () => 'file:///opt/hermes/index.html'
  )
  const impostorFrame = { url: 'data:/opt/hermes/index.html' }
  expect(
    await fileHandlers.get('hermes:window:openPluginViewer')!(
      { sender: Object.assign(new EventEmitter(), { id: 7, mainFrame: impostorFrame }), senderFrame: impostorFrame },
      'demo',
      { id: 'watch', url: 'https://example.org', title: 'view' }
    )
  ).toBe(false)
  sender.emit('did-start-navigation', {}, 'http://127.0.0.1:5174/', false, true)
  expect(win.destroy).toHaveBeenCalledOnce()
})

it('does not let an obsolete load failure destroy a newer same-key viewer', async () => {
  const win = fakeWindow()
  const viewers = api.createPluginViewerWindows(() => win as unknown as BrowserWindow)
  const input = { id: 'watch', title: 'Session A', url: 'http://127.0.0.1:9234/viewer?ticket=one' }
  let rejectOld!: (error: Error) => void
  win.loadURL.mockImplementationOnce(
    () =>
      new Promise<void>((_resolve, reject) => {
        rejectOld = reject
      })
  )

  const older = viewers.open(1, 'demo', input)
  expect(await viewers.open(1, 'demo', { ...input, url: input.url.replace('one', 'two') })).toBe(true)
  rejectOld(new Error('ERR_ABORTED'))
  expect(await older).toBe(false)
  expect(win.destroy).not.toHaveBeenCalled()

  // A failure of the current request still owns cleanup.
  win.loadURL.mockRejectedValueOnce(new Error('ERR_CONNECTION_REFUSED'))
  expect(await viewers.open(1, 'demo', input)).toBe(false)
  expect(win.destroy).toHaveBeenCalledOnce()
})

it('owns passive sandboxed viewer windows and rejects unsafe options before creation', async () => {
  expect(api.createPluginViewerWindows).toBeTypeOf('function')
  const windows: ReturnType<typeof fakeWindow>[] = []

  const factory = vi.fn((_options: BrowserWindowConstructorOptions) => {
    const win = fakeWindow()
    windows.push(win)

    return win as unknown as BrowserWindow
  })

  const viewers = api.createPluginViewerWindows(factory)
  const input = { id: 'watch', title: 'Session A', url: 'http://127.0.0.1:9234/viewer?ticket=one' }
  expect(await viewers.open(1, 'demo', input)).toBe(true)
  windows[0].emit('ready-to-show')
  expect(windows[0].showInactive).toHaveBeenCalledOnce()
  expect(factory.mock.calls[0][0]).toMatchObject({
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webviewTag: false,
      focusOnNavigation: false
    }
  })
  expect(factory.mock.calls[0][0].webPreferences).not.toHaveProperty('preload')
  expect(await viewers.open(1, 'demo', { ...input, url: input.url.replace('one', 'two') })).toBe(true)
  expect(factory).toHaveBeenCalledTimes(1)
  expect(windows[0].focus).not.toHaveBeenCalled()
  expect(await viewers.open(2, 'demo', input)).toBe(true)
  expect(factory).toHaveBeenCalledTimes(2)
  expect(await viewers.open(1, 'demo', { ...input, webPreferences: { nodeIntegration: true } })).toBe(false)
  expect(await viewers.open(1, 'demo', { ...input, url: 'file:///etc/passwd' })).toBe(false)
  expect(await viewers.open(1, 'demo', { ...input, id: '../unsafe' })).toBe(false)
  expect(await viewers.open(1, 'demo', { ...input, url: 'https://other.example/viewer' })).toBe(false)
  const event = { preventDefault: vi.fn() }
  windows[0].webContents.emit('will-navigate', event, 'https://other.example')
  expect(event.preventDefault).toHaveBeenCalledOnce()
  viewers.closeOwner(1)
  expect(windows[0].destroy).toHaveBeenCalledOnce()
  expect(windows[1].destroy).not.toHaveBeenCalled()
})
