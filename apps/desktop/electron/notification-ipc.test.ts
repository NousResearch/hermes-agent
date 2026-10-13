import { EventEmitter } from 'node:events'

import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { beforeEach, expect, it, vi } from 'vitest'

import type { HermesNotification } from './notification-types'

const host = vi.hoisted(() => ({
  handle: vi.fn(),
  fromWebContents: vi.fn(),
  deliver: vi.fn<(notification: EventEmitter) => void>(),
  linuxCreate: vi.fn(),
  shown: [] as EventEmitter[]
}))

vi.mock('electron', () => ({
  BrowserWindow: { fromWebContents: host.fromWebContents },
  ipcMain: { handle: host.handle },
  Notification: class extends EventEmitter {
    static isSupported() {
      return true
    }

    show() {
      host.shown.push(this)
      host.deliver(this)
    }
  }
}))

vi.mock('./notification-linux', () => ({
  createLinuxNotifications: () => ({ create: host.linuxCreate, dispose: vi.fn() })
}))

import { registerNativeNotifications } from './notification-ipc'

function windowStub() {
  return {
    isDestroyed: vi.fn(() => false),
    webContents: { isDestroyed: vi.fn(() => false), send: vi.fn() }
  }
}

beforeEach(() => {
  host.handle.mockReset()
  host.fromWebContents.mockReset()
  host.shown.length = 0
  host.deliver.mockReset()
  host.deliver.mockImplementation(notification => notification.emit('show'))
  host.linuxCreate.mockReset()
})

it('returns native clicks and approval actions to the emitting window, not the primary', async () => {
  const primary = windowStub()
  const source = windowStub()
  host.fromWebContents.mockReturnValue(source)
  const focusWindow = vi.fn()
  registerNativeNotifications({
    getMainWindow: () => primary as unknown as BrowserWindow,
    focusWindow,
    platform: 'darwin'
  })

  const notify = host.handle.mock.calls[0][1] as (
    event: IpcMainInvokeEvent,
    payload: HermesNotification
  ) => Promise<boolean>

  const payload = {
    kind: 'approval',
    sessionId: 'runtime-source',
    focusSessionId: 'stored-source',
    title: 'Approval',
    actions: [
      { id: 'approve', text: 'Approve' },
      { id: 'reject', text: 'Reject' }
    ]
  }

  expect(await notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, payload)).toBe(true)
  expect(focusWindow).not.toHaveBeenCalled()
  expect(primary.webContents.send).not.toHaveBeenCalled()
  host.shown[0].emit('click')
  expect(focusWindow).toHaveBeenCalledWith(source)
  expect(source.webContents.send).toHaveBeenCalledWith('hermes:focus-session', payload.focusSessionId)
  host.shown[0].emit('action', { actionIndex: 1 }, undefined)
  expect(source.webContents.send).toHaveBeenCalledWith('hermes:notification-action', {
    sessionId: payload.sessionId,
    actionId: 'reject'
  })
  expect(primary.webContents.send).not.toHaveBeenCalled()

  // A response must never jump to another window's gateway after its owner closes.
  source.isDestroyed.mockReturnValue(true)
  host.shown[0].emit('action', { actionIndex: 0 }, undefined)
  expect(primary.webContents.send).not.toHaveBeenCalled()
  host.shown[0].emit('click')
  expect(primary.webContents.send).toHaveBeenCalledWith('hermes:focus-session', payload.focusSessionId)
})

it('delivers plugin callbacks to their source and falls back only for navigation after it closes', async () => {
  const primary = windowStub()
  const source = windowStub()
  host.fromWebContents.mockReturnValue(source)
  const focusWindow = vi.fn()
  registerNativeNotifications({
    getMainWindow: () => primary as unknown as BrowserWindow,
    focusWindow,
    platform: 'darwin'
  })

  const notify = host.handle.mock.calls[0][1] as (
    event: IpcMainInvokeEvent,
    payload: HermesNotification
  ) => Promise<boolean>

  await notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {
    kind: 'plugin',
    notifyId: 'source-callback',
    activate: '/plugin',
    actions: [{ id: 'open', text: 'Open', activate: '/plugin/detail' }]
  })
  host.shown[0].emit('action', { actionIndex: 0 }, undefined)
  expect(source.webContents.send).toHaveBeenCalledWith(
    'hermes:notification-activate',
    expect.objectContaining({
      actionId: 'open',
      notifyId: 'source-callback',
      activate: '/plugin/detail'
    })
  )
  expect(primary.webContents.send).not.toHaveBeenCalled()
  source.isDestroyed.mockReturnValue(true)
  host.shown[0].emit('click')
  expect(primary.webContents.send).toHaveBeenCalledWith(
    'hermes:notification-activate',
    expect.objectContaining({
      activate: '/plugin',
      notifyId: undefined
    })
  )
})

it('shares pending native delivery with peer renderers and reports asynchronous failure', async () => {
  host.deliver.mockImplementation(() => undefined)
  const source = windowStub()
  host.fromWebContents.mockReturnValue(source)
  registerNativeNotifications({
    getMainWindow: () => source as unknown as BrowserWindow,
    focusWindow: vi.fn(),
    platform: 'darwin'
  })

  const notify = host.handle.mock.calls[0][1] as (
    event: IpcMainInvokeEvent,
    payload: HermesNotification
  ) => Promise<boolean>

  const event = { sender: source.webContents } as unknown as IpcMainInvokeEvent
  const payload = { kind: 'turnDone', sessionId: 'shared-pending', title: 'Done' }
  const first = notify(event, payload)
  const peer = notify(event, payload)
  const settled = vi.fn()
  void first.then(settled)
  await Promise.resolve()
  expect(settled).not.toHaveBeenCalled()
  expect(host.shown).toHaveLength(1)
  host.shown[0].emit('failed', {}, 'Application is not signed')
  await expect(first).resolves.toBe(false)
  await expect(peer).resolves.toBe(false)
})

it('keeps Linux delivery governed by its asynchronous boolean result', async () => {
  const source = windowStub()
  host.fromWebContents.mockReturnValue(source)
  let finish!: (delivered: boolean) => void

  const delivered = new Promise<boolean>(resolve => {
    finish = resolve
  })

  const notification = Object.assign(new EventEmitter(), { show: vi.fn(() => delivered), close: vi.fn() })
  host.linuxCreate.mockReturnValue(notification)
  registerNativeNotifications({
    getMainWindow: () => source as unknown as BrowserWindow,
    focusWindow: vi.fn(),
    platform: 'linux'
  })

  const notify = host.handle.mock.calls[0][1] as (
    event: IpcMainInvokeEvent,
    payload: HermesNotification
  ) => Promise<boolean>

  const result = notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {
    kind: 'turnDone',
    sessionId: 'linux-delivery',
    title: 'Done'
  })

  expect(host.shown).toHaveLength(0)
  finish(false)
  await expect(result).resolves.toBe(false)
})
