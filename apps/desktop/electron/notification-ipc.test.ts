import { EventEmitter } from 'node:events'

import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { beforeEach, expect, it, vi } from 'vitest'

import type { HermesNotification } from './notification-types'

const host = vi.hoisted(() => ({
  handle: vi.fn(),
  fromWebContents: vi.fn(),
  shown: [] as EventEmitter[],
  showEvent: 'show' as 'show' | 'failed' | 'none'
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

      if (host.showEvent !== 'none') {
        queueMicrotask(() => this.emit(host.showEvent))
      }
    }
  }
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
  host.showEvent = 'show'
})

it('reports a macOS native failure after show returns', async () => {
  host.showEvent = 'failed'
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

  expect(await notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {
    kind: 'test', title: 'Test'
  })).toBe(false)
})

it('bounds a macOS notification with no native delivery event', async () => {
  vi.useFakeTimers()

  try {
    host.showEvent = 'none'
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

    const delivery = notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {
      kind: 'test', title: 'Test'
    })

    await vi.advanceTimersByTimeAsync(5000)
    expect(await delivery).toBe(false)
  } finally {
    vi.useRealTimers()
  }
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
