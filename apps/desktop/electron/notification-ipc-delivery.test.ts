import { EventEmitter } from 'node:events'

import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesNotification } from './notification-types'

const host = vi.hoisted(() => ({
  handle: vi.fn(),
  fromWebContents: vi.fn(),
  supported: true,
  shown: [] as EventEmitter[]
}))

vi.mock('electron', () => ({
  BrowserWindow: { fromWebContents: host.fromWebContents },
  ipcMain: { handle: host.handle },
  Notification: class extends EventEmitter {
    static isSupported() {
      return host.supported
    }
    show() {
      host.shown.push(this)
    }
    close() {}
  }
}))

import { registerNativeNotifications } from './notification-ipc'

beforeEach(() => {
  vi.useFakeTimers()
  host.handle.mockReset()
  host.fromWebContents.mockReset()
  host.supported = true
  host.shown.length = 0
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
})

function setup() {
  const source = { isDestroyed: () => false, webContents: { send: vi.fn() } }
  host.fromWebContents.mockReturnValue(source)
  const focusWindow = vi.fn()
  registerNativeNotifications({
    getMainWindow: () => source as unknown as BrowserWindow,
    focusWindow,
    platform: 'win32'
  })

  return {
    source,
    focusWindow,
    notify: host.handle.mock.calls[0][1] as (event: IpcMainInvokeEvent, payload: HermesNotification) => Promise<boolean>
  }
}

it('shares a pending Windows outcome beyond the one-second dedupe interval', async () => {
  const { source, notify } = setup()
  const event = { sender: source.webContents } as unknown as IpcMainInvokeEvent
  const payload = { kind: 'plugin', tag: 'durable-event', title: 'Synthetic' }
  const outcomes = vi.fn()
  const first = notify(event, payload).then(outcomes)
  await vi.advanceTimersByTimeAsync(1500)
  const second = notify(event, payload).then(outcomes)
  expect(host.shown).toHaveLength(1)
  expect(outcomes).not.toHaveBeenCalled()
  host.shown[0].emit('show')
  await Promise.all([first, second])
  expect(outcomes.mock.calls).toEqual([[true], [true]])
})

it('returns the shared failure and permits a later retry without marking it successful', async () => {
  const { source, notify } = setup()
  const event = { sender: source.webContents } as unknown as IpcMainInvokeEvent
  const payload = { kind: 'plugin', tag: 'retry-event' }
  const first = notify(event, payload)
  const peer = notify(event, payload)
  host.shown[0].emit('failed', {}, 'synthetic')
  expect(await first).toBe(false)
  expect(await peer).toBe(false)
  await vi.advanceTimersByTimeAsync(1000)
  const retry = notify(event, payload)
  expect(host.shown).toHaveLength(2)
  host.shown[1].emit('show')
  expect(await retry).toBe(true)
})

it('keeps the captured session click route after confirmed delivery', async () => {
  const { source, focusWindow, notify } = setup()

  const delivery = notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {
    kind: 'turn-complete',
    sessionId: 'runtime',
    focusSessionId: 'stored'
  })

  host.shown[0].emit('show')
  expect(await delivery).toBe(true)
  host.shown[0].emit('click')
  expect(focusWindow).toHaveBeenCalledWith(source)
  expect(source.webContents.send).toHaveBeenCalledWith('hermes:focus-session', 'stored')
})

it('refuses unsupported notifications before invoking native show', async () => {
  const { source, notify } = setup()
  host.supported = false
  expect(await notify({ sender: source.webContents } as unknown as IpcMainInvokeEvent, {})).toBe(false)
  expect(host.shown).toHaveLength(0)
})
