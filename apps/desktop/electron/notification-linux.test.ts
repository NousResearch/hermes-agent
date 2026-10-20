import { EventEmitter } from 'node:events'

import type { Message } from 'dbus-native'
import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesNotification } from './notification-types'

const host = vi.hoisted(() => ({ handle: vi.fn(), fromWebContents: vi.fn(), createClient: vi.fn() }))
vi.mock('electron', () => ({
  BrowserWindow: { fromWebContents: host.fromWebContents },
  ipcMain: { handle: host.handle },
  Notification: class extends EventEmitter {
    static isSupported() {
      return true
    }
    show() {}
    close() {}
  }
}))
vi.mock('dbus-native', () => ({
  createClient: host.createClient,
  Variant: class {
    constructor(
      public signature: string,
      public value: unknown
    ) {}
  }
}))

import { registerNativeNotifications } from './notification-ipc'

function setup(alreadyRunning = false) {
  host.handle.mockClear()

  const connection = Object.assign(new EventEmitter(), {
    stream: { destroy: vi.fn(() => connection.emit('close')), unref: vi.fn() }
  })

  let stalled = '',
    owner = ':1.20',
    id = 0,
    activated = alreadyRunning,
    raceOwnerReply = false

  const calls: Message[] = []
  const failures = new Map<string, string>()

  const replaceOwner = () => {
    const old = owner
    owner = ':1.21'
    id = 0
    connection.emit('message', {
      type: 4,
      sender: 'org.freedesktop.DBus',
      path: '/org/freedesktop/DBus',
      interface: 'org.freedesktop.DBus',
      member: 'NameOwnerChanged',
      body: ['org.freedesktop.Notifications', old, owner]
    })
  }

  const invoke = (
    message: Message,
    options: { signal?: AbortSignal; timeout?: number },
    callback?: (error: Error | null, value?: unknown) => void
  ) => {
    calls.push(message)

    const promise = new Promise((resolve, reject) => {
      // Like the real bus: a call addressed to a unique name that has gone away
      // is answered with an error, not routed to the new owner.
      if (message.destination?.startsWith(':') && message.destination !== owner) {
        reject(Object.assign(new Error('no such name'), { dbusName: 'org.freedesktop.DBus.Error.NameHasNoOwner' }))

        return
      }

      if (failures.has(message.member ?? '')) {
        reject(Object.assign(new Error('fixture failure'), { dbusName: failures.get(message.member ?? '') }))

        return
      }

      if (message.member === stalled) {
        const fail = () => reject(new Error('service unavailable'))

        if (options.signal) {
          options.signal.addEventListener('abort', fail, { once: true })
        } else {
          setTimeout(fail, options.timeout ?? 5000)
        }

        return
      }

      if (message.member === 'GetNameOwner' && !activated) {
        reject(Object.assign(new Error('no owner'), { dbusName: 'org.freedesktop.DBus.Error.NameHasNoOwner' }))

        return
      }

      if (message.member === 'StartServiceByName') {
        if (alreadyRunning) {
          reject(
            Object.assign(new Error('no activation file'), { dbusName: 'org.freedesktop.DBus.Error.ServiceUnknown' })
          )

          return
        }

        activated = true
      }

      const values: Record<string, unknown> = {
        Hello: ':1.10',
        StartServiceByName: 1,
        GetNameOwner: owner,
        GetCapabilities: ['actions', 'body'],
        Notify: message.member === 'Notify' ? ++id : id
      }

      const value = values[message.member ?? '']

      // Delivery can be followed by a signal in the same read turn.
      if (callback) {
        callback(null, value)
      }

      resolve(value)

      if (message.member === 'GetNameOwner' && raceOwnerReply) {
        raceOwnerReply = false
        replaceOwner() // Reply followed by owner replacement in the same read batch.
      }
    })

    if (callback) {
      void promise.catch(error => callback(error))
    }

    return promise
  }

  host.createClient.mockReturnValue({ connection, invoke, invokeDbus: invoke })
  const primary = { isDestroyed: vi.fn(() => false), webContents: { send: vi.fn() } }
  const source = { isDestroyed: vi.fn(() => false), webContents: { send: vi.fn() } }
  host.fromWebContents.mockReturnValue(source)
  const focusWindow = vi.fn()

  const { dispose } = registerNativeNotifications({
    getMainWindow: () => primary as unknown as BrowserWindow,
    focusWindow,
    platform: 'linux'
  })

  const notify = (payload: HermesNotification) =>
    Promise.resolve(
      host.handle.mock.calls[0][1]({ sender: source.webContents } as unknown as IpcMainInvokeEvent, payload)
    )

  const signal = (member: string, body: unknown[], sender = owner) =>
    connection.emit('message', {
      type: 4,
      path: '/org/freedesktop/Notifications',
      interface: 'org.freedesktop.Notifications',
      member,
      sender,
      body
    })

  return {
    connection,
    dispose,
    calls,
    primary,
    source,
    focusWindow,
    notify,
    signal,
    stall: (method: string) => {
      stalled = method
    },
    replaceOwner,
    raceOwnerReply: () => {
      raceOwnerReply = true
    },
    fail: (member: string, name: string) => {
      failures.set(member, name)
    },
    lastId: () => id
  }
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubEnv('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/test-notifications')
  vi.stubEnv('ELECTRON_USE_UBUNTU_NOTIFIER', undefined)
  host.handle.mockReset()
  host.fromWebContents.mockReset()
  host.createClient.mockReset()
})
afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllEnvs()
})

it('bounds failed delivery without dropping older callbacks, and retries only on a later request', async () => {
  const unavailable = setup()
  unavailable.fail('StartServiceByName', 'org.freedesktop.DBus.Error.ServiceUnknown')
  expect(await unavailable.notify({ tag: 'test' })).toBe(false)
  expect(await unavailable.notify({ tag: 'test' })).toBe(false)
  expect(unavailable.calls.filter(call => call.member === 'StartServiceByName')).toHaveLength(1)
  unavailable.connection.emit('close')

  // libnotify's getenv guard disables actions even for an empty value.
  vi.stubEnv('ELECTRON_USE_UBUNTU_NOTIFIER', '')
  const unity = setup(true)
  expect(await unity.notify({ tag: 'unity', actions: [{ id: 'ok', text: 'OK' }] })).toBe(true)
  expect(unity.calls.find(call => call.member === 'Notify')?.body?.[5]).toEqual([])
  unity.connection.emit('close')
  vi.stubEnv('ELECTRON_USE_UBUNTU_NOTIFIER', undefined)

  for (const method of ['Hello', 'AddMatch', 'StartServiceByName', 'GetNameOwner', 'GetCapabilities']) {
    const startup = setup()
    startup.stall(method)
    const attempt = startup.notify({ tag: method })
    await vi.advanceTimersByTimeAsync(6000)
    expect(await attempt).toBe(false)
    await vi.advanceTimersByTimeAsync(11000)
    startup.stall('')
    expect(await startup.notify({ tag: 'startup-recovered' })).toBe(true)
    startup.connection.emit('close')
  }

  const h = setup()
  expect(await h.notify({ tag: 'old', focusSessionId: 'old-session' })).toBe(true)
  expect(h.calls.some(call => call.member === 'StartServiceByName')).toBe(true)
  const oldId = h.lastId()
  h.stall('Notify')
  const pending = h.notify({ tag: 'stalled' })
  const duplicate = h.notify({ tag: 'stalled' })
  await vi.advanceTimersByTimeAsync(6000)
  expect(await pending).toBe(false)
  expect(await duplicate).toBe(false)
  const attempts = h.calls.length
  expect(await h.notify({ tag: 'cooldown' })).toBe(false)
  expect(h.calls).toHaveLength(attempts)
  h.signal('ActionInvoked', [oldId, 'default'])
  expect(h.source.webContents.send).toHaveBeenCalledWith('hermes:focus-session', 'old-session')
  const afterConsumption = h.calls.length
  await vi.advanceTimersByTimeAsync(11000)
  expect(h.calls).toHaveLength(afterConsumption) // No automatic replay of an ambiguous Notify.
  h.stall('')
  expect(await h.notify({ tag: 'recovered' })).toBe(true)
  h.stall('CloseNotification')
  const closesBeforeExpiry = h.calls.filter(call => call.member === 'CloseNotification').length
  await vi.advanceTimersByTimeAsync(10 * 60_000 + 6000)
  expect(h.calls.filter(call => call.member === 'CloseNotification')).toHaveLength(closesBeforeExpiry + 1)
  h.connection.emit('close')
})

it('releases naturally closed notifications while retaining other click targets', async () => {
  const h = setup(true)
  expect(await h.notify({ tag: 'closed', focusSessionId: 'closed-session' })).toBe(true)
  const closedId = h.lastId()
  expect(await h.notify({ tag: 'active', focusSessionId: 'active-session' })).toBe(true)
  const activeId = h.lastId()
  await vi.advanceTimersByTimeAsync(1100)
  const timersBeforeClose = vi.getTimerCount()

  h.signal('NotificationClosed', [closedId, 2], ':1.666')
  expect(vi.getTimerCount()).toBe(timersBeforeClose)
  h.signal('NotificationClosed', [closedId, 2])
  expect(vi.getTimerCount()).toBe(timersBeforeClose - 1)
  h.signal('ActionInvoked', [closedId, 'default'])
  expect(h.source.webContents.send).not.toHaveBeenCalled()

  h.signal('ActionInvoked', [activeId, 'default'])
  expect(h.source.webContents.send).toHaveBeenCalledWith('hermes:focus-session', 'active-session')
  expect(vi.getTimerCount()).toBe(0)
  h.connection.emit('close')
})

it('preserves activation, dedupe and source ownership while fencing daemon ID reuse', async () => {
  const race = setup(true)
  race.raceOwnerReply()
  expect(await race.notify({ tag: 'obsolete-owner', focusSessionId: 'must-not-open' })).toBe(false)
  expect(race.calls.filter(call => call.member === 'Notify')).toHaveLength(0)
  // A daemon swap is not a daemon failure: the next notification goes to the
  // new owner right away instead of sitting out the failure cooldown.
  expect(await race.notify({ tag: 'after-race' })).toBe(true)
  expect(race.calls.filter(call => call.member === 'Notify')).toHaveLength(1)
  race.connection.emit('close')

  const h = setup(true)

  const payload = {
    kind: 'approval',
    sessionId: 'runtime',
    focusSessionId: 'stored',
    silent: true,
    actions: [
      { id: 'approve', text: 'Approve' },
      { id: 'reject', text: 'Reject' }
    ]
  }

  expect(await h.notify(payload)).toBe(true)
  const firstId = h.lastId()
  expect(await h.notify(payload)).toBe(true)
  expect(h.calls.filter(call => call.member === 'Notify')).toHaveLength(1)
  h.signal('ActionInvoked', [firstId, '1'], ':1.666')
  expect(h.source.webContents.send).not.toHaveBeenCalled()
  h.fail('CloseNotification', 'org.freedesktop.DBus.Error.InvalidArgs')
  h.signal('ActionInvoked', [firstId, '1'])
  expect(h.source.webContents.send).toHaveBeenCalledWith('hermes:notification-action', {
    sessionId: 'runtime',
    actionId: 'reject'
  })
  expect(h.primary.webContents.send).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(0)
  expect(await h.notify({ tag: 'plugin', notifyId: 'source-callback', activate: '/plugin' })).toBe(true)
  const pluginId = h.lastId()
  h.source.isDestroyed.mockReturnValue(true)
  h.signal('ActionInvoked', [pluginId, 'default'])
  expect(h.primary.webContents.send).toHaveBeenCalledWith('hermes:notification-activate', {
    activate: '/plugin',
    notifyId: undefined,
    tag: 'plugin'
  })
  h.connection.emit('close')

  const reused = setup(true)
  expect(await reused.notify({ tag: 'old-owner', focusSessionId: 'must-not-open' })).toBe(true)
  const retainedId = reused.lastId()
  reused.replaceOwner()
  expect(await reused.notify({ tag: 'new-owner', focusSessionId: 'new-session' })).toBe(true)
  expect(reused.lastId()).toBe(retainedId)
  reused.signal('ActionInvoked', [retainedId, 'default'], ':1.20')
  expect(reused.source.webContents.send).not.toHaveBeenCalled()
  reused.signal('ActionInvoked', [retainedId, 'default'])
  expect(reused.source.webContents.send).toHaveBeenCalledWith('hermes:focus-session', 'new-session')

  // Quit teardown closes the bus; a delivered notification's callbacks die with it.
  reused.dispose()
  expect(reused.connection.stream.destroy).toHaveBeenCalled()
  reused.signal('ActionInvoked', [retainedId, 'default'])
  expect(reused.source.webContents.send).toHaveBeenCalledTimes(1)
})
