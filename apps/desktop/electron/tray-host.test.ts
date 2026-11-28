import { EventEmitter } from 'node:events'

import { expect, test, vi } from 'vitest'

const createClient = vi.hoisted(() => vi.fn())
vi.mock('dbus-native', () => ({ createClient }))

import { watchLinuxTrayHost } from './tray-host'

function bus(hostRegistered: boolean) {
  const connection = Object.assign(new EventEmitter(), { stream: { destroy: vi.fn() } })
  const invoke = vi.fn(async (): Promise<unknown> => ({ signature: 'b', value: hostRegistered }))

  const instance = {
    connection,
    invoke,
    invokeDbus: vi.fn(async ({ member }: { member: string }) => (member === 'GetNameOwner' ? ':1.42' : ':1.99'))
  }

  createClient.mockReturnValue(instance)

  return instance
}

test('a registered host is required, and losing its owner reports loss exactly once', async () => {
  const instance = bus(true)
  const lost = vi.fn()
  const dispose = await watchLinuxTrayHost(lost)
  expect(lost).not.toHaveBeenCalled()
  instance.connection.emit('message', {
    type: 4,
    sender: 'org.freedesktop.DBus',
    interface: 'org.freedesktop.DBus',
    member: 'NameOwnerChanged',
    body: ['org.kde.StatusNotifierWatcher', ':1.42', '']
  })
  instance.connection.emit('close')
  expect(lost).toHaveBeenCalledOnce()
  expect(instance.connection.stream.destroy).toHaveBeenCalledOnce()
  dispose()
})

function signal(instance: ReturnType<typeof bus>, sender: string, member: string, body?: unknown[]) {
  instance.connection.emit('message', {
    type: 4,
    sender,
    path: '/StatusNotifierWatcher',
    interface: 'org.kde.StatusNotifierWatcher',
    member,
    body
  })
}

function ownerChanged(instance: ReturnType<typeof bus>, from: string, to: string) {
  instance.connection.emit('message', {
    type: 4,
    sender: 'org.freedesktop.DBus',
    interface: 'org.freedesktop.DBus',
    member: 'NameOwnerChanged',
    body: ['org.kde.StatusNotifierWatcher', from, to]
  })
}

const registrations = (instance: ReturnType<typeof bus>) =>
  instance.invoke.mock.calls
    .map(([message]: any[]) => message)
    .filter(message => message?.member === 'RegisterStatusNotifierItem')

test('a host that returns after a screen lock is reported back once its item is re-registered (#145282)', async () => {
  vi.useFakeTimers()

  try {
    const instance = bus(true)
    const change = vi.fn()
    const dispose = await watchLinuxTrayHost(change)
    const item = `org.freedesktop.StatusNotifierItem-${process.pid}-1`
    signal(instance, ':1.42', 'StatusNotifierItemRegistered', [item])

    signal(instance, ':1.42', 'StatusNotifierHostUnregistered')
    ownerChanged(instance, ':1.42', '')
    ownerChanged(instance, '', ':1.43')
    signal(instance, ':1.43', 'StatusNotifierHostRegistered')
    expect(change.mock.calls).toEqual([[false]])
    expect(registrations(instance)).toEqual([])

    await vi.advanceTimersByTimeAsync(1000)
    expect(registrations(instance)).toEqual([expect.objectContaining({ destination: ':1.43', body: [item] })])
    expect(change.mock.calls).toEqual([[false], [true]])

    // Later loss is still seen from the new owner.
    signal(instance, ':1.43', 'StatusNotifierHostUnregistered')
    expect(change.mock.calls).toEqual([[false], [true], [false]])
    expect(instance.connection.stream.destroy).not.toHaveBeenCalled()
    dispose()
  } finally {
    vi.useRealTimers()
  }
})

test('a host that leaves again before the re-registration delay stays lost', async () => {
  vi.useFakeTimers()

  try {
    const instance = bus(true)
    const change = vi.fn()
    const dispose = await watchLinuxTrayHost(change)
    signal(instance, ':1.42', 'StatusNotifierHostUnregistered')
    signal(instance, ':1.42', 'StatusNotifierHostRegistered')
    signal(instance, ':1.42', 'StatusNotifierHostUnregistered')

    await vi.advanceTimersByTimeAsync(1000)
    expect(change.mock.calls).toEqual([[false]])
    dispose()
  } finally {
    vi.useRealTimers()
  }
})

test('dbus-native 0.15 returns the registered flag as a bare boolean', async () => {
  const instance = bus(true)
  instance.invoke.mockResolvedValue(true)
  const lost = vi.fn()
  const dispose = await watchLinuxTrayHost(lost)
  expect(lost).not.toHaveBeenCalled()
  dispose()
})

test('a missing host and a failed query reject without leaving an open bus connection', async () => {
  const lost = vi.fn()
  const absent = bus(false)
  await expect(watchLinuxTrayHost(lost)).rejects.toThrow('No system tray host')
  expect(absent.connection.stream.destroy).toHaveBeenCalledOnce()
  expect(lost).not.toHaveBeenCalled()
  const failed = bus(true)
  failed.invoke.mockRejectedValue(new Error('D-Bus unavailable'))
  await expect(watchLinuxTrayHost(lost)).rejects.toThrow('D-Bus unavailable')
  expect(failed.connection.stream.destroy).toHaveBeenCalledOnce()
})
