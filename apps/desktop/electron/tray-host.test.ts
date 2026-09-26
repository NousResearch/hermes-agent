import { EventEmitter } from 'node:events'

import { expect, test, vi } from 'vitest'

const createClient = vi.hoisted(() => vi.fn())
vi.mock('dbus-native', () => ({ createClient }))

import { watchLinuxTrayHost } from './tray-host'

function bus(hostRegistered: boolean) {
  const connection = Object.assign(new EventEmitter(), { stream: { destroy: vi.fn() } })
  const invoke = vi.fn(async () => ({ signature: 'b', value: hostRegistered }))

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
