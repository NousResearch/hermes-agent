import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

const electron = vi.hoisted(() => ({
  contextBridge: { exposeInMainWorld: vi.fn() },
  ipcRenderer: {
    invoke: vi.fn(async () => ({ ok: true })),
    on: vi.fn(),
    removeListener: vi.fn(),
    send: vi.fn(),
    sendSync: vi.fn(() => ({}))
  },
  webFrame: {},
  webUtils: {}
}))

vi.mock('electron', () => electron)

test('the native preload exposes routed peer opening and default preference events with unsubscribe', async () => {
  await import('./preload')
  const [name, bridge] = electron.contextBridge.exposeInMainWorld.mock.calls[0]
  assert.equal(name, 'hermesDesktop')
  const route = { connectionId: 'remote-work', profile: 'work' }

  await bridge.openWindow(route)
  assert.deepEqual(electron.ipcRenderer.invoke.mock.lastCall, ['hermes:window:openInstance', route])
  await bridge.openWindow()
  assert.deepEqual(electron.ipcRenderer.invoke.mock.lastCall, ['hermes:window:openInstance', undefined])
  await bridge.profile.setDefault(route)
  assert.deepEqual(electron.ipcRenderer.invoke.mock.lastCall, ['hermes:profile:default:set', route])
  await bridge.profile.getDefault()
  assert.deepEqual(electron.ipcRenderer.invoke.mock.lastCall, ['hermes:profile:default:get'])
  bridge.setF12ShortcutActive(true)
  assert.equal(electron.ipcRenderer.send.mock.lastCall?.[0], 'hermes:f12ShortcutActive')
  assert.equal(electron.ipcRenderer.send.mock.lastCall?.[1], true)

  const shortcutInputs: unknown[] = []
  const stopShortcut = bridge.onF12Shortcut((input: unknown) => shortcutInputs.push(input))
  const [shortcutChannel, shortcutListener] = electron.ipcRenderer.on.mock.lastCall!
  assert.equal(shortcutChannel, 'hermes:f12-shortcut')
  shortcutListener({}, { key: 'F12', repeat: true })
  assert.deepEqual(shortcutInputs, [{ key: 'F12', repeat: true }])
  stopShortcut()
  assert.deepEqual(electron.ipcRenderer.removeListener.mock.lastCall, [shortcutChannel, shortcutListener])

  const changes: unknown[] = []
  const unsubscribe = bridge.profile.onDefaultChanged((value: unknown) => changes.push(value))
  const [channel, listener] = electron.ipcRenderer.on.mock.lastCall!
  assert.equal(channel, 'hermes:profile:default:changed')
  listener({}, route)
  listener({}, null)
  assert.deepEqual(changes, [route, null])
  unsubscribe()
  assert.deepEqual(electron.ipcRenderer.removeListener.mock.lastCall, [channel, listener])
})
