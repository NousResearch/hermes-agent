import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

const skin = {
  profile: 'default',
  skin: { name: 'neon', colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' } }
}

const electron = vi.hoisted(() => ({
  contextBridge: { exposeInMainWorld: vi.fn() },
  ipcRenderer: {
    invoke: vi.fn(async () => ({ ok: true })),
    on: vi.fn(),
    removeListener: vi.fn(),
    sendSync: vi.fn((channel: string) => (channel === 'hermes:skin:local' ? skin : {}))
  },
  webFrame: {},
  webUtils: {}
}))

vi.mock('electron', () => electron)

test('the preload carries the local skin across before the renderer starts', async () => {
  await import('./preload')
  const [, bridge] = electron.contextBridge.exposeInMainWorld.mock.calls[0]

  assert.deepEqual(bridge.localSkin, skin)
  assert.equal(
    electron.ipcRenderer.sendSync.mock.calls.some(([channel]) => channel === 'hermes:skin:local'),
    true
  )
})
