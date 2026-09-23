import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDesktopConnectionFleetIpc } from './desktop-connection-fleet-runtime'

test('fleet IPC registers its roster and update handlers only when composed', () => {
  const registered: string[] = []
  const ipcMain = { handle: (channel: string) => registered.push(channel) }

  const runtime = registerDesktopConnectionFleetIpc({
    ipcMain,
    createRegistryGatewayWsUrlHandler: () => async () => undefined
  })

  assert.deepEqual(registered, [
    'hermes:agents:roster',
    'hermes:gateway:ws-url-for',
    'hermes:connections:update-managed',
    'hermes:connections:update-all'
  ])
  assert.equal(typeof runtime.rememberConnectionInstallId, 'function')
  assert.equal(typeof runtime.probeSshProfileInventory, 'function')
  assert.equal(typeof runtime.enumerateRegistryAgentSources, 'function')
})
