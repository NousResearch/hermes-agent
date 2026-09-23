import assert from 'node:assert/strict'

import { test } from 'vitest'

import { BackendDialClaims } from './backend-dial-claim'
import { registerDesktopConnectionDialIpc } from './desktop-connection-dial-ipc'
import { WindowConnectionRouteRegistry } from './window-connection-route'

test('two windows requesting one registry scope share the in-flight dial', async () => {
  let release!: () => void
  let dials = 0

  const gate = new Promise<void>(resolve => {
    release = resolve
  })

  const runtime = registerDesktopConnectionDialIpc({
    ipcMain: { handle: () => {}, on: () => {} },
    windowConnectionRoutes: new WindowConnectionRouteRegistry(),
    backendDialClaims: new BackendDialClaims(),
    applySpawnPriority: () => () => {},
    ensureRegistryBackend: async () => {
      dials += 1
      await gate

      return { baseUrl: 'https://remote.example', mode: 'remote' }
    }
  } as any)

  const route = { connectionId: 'remote-a', profile: 'default' }

  const first = runtime.connectDesktopProfileRoute(route)
  const second = runtime.connectDesktopProfileRoute(route)
  release()

  const [a, b] = await Promise.all([first, second])
  assert.equal(dials, 1)
  assert.deepEqual(a, b)
  assert.equal(a.connectionId, 'remote-a')
})
