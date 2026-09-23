import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDesktopPluginProfileRoutesIpc } from './desktop-plugin-profile-routes-ipc'

test('plugin routes register in order and retain SSH seeds plus failed-local fallback without credentials', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const order: string[] = []

  const registry = {
    primary: 'local',
    connections: [
      { id: 'local', kind: 'local', label: 'This computer' },
      { id: 'ssh', kind: 'ssh', label: 'Lab', host: 'secret-host', remoteProfile: 'remote-root' }
    ]
  }

  const enumerations = [{ connection: registry.connections[0], error: 'offline' }]

  registerDesktopPluginProfileRoutesIpc({
    ipcMain: {
      handle: (channel: string, callback: (...args: any[]) => Promise<any>) => {
        order.push(channel)
        handlers.set(channel, callback)
      }
    },
    readDesktopConnectionConfig: () => ({ mode: 'local' }),
    sanitizeDesktopConnectionConfig: (_config: unknown, profile: string) => ({ profile }),
    readDesktopConnectionsRegistry: () => registry,
    enumerateRegistryAgentSources: async () => enumerations,
    buildAgentRoster: () => []
  } as any)

  assert.deepEqual(order, ['hermes:connection-config:get', 'hermes:plugin-profile-routes'])
  assert.deepEqual(await handlers.get('hermes:connection-config:get')!(null, 'named'), { profile: 'named' })

  const routes = await handlers.get('hermes:plugin-profile-routes')!(null, [' fallback ', null, 7])

  assert.deepEqual(routes, [
    { connectionId: 'ssh', mode: 'remote', profile: 'default', targetProfile: 'remote-root' },
    { connectionId: 'local', mode: 'local', profile: 'fallback', targetProfile: 'fallback' }
  ])
  assert.equal(JSON.stringify(routes).includes('secret-host'), false)
})
