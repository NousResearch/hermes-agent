import { describe, expect, it } from 'vitest'

import type { DesktopConnectionsRegistry, DesktopRegistryConnection, HermesApiRequest } from '@/global'
import { deferred } from '@/test/deferred'

import { createBrowserSavedConnections } from './browser-saved-connections'

const row = (id: string): DesktopRegistryConnection => ({
  authMode: 'token',
  headerNames: [],
  id,
  kind: 'remote',
  label: id,
  tokenPreview: null,
  tokenSet: true,
  url: `https://${id}.example`
})

const registryOf = (...ids: string[]): DesktopConnectionsRegistry => ({
  connections: ids.map(row),
  lastUsed: ids[0] ?? '',
  launchMode: 'primary',
  primary: ids[0] ?? '',
  secureTokenStorage: true,
  version: 2
})

/** The serving gateway's list route, whose GETs answer only when the test says so. */
function gatewayList() {
  const gets: ReturnType<typeof deferred<DesktopConnectionsRegistry>>[] = []
  let stored = registryOf('alpha')

  const api = async <T,>(request: HermesApiRequest): Promise<T> => {
    if ((request.method ?? 'GET') === 'GET') {
      const answer = deferred<DesktopConnectionsRegistry>()
      gets.push(answer)

      return answer.promise as Promise<T>
    }

    stored = registryOf('alpha', 'beta')

    return { connection: row('beta'), ok: true, registry: stored } as T
  }

  const connections = createBrowserSavedConnections({
    api,
    bootstrap: { authRequired: true, basePath: '', privateSession: false, token: '' }
  })

  return { connections, gets, stored: () => stored }
}

const ids = (registry: DesktopConnectionsRegistry) => registry.connections.map(connection => connection.id)

describe('reading the saved connections', () => {
  it('shares one read among callers that ask while it is under way', async () => {
    const { connections, gets } = gatewayList()
    const first = connections.connections.list()
    const second = connections.connections.list()

    expect(gets).toHaveLength(1)
    gets[0]!.resolve(registryOf('alpha'))
    expect(ids(await first)).toEqual(['alpha'])
    expect(ids(await second)).toEqual(['alpha'])
  })

  it('reads afresh after a write, and a read from before the write publishes nothing', async () => {
    const { connections, gets, stored } = gatewayList()
    const changes: string[] = []
    connections.connections.onChanged!(change => changes.push(`${change.connectionId}:${change.reason}`))

    // Seed a snapshot, then start a poll that answers with the list as it was before the save.
    const seed = connections.connections.list()
    gets[0]!.resolve(registryOf('alpha'))
    await seed
    const before = connections.connections.list()

    await connections.connections.save({ kind: 'remote', label: 'beta', url: 'https://beta.example' })
    const after = connections.connections.list()

    expect(gets).toHaveLength(3)
    gets[1]!.resolve(registryOf('alpha'))
    gets[2]!.resolve(stored())

    expect(ids(await before)).toEqual(['alpha'])
    expect(ids(await after)).toEqual(['alpha', 'beta'])
    // The save's own answer added Beta; the older poll's answer did not take it away again.
    expect(changes).toEqual(['beta:saved'])
    expect(connections.gateway('beta')).toBeTruthy()
  })
})
