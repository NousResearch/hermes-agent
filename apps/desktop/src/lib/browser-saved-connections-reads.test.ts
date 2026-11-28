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
function gatewayList({ holdWrites = false } = {}) {
  const gets: ReturnType<typeof deferred<DesktopConnectionsRegistry>>[] = []
  // With `holdWrites`, a save answers only when the test calls its entry here.
  const writes: (() => void)[] = []
  let stored = registryOf('alpha')

  const api = async <T,>(request: HermesApiRequest): Promise<T> => {
    if ((request.method ?? 'GET') === 'GET') {
      const answer = deferred<DesktopConnectionsRegistry>()
      gets.push(answer)

      return answer.promise as Promise<T>
    }

    stored = registryOf('alpha', 'beta')
    const answer = { connection: row('beta'), ok: true, registry: stored }

    if (!holdWrites) {
      return answer as T
    }

    const held = deferred<T>()
    writes.push(() => held.resolve(answer as T))

    return held.promise
  }

  const connections = createBrowserSavedConnections({
    api,
    bootstrap: { authRequired: true, basePath: '', privateSession: false, token: '' }
  })

  return { connections, gets, stored: () => stored, writes }
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

    // The poll's caller gets the list the save answered with, not the older one it read.
    expect(ids(await before)).toEqual(['alpha', 'beta'])
    expect(ids(await after)).toEqual(['alpha', 'beta'])
    // The save's own answer added Beta; the older poll's answer did not take it away again.
    expect(changes).toEqual(['beta:saved'])
    expect(connections.gateway('beta')).toBeTruthy()
  })

  it('gives a consumer that publishes what it reads the newest list, whichever read lands last', async () => {
    const { connections, gets, stored } = gatewayList()
    const seen: string[][] = []
    const seed = connections.connections.list()
    gets[0]!.resolve(registryOf('alpha'))
    await seed

    // Settings starts a read, a save lands, a read after the save finishes first, the older last.
    // The newer read also sees Gamma, added in another tab: the save's own list would lack it.
    const older = connections.connections.list().then(registry => seen.push(ids(registry)))
    await connections.connections.save({ kind: 'remote', label: 'beta', url: 'https://beta.example' })
    const newer = connections.connections.list().then(registry => seen.push(ids(registry)))
    gets[2]!.resolve(registryOf(...ids(stored()), 'gamma'))
    await newer
    gets[1]!.resolve(registryOf('alpha'))
    await older

    expect(seen).toEqual([
      ['alpha', 'beta', 'gamma'],
      ['alpha', 'beta', 'gamma']
    ])
  })

  it('gives the save the last word when a read from before it answers in the same moment', async () => {
    const { connections, gets, writes } = gatewayList({ holdWrites: true })
    const seen: string[][] = []

    // Settings reads, then saves; the read (the list before the save) and the save answer together.
    const older = connections.connections.list().then(registry => seen.push(ids(registry)))

    const saved = connections.connections
      .save({ kind: 'remote', label: 'beta', url: 'https://beta.example' })
      .then(result => seen.push(ids(result.registry)))

    await Promise.resolve()
    gets[0]!.resolve(registryOf('alpha'))
    writes[0]!()
    await Promise.all([older, saved])

    expect(seen.at(-1)).toEqual(['alpha', 'beta'])
  })
})
