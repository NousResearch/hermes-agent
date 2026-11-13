import { expect, it } from 'vitest'

import { configForLocalPrimary, switchPrimary } from './connection-primary-route'
import { normalizeRegistry, setPrimaryConnection } from './connection-registry'

// Boot resolves connection.json's app-global route before the registry primary, so a
// promotion back to This device must retire that route in the same write. Without it the
// launch keeps dialing the gateway it was promoted away from, and every profile that only
// exists on that gateway is refused on the forced-local branch.

const registry = (primary = 'gw-b') =>
  normalizeRegistry({
    version: 2,
    primary,
    launchMode: 'last-used',
    lastUsed: primary,
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'gw-b', kind: 'remote', label: 'Gateway B', url: 'http://192.0.2.10:9119', authMode: 'token' }
    ]
  })

const globalRemote = {
  mode: 'remote',
  remote: { url: 'http://192.0.2.10:9119', authMode: 'token', token: 'ciphertext' },
  profiles: { research: { mode: 'remote', url: 'https://research.example.invalid', authMode: 'oauth' } }
}

it('promoting This device resets the v1 global route that still outranks it at boot', () => {
  const cleared = configForLocalPrimary(setPrimaryConnection(registry(), 'local'), globalRemote)

  expect(cleared?.mode).toBe('local')
  expect(cleared?.remote).toEqual({})
  // A per-profile override is not an app-global route, so it must survive the reset.
  expect(cleared?.profiles).toEqual(globalRemote.profiles)
})

it('promoting a remote gateway leaves connection.json alone', () => {
  const promoted = setPrimaryConnection(setPrimaryConnection(registry(), 'local'), 'gw-b')

  expect(configForLocalPrimary(promoted, globalRemote)).toBeNull()
})

it('an SSH app-global route is a route too', () => {
  const ssh = { mode: 'ssh', remote: { host: 'build.example.invalid', user: 'hermes' }, profiles: {} }

  expect(configForLocalPrimary(setPrimaryConnection(registry(), 'local'), ssh)?.mode).toBe('local')
})

it('a cloud app-global route is a route too', () => {
  const cloud = { mode: 'cloud', remote: { url: 'https://cloud.example.invalid', org: 'acme' }, profiles: {} }

  expect(configForLocalPrimary(setPrimaryConnection(registry(), 'local'), cloud)?.remote).toEqual({})
})

it('a local primary with no app-global route needs no rewrite', () => {
  const alreadyLocal = { mode: 'local', remote: {}, profiles: { research: { mode: 'local' } } }

  expect(configForLocalPrimary(setPrimaryConnection(registry(), 'local'), alreadyLocal)).toBeNull()
})

// A promotion runs two writes. Either one failing must leave both files as they were, or the
// registry says "local" while boot still reads the old remote from connection.json — the exact
// split this module exists to remove.

function fakeIo({ failConfig = false, failRegistry = false } = {}) {
  const state = {
    config: JSON.parse(JSON.stringify(globalRemote)) as Record<string, any>,
    registry: registry()
  }

  const writes: string[] = []
  const copy = (value: unknown) => JSON.parse(JSON.stringify(value))

  return {
    state,
    writes,
    io: {
      readConfig: () => copy(state.config),
      writeConfig: (next: Record<string, any>) => {
        writes.push('config')

        if (failConfig) {throw new Error('connection.json write failed')}
        state.config = copy(next)
      },
      readRegistry: () => copy(state.registry),
      writeRegistry: (next: unknown) => {
        writes.push('registry')

        if (failRegistry) {throw new Error('connections.json write failed')}
        state.registry = copy(next)
      },
      log: () => undefined
    }
  }
}

it('a promotion to This device commits connection.json before the saved list', async () => {
  const { state, writes, io } = fakeIo()

  const registryAfter = await switchPrimary('local', io)

  expect(registryAfter.primary).toBe('local')
  expect(state.config.mode).toBe('local')
  expect(state.config.remote).toEqual({})
  expect(writes).toEqual(['config', 'registry'])
})

it('a failed connection.json write leaves the saved list on the old primary', async () => {
  const { state, writes, io } = fakeIo({ failConfig: true })

  await expect(switchPrimary('local', io)).rejects.toThrow('connection.json write failed')

  expect(state.registry.primary).toBe('gw-b')
  // The failed write, then the rollback write. The saved list is never touched.
  expect(writes).toEqual(['config', 'config'])
})

it('a failed registry write restores the previous connection.json', async () => {
  const { state, io } = fakeIo({ failRegistry: true })

  await expect(switchPrimary('local', io)).rejects.toThrow('connections.json write failed')

  expect(state.config).toEqual(globalRemote)
  expect(state.registry.primary).toBe('gw-b')
})

it('a promotion to a remote writes only the saved list', async () => {
  const { state, writes, io } = fakeIo()

  state.registry = setPrimaryConnection(state.registry, 'local')
  state.config = { mode: 'local', remote: {}, profiles: {} }

  await switchPrimary('gw-b', io)

  expect(state.registry.primary).toBe('gw-b')
  expect(state.config).toEqual({ mode: 'local', remote: {}, profiles: {} })
  expect(writes).toEqual(['registry'])
})
