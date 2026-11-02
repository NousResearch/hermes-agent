import { describe, expect, it } from 'vitest'

import {
  connectionDialFieldsChanged,
  mergeConnectionInput,
  normalizeConnectionInput,
  normalizeRegistry
} from './connection-registry'
import { matchingConnectionId } from './connection-route-identity'
import { registryPrimaryBootRoute } from './desktop-remote-route'

const draft = (host = 'gw-host', url = 'http://127.0.0.1:9119') => ({
  kind: 'remote' as const,
  label: `${host} ${url}`,
  url,
  authMode: 'token',
  token: { encrypted: 'test-only' },
  sshTunnel: { host }
})

describe('remote gateway tunnel configuration', () => {
  it('persists, routes and recycles the tunnel; never persists a local port', () => {
    const registry = normalizeRegistry({})
    const entry = normalizeConnectionInput(draft(' gw-host '), registry)

    expect(entry).toMatchObject({ sshTunnel: { host: 'gw-host' }, url: 'http://127.0.0.1:9119' })

    const saved = normalizeRegistry(
      JSON.parse(JSON.stringify({ ...registry, primary: entry.id, connections: [...registry.connections, entry] }))
    )

    expect(saved.connections[1]).toEqual(entry)
    expect(registryPrimaryBootRoute(saved)).toMatchObject({ sshTunnel: entry.sshTunnel, connectionId: entry.id })
    expect(mergeConnectionInput({ id: entry.id, kind: 'remote', label: 'Renamed' }, entry).sshTunnel).toEqual(
      entry.sshTunnel
    )
    expect(connectionDialFieldsChanged(entry, { ...entry, sshTunnel: { host: 'other' } })).toBe(true)
    expect(connectionDialFieldsChanged(entry, { ...entry, sshTunnel: { ...entry.sshTunnel! } })).toBe(false)
  })

  it('deduplicates by SSH target plus URL, so equal loopback URLs on other hosts coexist', () => {
    const registry = normalizeRegistry({})
    const first = normalizeConnectionInput(draft('alice@box:2222'), registry)
    registry.connections.push(first)
    const second = normalizeConnectionInput(draft('other'), registry)
    const custom = normalizeConnectionInput(draft('alice@box:2222', 'http://127.0.0.1:9220'), registry)
    registry.connections.push(second, custom)

    expect(() => normalizeConnectionInput({ ...draft('alice@box:2222'), label: 'Duplicate' }, registry)).toThrow(
      /already exists/
    )
    expect(matchingConnectionId(registry, { ...second, kind: 'remote' }, 'unique')).toBe(second.id)
    expect(matchingConnectionId(registry, { ...first, kind: 'remote' }, 'unique')).toBe(first.id)
  })

  it('accepts password/OAuth sign-in and rejects non-loopback or HTTPS URLs and unsafe SSH targets', () => {
    const fresh = () => normalizeRegistry({})

    expect(normalizeConnectionInput({ ...draft(), authMode: 'oauth' }, fresh())).toMatchObject({
      authMode: 'oauth',
      sshTunnel: { host: 'gw-host' }
    })

    expect(() => normalizeConnectionInput(draft('gw-host', 'https://127.0.0.1:9119'), fresh())).toThrow(/http:\/\//)
    expect(() => normalizeConnectionInput(draft('gw-host', 'http://gateway.lan:9119'), fresh())).toThrow(/loopback/)
    expect(() => normalizeConnectionInput(draft('-oProxyCommand=bad'), fresh())).toThrow()
    expect(() => normalizeConnectionInput({ ...draft(), kind: 'cloud', authMode: 'oauth' }, fresh())).toThrow(/Remote/)
  })
})
