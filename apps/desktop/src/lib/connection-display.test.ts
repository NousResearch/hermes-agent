import { describe, expect, it } from 'vitest'

import type { DesktopRegistryConnection } from '@/global'

import {
  connectionEndpoint,
  connectionMatchesQuery,
  connectionTooltip,
  sortConnectionsForDisplay
} from './connection-display'

const connection = (
  id: string,
  label: string,
  kind: DesktopRegistryConnection['kind'] = 'remote'
): DesktopRegistryConnection => ({ id, kind, label, tokenPreview: null, tokenSet: false })

describe('connection display helpers', () => {
  it('anchors local first and sorts labels case-insensitively with numeric order', () => {
    const sorted = sortConnectionsForDisplay([
      connection('remote-10', 'Studio 10'),
      connection('zulu', 'zulu'),
      connection('remote-2', 'studio 2'),
      connection('local', 'This device', 'local'),
      connection('alpha', 'Alpha')
    ])

    expect(sorted.map(item => item.id)).toEqual(['local', 'alpha', 'remote-2', 'remote-10', 'zulu'])
  })

  it('does not mutate registry order', () => {
    const connections = [connection('zulu', 'Zulu'), connection('alpha', 'Alpha')]

    sortConnectionsForDisplay(connections)

    expect(connections.map(item => item.id)).toEqual(['zulu', 'alpha'])
  })

  it('matches labels, transport details, localized aliases, and accents', () => {
    const gateway: DesktopRegistryConnection = {
      ...connection('studio', 'Studio Genève', 'ssh'),
      host: 'studio.example.test',
      org: 'Editorial',
      port: 2222,
      remoteProfile: 'production',
      user: 'hermes'
    }

    expect(connectionMatchesQuery(gateway, 'GENEVE')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'studio 2222')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'ssh studio')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'ssh')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'example.test')).toBe(true)
    expect(connectionMatchesQuery(gateway, '2222')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'editorial production')).toBe(true)
    expect(connectionMatchesQuery(gateway, 'distant', ['Passerelle distante'])).toBe(true)
    expect(connectionMatchesQuery(gateway, 'homelab')).toBe(false)
  })

  it('keeps technical endpoints available on demand without exposing secrets', () => {
    const remote = {
      ...connection('work', 'Work gateway'),
      tokenPreview: 'sec...ret',
      tokenSet: true,
      url: 'https://work.example.test:9443'
    }

    const ssh = {
      ...connection('studio', 'Studio over SSH', 'ssh'),
      host: 'studio.example.test',
      keyPath: '/secret/key',
      port: 2222,
      user: 'hermes'
    }

    expect(connectionEndpoint(remote)).toBe('https://work.example.test:9443')
    expect(connectionTooltip(remote)).toBe('Work gateway\nhttps://work.example.test:9443')
    expect(connectionTooltip(remote)).not.toContain('sec...ret')
    expect(connectionEndpoint(ssh)).toBe('hermes@studio.example.test:2222')
    expect(connectionTooltip(ssh)).not.toContain('/secret/key')
    expect(connectionTooltip(connection('local', 'This device', 'local'))).toBe('This device')
  })
})
