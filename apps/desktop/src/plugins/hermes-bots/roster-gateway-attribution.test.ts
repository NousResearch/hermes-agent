/**
 * Gateway sections: order and labelling (§6.2 / §6.1).
 *
 * The roster buckets rows per connection and draws one heading per bucket. The
 * primary connection must lead (D7) and every heading must read the gateway's
 * cloud `name` when the source carries one, so a `hermes cloud` gateway is
 * attributed by its own name rather than by a positional accident.
 */

import { describe, expect, it } from 'vitest'

import { rosterGatewayOptions, rosterGatewaySections } from './roster-sections'
import type { GatewaySource, RosterRow } from './types'

const sources: GatewaySource[] = [
  { connectionId: 'local', kind: 'local', label: 'This device', name: 'This device', reachable: true },
  { connectionId: 'alpha', kind: 'remote', label: 'Alpha', reachable: true },
  {
    cloud: { bot: 'cloud-123', gateway_id: 'gw-1' },
    connectionId: 'cloud-123',
    kind: 'remote',
    label: 'cloud-123',
    name: 'cloud-123',
    reachable: true, url: 'http://127.0.0.1:18764'
  }
]

function row(connectionId: string, extra: Partial<RosterRow> = {}): RosterRow {
  return {
    connectionId,
    connectionKind: connectionId === 'local' ? 'local' : 'remote',
    connectionLabel: connectionId,
    name: 'default',
    ...extra
  }
}

describe('rosterGatewayOptions ordering (D7)', () => {
  it('puts the primary connection first, then the rest alphabetically', () => {
    expect(rosterGatewayOptions(sources, [], 'local').map(option => option.connectionId)).toEqual([
      'local',
      'alpha',
      'cloud-123'
    ])
  })

  it('still leads with the primary when the primary is a cloud gateway', () => {
    expect(rosterGatewayOptions(sources, [], 'cloud-123').map(option => option.connectionId)).toEqual([
      'cloud-123',
      'alpha',
      'local'
    ])
  })

  it('keeps alphabetical order when no primary is known (older shells)', () => {
    expect(rosterGatewayOptions(sources, []).map(option => option.connectionId)).toEqual([
      'alpha',
      'cloud-123',
      'local'
    ])
  })

  it('falls back to the row label when the source list is empty', () => {
    const options = rosterGatewayOptions([], [row('cloud-123', { connectionLabel: 'cloud-123', gatewayName: 'cloud-123' })], 'cloud-123')

    expect(options).toHaveLength(1)
    expect(options[0].name).toBe('cloud-123')
    expect(options[0].label).toBe('cloud-123')
  })
})

describe('rosterGatewaySections', () => {
  it('sections in option order and keeps the cloud name on the gateway option', () => {
    const rows = [row('cloud-123', { gatewayName: 'cloud-123' }), row('local')]
    const { sectioned, sections } = rosterGatewaySections(rows, rosterGatewayOptions(sources, rows, 'local'))

    expect(sectioned).toBe(true)
    expect(sections.map(section => section.id)).toEqual(['local', 'cloud-123'])
    expect(sections[1].option?.name).toBe('cloud-123')
    expect(sections[1].option?.cloud?.gateway_id).toBe('gw-1')
    expect(sections[1].rows[0].gatewayName).toBe('cloud-123')
  })

  it('never sections a single-source roster', () => {
    const rows = [row('local')]

    const { sectioned } = rosterGatewaySections(
      rows,
      rosterGatewayOptions([sources[0]], rows, 'local')
    )

    expect(sectioned).toBe(false)
  })
})
