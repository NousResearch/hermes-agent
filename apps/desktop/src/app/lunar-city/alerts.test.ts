import { describe, expect, it } from 'vitest'

import { acknowledgeAlert, EMPTY_ALERT_LEDGER, reconcileAlerts } from './alerts'
import { entityKey } from './identity'
import type { LunarCitySnapshot, LunarEntity, SourceHealth } from './model'

function entity(connectionId: string, state: string, observedAt: number): LunarEntity {
  const identity = { connectionId, kind: 'profile' as const, profile: 'same-worker' }

  return {
    identity,
    key: entityKey(identity),
    authority: 'authoritative',
    animation: state,
    sourceState: state,
    destination: 'triage',
    observedAt
  }
}

function snapshot(entities: LunarEntity[], sources: SourceHealth[] = []): LunarCitySnapshot {
  return { revision: 1, observedAt: 1, entities: new Map(entities.map(row => [row.key, row])), sources }
}

describe('local attention lifecycle', () => {
  it('isolates owners, deduplicates, acknowledges locally and requires newer observed recovery before reopening', () => {
    const a = entity('a', 'blocked', 10),
      b = entity('b', 'blocked', 10)

    const first = snapshot([a, b])
    let ledger = reconcileAlerts(EMPTY_ALERT_LEDGER, first)
    expect(ledger.incidents).toHaveLength(2)
    expect(reconcileAlerts(ledger, first)).toBe(ledger)
    const id = ledger.incidents.find(row => row.entityKey === a.key)!.id
    ledger = acknowledgeAlert(ledger, id)
    expect(a.sourceState).toBe('blocked')
    expect(ledger.incidents.filter(row => row.status === 'acknowledged')).toHaveLength(1)
    ledger = reconcileAlerts(ledger, snapshot([{ ...a, authority: 'stale', observedAt: 20 }]))
    expect(ledger.incidents.every(row => row.status !== 'resolved')).toBe(true)
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'heartbeat', 25)]))
    expect(ledger.incidents.find(row => row.id === id)?.status).toBe('acknowledged')
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'mystery-state', 30)]))
    expect(ledger.incidents.find(row => row.id === id)?.status).toBe('acknowledged')
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'working', 40)]))
    expect(ledger.incidents.find(row => row.id === id)?.status).toBe('resolved')
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'working', 50)]))
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'blocked', 45)]))
    expect(ledger.incidents.find(row => row.id === id)?.status).toBe('resolved')
    ledger = reconcileAlerts(ledger, snapshot([entity('a', 'blocked', 60)]))
    expect(ledger.incidents.find(row => row.id === id)).toMatchObject({ status: 'reopened', occurrences: 2 })
    expect(ledger.incidents.find(row => row.entityKey === b.key)?.status).toBe('open')
  })
  it('retains source failures across disappearance and bounds active and resolved populations', () => {
    const unhealthy: SourceHealth = { source: 'sessions:remote:profile', authority: 'partial', observedAt: 10 }
    let ledger = reconcileAlerts(EMPTY_ALERT_LEDGER, snapshot([], [unhealthy]))
    ledger = acknowledgeAlert(ledger, ledger.incidents[0]!.id)
    expect(reconcileAlerts(ledger, snapshot([]))).toBe(ledger)
    ledger = reconcileAlerts(ledger, snapshot([], [{ ...unhealthy, authority: 'authoritative' }]))
    expect(ledger.incidents[0]?.status).toBe('acknowledged')
    ledger = reconcileAlerts(ledger, snapshot([], [{ ...unhealthy, authority: 'authoritative', observedAt: 11 }]))
    expect(ledger.incidents[0]?.status).toBe('resolved')
    const rows = Array.from({ length: 200 }, (_, index) => entity(`remote-${index}`, 'failed', 20))
    ledger = reconcileAlerts(ledger, snapshot(rows))
    expect(ledger.omitted).toBeGreaterThan(0)
    expect(ledger.incidents.filter(row => row.status !== 'resolved').length).toBeLessThan(rows.length)
    ledger = reconcileAlerts(ledger, snapshot(rows.map(row => ({ ...row, sourceState: 'completed', observedAt: 30 }))))
    expect(ledger.incidents.every(row => row.status === 'resolved')).toBe(true)
    expect(ledger.incidents.length).toBeLessThan(100)
  })
})
