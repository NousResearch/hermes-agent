import type { EntityKey, LunarCitySnapshot } from './model'

export type AlertKind = 'failed' | 'blocked' | 'review' | 'resource-wait' | 'source-health'
export type AlertStatus = 'open' | 'acknowledged' | 'resolved' | 'reopened'
export interface CityAlert {
  id: string
  kind: AlertKind
  status: AlertStatus
  owner: string
  entityKey?: EntityKey
  label: string
  detail: string
  firstObservedAt: number
  lastObservedAt: number
  occurrences: number
}
export interface AlertLedger {
  incidents: readonly CityAlert[]
  /** Number omitted by the active capacity limit in the latest snapshot. */
  omitted: number
}
export const EMPTY_ALERT_LEDGER: AlertLedger = { incidents: [], omitted: 0 }
const PRIORITY: Record<AlertKind, number> = { failed: 0, blocked: 1, 'source-health': 2, review: 3, 'resource-wait': 4 }
const ACTIVE_LIMIT = 128
const HISTORY_LIMIT = 32

const STATES: Readonly<Record<string, AlertKind>> = {
  failed: 'failed',
  error: 'failed',
  blocked: 'blocked',
  triage: 'blocked',
  review: 'review',
  under_review: 'review',
  awaiting_approval: 'review',
  resource_wait: 'resource-wait',
  waiting_for_resource: 'resource-wait'
}

const HEALTHY = new Set([
  'running',
  'working',
  'work',
  'done',
  'completed',
  'idle',
  'rest',
  'resting',
  'pause',
  'paused',
  'ready',
  'queued',
  'queue',
  'handoff',
  'dependency',
  'orchestration'
])

interface Observation {
  owner: string
  entityKey?: EntityKey
  kind?: AlertKind
  healthy: boolean
  label: string
  detail: string
  observedAt: number
}

function observations(snapshot: LunarCitySnapshot): Observation[] {
  const rows: Observation[] = []

  for (const entity of snapshot.entities.values()) {
    if (entity.authority !== 'authoritative') {
      continue
    }

    const state = (entity.sourceState ?? entity.animation)
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/gu, '_')

    rows.push({
      owner: JSON.stringify(['entity', entity.key]),
      entityKey: entity.key,
      kind: STATES[state],
      healthy: HEALTHY.has(state),
      label: `${entity.identity.profile} · ${entity.identity.connectionId}`,
      detail: state.replaceAll('_', ' '),
      observedAt: entity.observedAt
    })
  }

  for (const source of snapshot.sources) {
    const healthy = source.authority === 'authoritative' && !source.error
    rows.push({
      owner: JSON.stringify(['source', source.source]),
      kind: healthy ? undefined : 'source-health',
      healthy,
      label: source.source,
      detail: source.error || source.authority,
      observedAt: source.observedAt
    })
  }

  return rows.filter(row => Number.isFinite(row.observedAt) && row.observedAt >= 0)
}

export function compareAlerts(left: CityAlert, right: CityAlert): number {
  return (
    PRIORITY[left.kind] - PRIORITY[right.kind] ||
    right.lastObservedAt - left.lastObservedAt ||
    left.id.localeCompare(right.id)
  )
}

/** Presentation history only. Absence, stale observations and unknown states never prove recovery. */
export function reconcileAlerts(previous: AlertLedger, snapshot: LunarCitySnapshot): AlertLedger {
  const incidents = new Map(previous.incidents.map(incident => [incident.id, incident]))

  for (const row of observations(snapshot)) {
    const owned = [...incidents.values()].filter(incident => incident.owner === row.owner)

    if (owned.some(incident => incident.lastObservedAt > row.observedAt)) {
      continue
    }

    if (row.healthy) {
      for (const incident of owned) {
        if (row.observedAt > incident.lastObservedAt) {
          incidents.set(incident.id, { ...incident, status: 'resolved', lastObservedAt: row.observedAt })
        }
      }

      continue
    }

    if (!row.kind) {
      continue
    }

    const id = JSON.stringify([row.owner, row.kind])
    const current = incidents.get(id)

    // A resolved incident can reopen only on a strictly newer adverse observation.
    if (
      current &&
      (row.observedAt < current.lastObservedAt ||
        (current.status === 'resolved' && row.observedAt <= current.lastObservedAt))
    ) {
      continue
    }

    if (!current) {
      incidents.set(id, {
        id,
        kind: row.kind,
        owner: row.owner,
        entityKey: row.entityKey,
        label: row.label,
        detail: row.detail,
        firstObservedAt: row.observedAt,
        lastObservedAt: row.observedAt,
        status: 'open',
        occurrences: 1
      })
    } else if (
      current.status === 'resolved' ||
      row.observedAt > current.lastObservedAt ||
      row.detail !== current.detail
    ) {
      incidents.set(id, {
        ...current,
        label: row.label,
        detail: row.detail,
        lastObservedAt: row.observedAt,
        status: current.status === 'resolved' ? 'reopened' : current.status,
        occurrences: current.occurrences + (current.status === 'resolved' ? 1 : 0)
      })
    }
  }

  const active = [...incidents.values()].filter(row => row.status !== 'resolved').sort(compareAlerts)

  const history = [...incidents.values()]
    .filter(row => row.status === 'resolved')
    .sort((a, b) => b.lastObservedAt - a.lastObservedAt || a.id.localeCompare(b.id))
    .slice(0, HISTORY_LIMIT)

  const next = [...active.slice(0, ACTIVE_LIMIT), ...history]
  const omitted = Math.max(0, active.length - ACTIVE_LIMIT)

  return omitted === previous.omitted &&
    next.length === previous.incidents.length &&
    next.every((row, index) => row === previous.incidents[index])
    ? previous
    : { incidents: next, omitted }
}

/** Local acknowledgement never changes the observed backend status. */
export function acknowledgeAlert(ledger: AlertLedger, id: string): AlertLedger {
  const incident = ledger.incidents.find(row => row.id === id)

  if (!incident || incident.status === 'resolved' || incident.status === 'acknowledged') {
    return ledger
  }

  return {
    ...ledger,
    incidents: ledger.incidents.map(row => (row.id === id ? { ...row, status: 'acknowledged' } : row))
  }
}
