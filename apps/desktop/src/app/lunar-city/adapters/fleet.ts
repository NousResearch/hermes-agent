import type { DesktopAgentRoster } from '@/global'

import { entityKey } from '../identity'
import type { LunarEntity, SourceHealth } from '../model'

export interface NormalizedFleet {
  entities: readonly LunarEntity[]
  sources: readonly SourceHealth[]
}

export interface FleetObservation {
  fresh?: boolean
  observedAt: number
  sourceObservedAt?: ReadonlyMap<string, number>
}

function sourceHealth(source: DesktopAgentRoster['sources'][number], observedAt: number): SourceHealth {
  return {
    authority: source.reachable && !source.error ? 'authoritative' : 'stale',
    ...(source.error ? { error: source.error } : {}),
    observedAt,
    source: `fleet:${source.connectionId}`
  }
}

/**
 * Converts Electron's fleet roster into presentation entities without using
 * the active connection or a profile display name as an ownership fallback.
 */
export function normalizeRoster(roster: DesktopAgentRoster, observation: FleetObservation): NormalizedFleet {
  const fresh = observation.fresh !== false
  const healthByConnection = new Map(roster.sources.map(source => [source.connectionId, source]))

  const observedAtFor = (connectionId: string) =>
    observation.sourceObservedAt?.get(connectionId) ?? observation.observedAt

  const sources = [...roster.sources]
    .sort((left, right) => left.connectionId.localeCompare(right.connectionId))
    .map(source => ({
      ...sourceHealth(source, observedAtFor(source.connectionId)),
      ...(fresh ? {} : { authority: 'stale' as const })
    }))

  const entities = [...roster.agents]
    .sort(
      (left, right) => left.connectionId.localeCompare(right.connectionId) || left.profile.localeCompare(right.profile)
    )
    .map(agent => {
      const source = healthByConnection.get(agent.connectionId)
      const available = fresh && source?.reachable === true && !source.error
      const identity = { kind: 'profile' as const, connectionId: agent.connectionId, profile: agent.profile }

      return {
        animation: available ? 'rest' : 'unavailable',
        authority: fresh ? (available ? 'authoritative' : source ? 'stale' : 'unknown') : 'stale',
        destination: available ? 'garden' : source ? 'unavailable' : 'unknown',
        identity,
        key: entityKey(identity),
        observedAt: observedAtFor(agent.connectionId)
      } satisfies LunarEntity
    })

  return { entities, sources }
}
