import type { DestinationId, EntityKey, LunarCitySnapshot, LunarEntity } from '../model'

export interface EntityListProps {
  cameraOrder?: readonly EntityKey[]
  snapshot: LunarCitySnapshot
  selectedEntityKey?: EntityKey
  onSelect(entity: LunarEntity): void
  districtOrder?: readonly string[]
}

const MANIFEST_DISTRICT_ORDER: readonly DestinationId[] = [
  'bus',
  'project',
  'depot',
  'review',
  'triage',
  'garden',
  'council',
  'unavailable',
  'unknown'
]

const DESTINATION_LABELS: Readonly<Record<string, string>> = {
  bus: 'Transit Hub',
  council: 'Council District',
  depot: 'Resource Depot',
  garden: 'Garden District',
  lab: 'Research Lab',
  library: 'Library District',
  project: 'Project District',
  review: 'Review Office',
  triage: 'Triage District',
  unavailable: 'Unavailable',
  unknown: 'Unknown destination'
}

const ANIMATION_LABELS: Readonly<Record<string, string>> = {
  blocked: 'Blocked',
  done: 'Completed',
  failed: 'Failed',
  handoff: 'Handing off',
  heartbeat: 'Heartbeat',
  queue: 'Queued',
  rest: 'Idle',
  triage: 'In triage',
  unavailable: 'Unavailable',
  wait: 'Waiting',
  work: 'Working'
}

const AUTHORITY_LABELS: Readonly<Record<LunarEntity['authority'], string>> = {
  authoritative: 'Authoritative',
  partial: 'Partial',
  stale: 'Stale',
  unknown: 'Unknown authority'
}

function titleCase(value: string): string {
  return value.replace(/[_-]+/gu, ' ').replace(/\b\w/gu, character => character.toUpperCase())
}

export function entityFriendlyLabel(entity: LunarEntity): string {
  const identity = entity.identity

  if (identity.kind === 'profile') {
    return entity.presentation?.configuredTitle ?? `Profile ${titleCase(identity.profile)}`
  }

  if (identity.kind === 'session') {
    return `Session ${titleCase(identity.sessionId)}`
  }

  if (identity.kind === 'subagent') {
    return `Subagent ${titleCase(identity.subagentId)}`
  }

  if (identity.workerId) {
    return `Worker ${titleCase(identity.workerId)}`
  }

  return `Task ${titleCase(identity.taskId)}`
}

function presentationDetails(entity: LunarEntity): string[] {
  const presentation = entity.presentation

  if (!presentation) {
    return []
  }

  return [
    ...(presentation.profileHandle ? [`Handle: ${presentation.profileHandle}`] : []),
    ...(presentation.sourceLabel ? [`Source: ${presentation.sourceLabel}`] : []),
    `Groups: ${presentation.groups.length > 0 ? presentation.groups.map(group => group.name).join(', ') : 'None'}`,
    `Profile metadata: ${titleCase(presentation.metadata.state)}`,
    `Metadata source: ${presentation.metadata.source}`,
    ...(presentation.placement.overflow ? ['District detail: aggregate LOD'] : [])
  ]
}

/**
 * Formats ownership fields without title-casing, trimming, or otherwise
 * changing external identifiers. These fields are the assistive-technology
 * identity for rows whose human labels may legitimately collide.
 */
function identityDetails(entity: LunarEntity): string {
  const identity = entity.identity
  const fields = [`kind ${identity.kind}`, `connectionId ${identity.connectionId}`, `profile ${identity.profile}`]

  if (identity.kind === 'profile') {
    return fields.join('; ')
  }

  if (identity.kind === 'session') {
    fields.push(`sessionId ${identity.sessionId}`)
  } else if (identity.kind === 'subagent') {
    fields.push(
      `sessionId ${identity.sessionId}`,
      `parentSessionId ${identity.sessionId}`,
      `subagentId ${identity.subagentId}`
    )
  } else {
    fields.push(`board ${identity.board}`, `taskId ${identity.taskId}`)

    if (identity.runId !== undefined) {
      fields.push(`runId ${identity.runId}`)
    }

    if (identity.workerId !== undefined) {
      fields.push(`workerId ${identity.workerId}`)
    }
  }

  return fields.join('; ')
}

function destinationLabel(destination: string): string {
  return DESTINATION_LABELS[destination] ?? titleCase(destination)
}

function stateLabel(animation: string): string {
  return ANIMATION_LABELS[animation] ?? titleCase(animation)
}

function districtRank(destination: string, districtOrder: readonly string[]): number {
  const index = districtOrder.indexOf(destination)

  return index === -1 ? districtOrder.length : index
}

/**
 * Returns a presentation-only ordered copy. The snapshot itself remains
 * immutable and exact entity keys are retained for selection.
 */
export function orderedEntities(
  snapshot: LunarCitySnapshot,
  districtOrder: readonly string[] = MANIFEST_DISTRICT_ORDER,
  cameraOrder: readonly EntityKey[] = []
): readonly LunarEntity[] {
  const cameraRanks = new Map(cameraOrder.map((key, index) => [key, index]))

  return [...snapshot.entities.values()].sort((left, right) => {
    const districtComparison =
      districtRank(left.destination, districtOrder) - districtRank(right.destination, districtOrder)

    const cameraComparison =
      (cameraRanks.get(left.key) ?? Number.MAX_SAFE_INTEGER) - (cameraRanks.get(right.key) ?? Number.MAX_SAFE_INTEGER)

    return districtComparison || cameraComparison || left.key.localeCompare(right.key)
  })
}

export function EntityList({ cameraOrder, districtOrder, onSelect, selectedEntityKey, snapshot }: EntityListProps) {
  const entities = orderedEntities(snapshot, districtOrder, cameraOrder)

  return (
    <section aria-label="Lunar City entities" className="lunar-city-entity-list">
      <h2>Entities</h2>
      {entities.length === 0 ? (
        <p role="status">No entities observed.</p>
      ) : (
        <ul>
          {entities.map(entity => {
            const name = entityFriendlyLabel(entity)
            const state = stateLabel(entity.animation)
            const destination = destinationLabel(entity.destination)
            const authority = AUTHORITY_LABELS[entity.authority]
            const descriptionId = `lunar-city-entity-${entity.key.replace(/[^a-z0-9_-]/giu, '-')}`
            const description = `${state}; ${destination}; ${authority}`
            const identity = identityDetails(entity)
            const presentation = presentationDetails(entity)
            const accessibleDescription = `${description}; ${presentation.length > 0 ? `${presentation.join('; ')}; ` : ''}Identity: ${identity}; Entity key ${entity.key}`

            return (
              <li key={entity.key}>
                <button
                  aria-describedby={descriptionId}
                  aria-label={`${name}, ${accessibleDescription}`}
                  aria-pressed={entity.key === selectedEntityKey}
                  onClick={() => onSelect(entity)}
                  type="button"
                >
                  {name}
                </button>
                <span className="lunar-city-entity-details" id={descriptionId}>
                  {accessibleDescription}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
