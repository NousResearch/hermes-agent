import type { DesktopAgentRoster } from '@/global'

import type {
  DestinationId,
  EntityKey,
  LunarEntity,
  LunarEntityPresentation,
  LunarGroupMembership,
  LunarPresentationMetadata,
  Vec3
} from '../model'

export type ScopedProfilesListRequest = (connectionId: string, representativeProfile: string) => Promise<unknown>
export type BotMetadataProvenance = ReadonlyMap<string, LunarPresentationMetadata>

export interface BotRosterPlacementOptions {
  /** Power-of-two test seam; production always uses the declared 512-square lattice. */
  latticeSide?: number
  /** Route-local assignments preserve exact-key placement across incremental roster changes. */
  placementState?: BotRosterPlacementState
}

export interface BotRosterPlacementState {
  assignments: Map<DestinationId, Map<EntityKey, number | undefined>>
  capacity?: number
}

export function createBotRosterPlacementState(): BotRosterPlacementState {
  return { assignments: new Map() }
}

interface RichProfileRow {
  displayName?: string
  groups: readonly LunarGroupMembership[]
  name: string
  title?: string
}

interface ProjectedMembership {
  connectionId: string
  group: LunarGroupMembership
  profile: string
}

interface SourceDetails {
  projected: readonly ProjectedMembership[]
  rows: ReadonlyMap<string, RichProfileRow>
}

interface StagedEntity {
  destination: DestinationId
  entity: LunarEntity
  presentation?: Omit<LunarEntityPresentation, 'placement'>
  primary?: ReturnType<typeof primaryPlacement>
}

const MAX_SOURCES = 64
const MAX_PROFILE_ROWS = 2048
const MAX_GROUPS_PER_PROFILE = 64
const MAX_PROJECTED_ROOMS = 128
const MAX_ROOM_MEMBERS = 512
const MAX_PRESENTATION_CHARS = 160
const NEAR_WORKER_BUDGET = 24
const DISTRICT_LATTICE_SIDE = 512
const DISTRICT_LATTICE_CAPACITY = DISTRICT_LATTICE_SIDE * DISTRICT_LATTICE_SIDE
const DISTRICT_PLACEMENT_SPAN = 0.5

/**
 * Binding table approved from the live Hermes Bots group inventory. Matching
 * is exact after Unicode case folding; unknown groups deliberately fall back
 * to the garden rather than inferring authority from keywords.
 */
export const GROUP_DISTRICT_BINDINGS = Object.freeze([
  ['Arts Studio', 'garden'],
  ['Acceptance & Release', 'review'],
  ['Archive and Acquisition', 'library'],
  ['CI Repair Triage', 'triage'],
  ['Community Intake', 'bus'],
  ['Content Studio', 'library'],
  ['Control Plane Incidents', 'council'],
  ['Core Runtime & UX Repairs', 'project'],
  ['Data & Performance Repairs', 'lab'],
  ['Editorial Desk', 'library'],
  ['Engineering Guild', 'project'],
  ['Federation Council', 'council'],
  ['Knowledge Commons', 'library'],
  ['Memory Stewardship', 'library'],
  ['Operations and Release', 'depot'],
  ['PR Merge Train', 'depot'],
  ['Research Lab', 'lab'],
  ['Research Review Board', 'review'],
  ['Upstream Hermes Maintenance', 'project']
] as const satisfies readonly (readonly [string, DestinationId])[])

const DISTRICT_BY_GROUP = new Map(
  GROUP_DISTRICT_BINDINGS.map(([name, destination]) => [name.toLocaleLowerCase(), destination])
)

const GROUP_PRECEDENCE = new Map(GROUP_DISTRICT_BINDINGS.map(([name], index) => [name.toLocaleLowerCase(), index]))

// Mirrors the checked-in v2 manifest anchors. Placement offsets are bounded
// presentation hints; navigation still owns all runtime movement.
const DISTRICT_ANCHORS: Readonly<Record<DestinationId, Vec3>> = Object.freeze({
  bus: { x: 0, y: 0, z: -1 },
  council: { x: 27, y: 0, z: 31 },
  depot: { x: -31, y: 0, z: 12 },
  garden: { x: -8, y: 0, z: 34 },
  lab: { x: 10, y: 0, z: 32 },
  library: { x: -22, y: 0, z: 28 },
  project: { x: 18, y: 0, z: 38 },
  review: { x: 33, y: 0, z: 10 },
  triage: { x: 4, y: 0, z: 25 },
  unavailable: { x: 0, y: 0, z: 0 },
  unknown: { x: -8, y: 0, z: 34 }
})

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function safeString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined
  }

  const normalized = value.trim()

  return normalized && normalized.length <= MAX_PRESENTATION_CHARS ? normalized : undefined
}

function groupId(name: string): string {
  const slug = name
    .normalize('NFKC')
    .toLocaleLowerCase()
    .replace(/[^a-z0-9]+/gu, '-')
    .replace(/^-+|-+$/gu, '')

  return slug || `group-${hash(name).toString(16)}`
}

function membership(idValue: unknown, nameValue: unknown): LunarGroupMembership | undefined {
  const name = safeString(nameValue)

  if (!name) {
    return undefined
  }

  return { id: safeString(idValue) ?? groupId(name), name }
}

function directGroups(meta: Record<string, unknown>): readonly LunarGroupMembership[] {
  const values = [
    ...(Array.isArray(meta.groups) ? meta.groups.slice(0, MAX_GROUPS_PER_PROFILE) : []),
    ...(meta.group === null || meta.group === undefined ? [] : [meta.group])
  ]

  const byId = new Map<string, LunarGroupMembership>()

  for (const value of values) {
    const group = membership(undefined, value)

    if (group) {
      byId.set(group.id.toLocaleLowerCase(), group)
    }
  }

  return [...byId.values()]
}

function memberOwner(
  value: unknown,
  sourceConnectionId: string
): { connectionId: string; profile: string } | undefined {
  const member = record(value)
  const route = record(member.route)

  const connectionId =
    safeString(member.connectionId) ?? safeString(route.connectionId) ?? safeString(sourceConnectionId)

  const profile =
    safeString(member.targetProfile) ??
    safeString(route.targetProfile) ??
    safeString(route.profile) ??
    safeString(member.name)

  return connectionId && profile ? { connectionId, profile } : undefined
}

function projectedGroups(row: Record<string, unknown>, sourceConnectionId: string): readonly ProjectedMembership[] {
  const uiMeta = record(row.ui_meta)
  const projection = record(uiMeta['hermes-bots-groups'])
  const rooms = Object.entries(record(projection.rooms)).slice(0, MAX_PROJECTED_ROOMS)
  const projected: ProjectedMembership[] = []

  for (const [roomKey, roomValue] of rooms) {
    const room = record(roomValue)
    const fallbackName = roomKey.replace(/^(?:id|name):/u, '')
    const group = membership(room.roomId ?? roomKey.replace(/^id:/u, ''), room.name ?? fallbackName)

    if (!group || !Array.isArray(room.members)) {
      continue
    }

    for (const member of room.members.slice(0, MAX_ROOM_MEMBERS)) {
      const owner = memberOwner(member, sourceConnectionId)

      if (owner) {
        projected.push({ ...owner, group })
      }
    }
  }

  return projected
}

function parseSourceDetails(value: unknown, connectionId: string): SourceDetails {
  const rows = Array.isArray(record(value).profiles)
    ? (record(value).profiles as unknown[]).slice(0, MAX_PROFILE_ROWS)
    : []

  const profiles = new Map<string, RichProfileRow>()
  const projected: ProjectedMembership[] = []

  for (const value of rows) {
    const row = record(value)
    const name = safeString(row.name)

    if (!name) {
      continue
    }

    const uiMeta = record(row.ui_meta)
    const botMeta = record(uiMeta['hermes-bots'])
    profiles.set(name, {
      displayName: safeString(row.display_name),
      groups: directGroups(botMeta),
      name,
      title: safeString(botMeta.title) ?? safeString(row.title) ?? safeString(row.display_name)
    })
    projected.push(...projectedGroups(row, connectionId))
  }

  return { projected, rows: profiles }
}

function groupsFor(
  details: SourceDetails | undefined,
  connectionId: string,
  profile: string
): readonly LunarGroupMembership[] {
  const byName = new Map<string, LunarGroupMembership>()

  for (const group of details?.rows.get(profile)?.groups ?? []) {
    byName.set(group.name.toLocaleLowerCase(), group)
  }

  for (const projected of details?.projected ?? []) {
    if (projected.connectionId === connectionId && projected.profile === profile) {
      // The room projection carries the durable room id, so it wins over the
      // same membership's name-derived id from the per-profile convenience field.
      byName.set(projected.group.name.toLocaleLowerCase(), projected.group)
    }
  }

  return [...byName.values()].sort((left, right) => {
    const leftRank = GROUP_PRECEDENCE.get(left.name.toLocaleLowerCase()) ?? Number.MAX_SAFE_INTEGER
    const rightRank = GROUP_PRECEDENCE.get(right.name.toLocaleLowerCase()) ?? Number.MAX_SAFE_INTEGER

    return leftRank - rightRank || left.id.localeCompare(right.id)
  })
}

function primaryPlacement(groups: readonly LunarGroupMembership[]): {
  destination: DestinationId
  group?: LunarGroupMembership
} {
  for (const [name, destination] of GROUP_DISTRICT_BINDINGS) {
    const group = groups.find(candidate => candidate.name.toLocaleLowerCase() === name.toLocaleLowerCase())

    if (group) {
      return { destination, group }
    }
  }

  return { destination: 'garden' }
}

function hash(value: string): number {
  let result = 2166136261

  for (let index = 0; index < value.length; index += 1) {
    result ^= value.charCodeAt(index)
    result = Math.imul(result, 16777619)
  }

  return result >>> 0
}

export function assignStablePlacementSlots(
  keys: readonly EntityKey[],
  capacity = DISTRICT_LATTICE_CAPACITY,
  previous: ReadonlyMap<EntityKey, number | undefined> = new Map()
): ReadonlyMap<EntityKey, number | undefined> {
  if (
    !Number.isSafeInteger(capacity) ||
    capacity <= 0 ||
    capacity > DISTRICT_LATTICE_CAPACITY ||
    (capacity & (capacity - 1)) !== 0
  ) {
    throw new Error('placement lattice capacity must be a positive power of two')
  }

  const slots = new Map<EntityKey, number | undefined>()
  const occupied = new Uint8Array(capacity)
  const orderedKeys = [...keys].sort()

  for (const key of orderedKeys) {
    const retained = previous.get(key)

    if (retained !== undefined && retained >= 0 && retained < capacity && !occupied[retained]) {
      occupied[retained] = 1
      slots.set(key, retained)
    }
  }

  for (const key of orderedKeys) {
    if (slots.has(key)) {
      continue
    }

    let slot = hash(key) % capacity
    const step = capacity === 1 ? 1 : (hash(`probe:${key}`) % capacity) | 1
    let probes = 0

    while (probes < capacity && occupied[slot]) {
      slot = (slot + step) % capacity
      probes += 1
    }

    if (probes === capacity) {
      slots.set(key, undefined)

      continue
    }

    occupied[slot] = 1
    slots.set(key, slot)
  }

  return slots
}

function positionFor(destination: DestinationId, slot: number, latticeSide: number): Vec3 {
  const anchor = DISTRICT_ANCHORS[destination]
  const column = slot % latticeSide
  const row = Math.floor(slot / latticeSide)
  const scale = latticeSide === 1 ? 0 : DISTRICT_PLACEMENT_SPAN / (latticeSide - 1)

  return {
    x: Number((anchor.x + column * scale - DISTRICT_PLACEMENT_SPAN / 2).toFixed(4)),
    y: anchor.y,
    z: Number((anchor.z + row * scale - DISTRICT_PLACEMENT_SPAN / 2).toFixed(4))
  }
}

function basePresentation(
  roster: DesktopAgentRoster,
  entity: LunarEntity,
  details: SourceDetails | undefined,
  metadata: LunarPresentationMetadata
): Omit<LunarEntityPresentation, 'placement'> {
  if (entity.identity.kind !== 'profile') {
    return { groups: [], metadata }
  }

  const agent = roster.agents.find(
    candidate =>
      candidate.connectionId === entity.identity.connectionId && candidate.profile === entity.identity.profile
  )

  const source = roster.sources.find(candidate => candidate.connectionId === entity.identity.connectionId)
  const groups = groupsFor(details, entity.identity.connectionId, entity.identity.profile)
  const title = details?.rows.get(entity.identity.profile)?.title

  return {
    ...(title ? { configuredTitle: title } : {}),
    groups,
    metadata,
    ...(safeString(agent?.handle) ? { profileHandle: safeString(agent?.handle) } : {}),
    ...(safeString(agent?.connectionLabel ?? source?.label)
      ? { sourceLabel: safeString(agent?.connectionLabel ?? source?.label) }
      : {})
  }
}

/**
 * Adds bounded exact-source presentation metadata to fleet-authoritative rows.
 * Request failures are presentation failures only: enumeration, identity,
 * availability, and authority remain exactly as normalizeRoster supplied.
 */
export async function enrichBotRosterEntities(
  roster: DesktopAgentRoster,
  entities: readonly LunarEntity[],
  request: ScopedProfilesListRequest,
  provenance: BotMetadataProvenance = new Map(),
  options: BotRosterPlacementOptions = {}
): Promise<readonly LunarEntity[]> {
  const latticeSide =
    options.latticeSide !== undefined &&
    Number.isSafeInteger(options.latticeSide) &&
    options.latticeSide > 0 &&
    options.latticeSide <= DISTRICT_LATTICE_SIDE &&
    (options.latticeSide & (options.latticeSide - 1)) === 0
      ? options.latticeSide
      : DISTRICT_LATTICE_SIDE

  const latticeCapacity = latticeSide * latticeSide
  const placementState = options.placementState

  if (placementState && placementState.capacity !== latticeCapacity) {
    placementState.assignments.clear()
    placementState.capacity = latticeCapacity
  }

  const representatives = new Map<string, string>()

  for (const agent of [...roster.agents].sort(
    (left, right) => left.connectionId.localeCompare(right.connectionId) || left.profile.localeCompare(right.profile)
  )) {
    if (!representatives.has(agent.connectionId) && representatives.size < MAX_SOURCES) {
      representatives.set(agent.connectionId, agent.profile)
    }
  }

  const detailsBySource = new Map<string, SourceDetails>()
  const metadataBySource = new Map<string, LunarPresentationMetadata>()

  for (const [connectionId, profile] of representatives) {
    try {
      detailsBySource.set(connectionId, parseSourceDetails(await request(connectionId, profile), connectionId))
      metadataBySource.set(
        connectionId,
        provenance.get(connectionId) ?? { source: `profiles:${connectionId}`, state: 'fresh' }
      )
    } catch {
      // The fleet row remains visible with its source-derived label and handle.
      metadataBySource.set(
        connectionId,
        provenance.get(connectionId) ?? { source: `profiles:${connectionId}`, state: 'unavailable' }
      )
    }
  }

  const staged: readonly StagedEntity[] = entities.map(entity => {
    if (entity.identity.kind !== 'profile') {
      return { destination: entity.destination, entity }
    }

    const presentation = basePresentation(
      roster,
      entity,
      detailsBySource.get(entity.identity.connectionId),
      metadataBySource.get(entity.identity.connectionId) ?? {
        source: `profiles:${entity.identity.connectionId}`,
        state: 'unavailable'
      }
    )

    const primary = primaryPlacement(presentation.groups)
    const destination = entity.destination === 'unavailable' ? 'unavailable' : primary.destination

    return { entity, destination, presentation, primary }
  })

  const slotsByDestination = new Map<DestinationId, ReadonlyMap<EntityKey, number | undefined>>()

  const globalRanks = new Map(
    staged
      .filter(value => value.presentation !== undefined)
      .map(value => value.entity.key)
      .sort()
      .map((key, index) => [key, index])
  )

  const activeDestinations = new Set(staged.map(value => value.destination))

  for (const destination of activeDestinations) {
    const keys = staged
      .filter(value => value.presentation !== undefined && value.destination === destination)
      .map(value => value.entity.key)
      .sort()

    const assigned = assignStablePlacementSlots(keys, latticeCapacity, placementState?.assignments.get(destination))
    slotsByDestination.set(destination, assigned)
    placementState?.assignments.set(destination, new Map(assigned))
  }

  if (placementState) {
    for (const destination of placementState.assignments.keys()) {
      if (!activeDestinations.has(destination)) {
        placementState.assignments.delete(destination)
      }
    }
  }

  return staged.map(value => {
    if (!value.presentation || !value.primary) {
      return value.entity
    }

    const slot = slotsByDestination.get(value.destination)?.get(value.entity.key)
    const overflow = slot === undefined || (globalRanks.get(value.entity.key) ?? 0) >= NEAR_WORKER_BUDGET

    const placement = {
      lodHint: overflow ? 1 : 0,
      overflow,
      ...(value.primary.group ? { primaryGroupId: value.primary.group.id } : {}),
      ...(slot === undefined ? {} : { slot })
    }

    return {
      ...value.entity,
      destination: value.destination,
      position: slot === undefined ? undefined : positionFor(value.destination, slot, latticeSide),
      presentation: { ...value.presentation, placement }
    }
  })
}
