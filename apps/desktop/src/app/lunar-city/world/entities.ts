import { characterPresentationForEntity } from '../character-presentation'
import type {
  AuthorityState,
  CharacterAssetManifest,
  DestinationId,
  EntityIdentity,
  EntityKey,
  LunarCitySnapshot,
  LunarEntity,
  Vec3,
  WorkerCharacterPresentation
} from '../model'
import type { WorkerSocialPose } from '../worker-social'

import type { EntityPositionState } from './entity-position-state'

export interface EntityVisual {
  isAnimationActive?(): boolean
  supportsAnimation?(clip: string): boolean
  setFacing?(yaw: number): void
  dispose?(): void
  setAnimation?(clip: string): void
  setLod?(lodIndex: number): void
  setPosition?(position: Vec3): void
  setStaticPose?(pose: string): void
}

export interface InstancedEntityMember {
  animation: string
  character?: WorkerCharacterPresentation
  identity: EntityIdentity
  key: EntityKey
  position: Vec3
  variant: string | undefined
}

/** Dynamic, exact worker focus data consumed by the camera and occlusion pass. */
export interface EntityFocusMetadata {
  cameraAnchor: Vec3
  focusEntityKey: EntityKey
  occlusionGroup: string
}

export interface InstancedEntityGroup {
  dispose?(): void
  sync(members: readonly InstancedEntityMember[]): void
}

export interface EntityPresentationFactory {
  createAnimated(
    entity: LunarEntity,
    variant: string | undefined,
    character?: WorkerCharacterPresentation
  ): EntityVisual
  createInstancedGroup(groupKey: string): InstancedEntityGroup
}

export interface LodNode {
  setEnabled?(enabled: boolean): void
}

export interface LodEntry {
  distance: number
  node: LodNode
}

export interface LodSelection {
  distance: number
  lodAdvance: number
  selected: boolean
}

export function selectLodIndex(entries: readonly Pick<LodEntry, 'distance'>[], selection: LodSelection): number {
  if (entries.length === 0) {
    return -1
  }

  let index = 0

  if (!selection.selected) {
    for (let candidate = 1; candidate < entries.length; candidate += 1) {
      if (selection.distance >= entries[candidate]!.distance) {
        index = candidate
      }
    }

    index = Math.min(entries.length - 1, index + Math.max(0, selection.lodAdvance))
  }

  return index
}

export interface EntityPositionRequest {
  entity: LunarEntity
  reason: 'spawn' | 'restore' | 'correction'
  candidate: Vec3
  previous?: Vec3
}

export interface EntityRegistryOptions {
  positionState?: EntityPositionState
  resolvePosition?: (request: EntityPositionRequest) => Vec3 | undefined
  characterAssets?: CharacterAssetManifest
  diagnostic?: (message: string) => void
  factory: EntityPresentationFactory
  focusAnchors?: Map<EntityKey, () => Vec3 | undefined>
  focusMetadata?: Map<EntityKey, () => EntityFocusMetadata | undefined>
  workerClips: ReadonlySet<string>
}

interface RetainedEntity {
  /** Last authoritative source origin, separate from the locally resolved navigation position. */
  sourcePosition?: Vec3
  socialPose?: WorkerSocialPose
  animation: string
  authority: AuthorityState
  character?: WorkerCharacterPresentation
  entity: LunarEntity
  lodFloor: number
  lodIndex: number
  moving: boolean
  nearby: boolean
  position: Vec3
  visual: EntityVisual | undefined
}

const MAX_ANIMATED_WORKERS = 24
const CONTINUOUS_WORKER_ANIMATIONS = new Set(['listen', 'talk', 'think', 'walk', 'work'])

export interface AggregatePopulation {
  animations: Readonly<Record<string, number>>
  total: number
}

export interface NavigationPresentationEntity {
  animation: string
  key: EntityKey
  position: Vec3
}

function copied(position: Vec3 | undefined): Vec3 {
  return position ? { x: position.x, y: position.y, z: position.z } : { x: 0, y: 0, z: 0 }
}

function groupKey(record: RetainedEntity, lodIndex = record.lodIndex): string {
  const base = record.entity.variant
    ? `worker:${record.entity.variant}:${record.animation}`
    : `worker:${record.animation}`

  const character = record.character?.kitId ? `:kit:${record.character.kitId}` : ''

  return lodIndex === 0 ? `${base}${character}` : `${base}${character}:lod:${lodIndex}`
}

function isOverflow(record: RetainedEntity): boolean {
  return record.entity.presentation?.placement.overflow === true
}

function hasPhysicalPlacement(record: RetainedEntity): boolean {
  const placement = record.entity.presentation?.placement

  return placement === undefined || placement.slot !== undefined
}

function wantsIndividualVisual(record: RetainedEntity, selection: EntityKey | undefined): boolean {
  return (
    hasPhysicalPlacement(record) &&
    (record.entity.key === selection ||
      (!isOverflow(record) &&
        (record.moving ||
          record.nearby ||
          record.animation === 'walk' ||
          (record.character !== undefined && record.lodFloor === 0))))
  )
}

function effectiveLodIndex(
  record: RetainedEntity,
  selection: EntityKey | undefined,
  activeKeys: ReadonlySet<EntityKey>
): number {
  if (record.entity.key === selection) {
    return 0
  }

  return wantsIndividualVisual(record, selection) && !activeKeys.has(record.entity.key)
    ? Math.max(1, record.lodIndex)
    : record.lodIndex
}

function presentedAnimation(record: RetainedEntity): string {
  return record.moving ? 'walk' : record.socialPose?.animation ?? record.animation
}

function immutableAggregate(total: number, animations: Map<string, number>): AggregatePopulation {
  return Object.freeze({ animations: Object.freeze(Object.fromEntries(animations)), total })
}

/**
 * Enables one and only one declared LOD subtree.  Selection deliberately pins
 * to the near representation, so an authoritative selected worker cannot be
 * removed by a quality downgrade.
 */
export function applyLodSelection(entries: readonly LodEntry[], selection: LodSelection): number {
  const index = selectLodIndex(entries, selection)

  for (const [candidate, entry] of entries.entries()) {
    entry.node.setEnabled?.(candidate === index)
  }

  return index
}

/**
 * Retains presentation objects by opaque identity.  Snapshot objects are read
 * only: all mutable position, animation, and visual resources live here.
 */
export function createEntityRegistry(options: EntityRegistryOptions) {
  const records = new Map<EntityKey, RetainedEntity>()
  const groups = new Map<string, InstancedEntityGroup>()
  const aggregates = new Map<DestinationId, AggregatePopulation>()
  const diagnostics = new Set<string>()
  const focusAnchors = options.focusAnchors
  const focusMetadata = options.focusMetadata
  let selected: EntityKey | undefined
  let rankedKeys: readonly EntityKey[] = []
  let activeKeys = new Set<EntityKey>()
  let nextActiveKeys = new Set<EntityKey>()
  const groupCounts = new Map<string, number>()
  let reducedMotion = false
  let disposed = false

  const rememberPosition = (record: RetainedEntity, observed = false): void => {
    if (record.authority === 'authoritative') {
      options.positionState?.save(record.entity.key, {
        position: record.position,
        sourcePosition: record.sourcePosition
      }, observed)
    }
  }

  const resolvePosition = (request: EntityPositionRequest): Vec3 | undefined => {
    const result = options.resolvePosition ? options.resolvePosition(request) : request.candidate

    return result && [result.x, result.y, result.z].every(Number.isFinite) ? copied(result) : undefined
  }

  const refreshActiveKeys = (): boolean => {
    nextActiveKeys.clear()

    if (selected) {
      const selectedRecord = records.get(selected)

      if (selectedRecord && hasPhysicalPlacement(selectedRecord)) {
        nextActiveKeys.add(selected)
      }
    }

    for (const key of rankedKeys) {
      if (nextActiveKeys.size >= MAX_ANIMATED_WORKERS) {
        break
      }

      const record = records.get(key)

      if (record && key !== selected && wantsIndividualVisual(record, selected)) {
        nextActiveKeys.add(key)
      }
    }

    let changed = activeKeys.size !== nextActiveKeys.size

    if (!changed) {
      for (const key of activeKeys) {
        if (!nextActiveKeys.has(key)) {
          changed = true

          break
        }
      }
    }

    const previous = activeKeys
    activeKeys = nextActiveKeys
    nextActiveKeys = previous

    return changed
  }

  const diagnoseOnce = (message: string): void => {
    if (!diagnostics.has(message)) {
      diagnostics.add(message)
      options.diagnostic?.(message)
    }
  }

  const release = (record: RetainedEntity): void => {
    record.visual?.dispose?.()
    record.visual = undefined
    focusAnchors?.delete(record.entity.key)
    focusMetadata?.delete(record.entity.key)
  }

  const publishAnchor = (record: RetainedEntity): void => {
    focusAnchors?.set(record.entity.key, () =>
      record.authority === 'authoritative' && hasPhysicalPlacement(record) ? copied(record.position) : undefined
    )
    focusMetadata?.set(record.entity.key, () =>
      record.authority === 'authoritative' && hasPhysicalPlacement(record)
        ? {
            cameraAnchor: copied(record.position),
            focusEntityKey: record.entity.key,
            occlusionGroup: 'workers'
          }
        : undefined
    )
  }

  const syncPresentation = (): void => {
    if (disposed) {
      return
    }

    const nextGroups = new Map<string, InstancedEntityMember[]>()
    const nextAggregates = new Map<DestinationId, { animations: Map<string, number>; total: number }>()
    groupCounts.clear()

    for (const record of records.values()) {
      const aggregate = nextAggregates.get(record.entity.destination) ?? {
        animations: new Map<string, number>(),
        total: 0
      }

      aggregate.total += 1
      aggregate.animations.set(record.animation, (aggregate.animations.get(record.animation) ?? 0) + 1)
      nextAggregates.set(record.entity.destination, aggregate)

      if (!hasPhysicalPlacement(record)) {
        record.visual?.dispose?.()
        record.visual = undefined

        continue
      }

      const lodIndex = effectiveLodIndex(record, selected, activeKeys)

      if (activeKeys.has(record.entity.key)) {
        if (!record.visual) {
          record.visual = options.factory.createAnimated(record.entity, record.entity.variant, record.character)
        }

        record.visual.setPosition?.(copied(record.position))
        record.visual.setLod?.(lodIndex)
        record.visual.setFacing?.(record.socialPose?.facing ?? 0)

        const animation = presentedAnimation(record)

        if (!reducedMotion && options.workerClips.has(animation)) {
          record.visual.setAnimation?.(animation)
        } else {
          record.visual.setStaticPose?.('idle')
          diagnoseOnce(`worker animation clip unavailable: ${animation}`)
        }

        continue
      }

      record.visual?.dispose?.()
      record.visual = undefined
      const key = groupKey(record, lodIndex)
      groupCounts.set(key, (groupCounts.get(key) ?? 0) + 1)
      const members = nextGroups.get(key) ?? []
      members.push({
        animation: record.animation,
        ...(record.character ? { character: record.character } : {}),
        identity: record.entity.identity,
        key: record.entity.key,
        position: copied(record.position),
        variant: record.entity.variant
      })
      nextGroups.set(key, members)
    }

    for (const [key, group] of groups) {
      if (!nextGroups.has(key)) {
        group.dispose?.()
        groups.delete(key)
      }
    }

    for (const [key, members] of nextGroups) {
      const group = groups.get(key) ?? options.factory.createInstancedGroup(key)
      groups.set(key, group)
      group.sync(members)
    }

    aggregates.clear()

    for (const [destination, aggregate] of nextAggregates) {
      aggregates.set(destination, immutableAggregate(aggregate.total, aggregate.animations))
    }
  }

  return {
    setReducedMotion(reduced: boolean): void { reducedMotion = reduced; syncPresentation() },
    setSocialPoses(poses: ReadonlyMap<EntityKey, WorkerSocialPose>): void {
      let changed = false

      for (const [key, record] of records) {
        const next = record.authority === 'authoritative' && !record.moving ? poses.get(key) : undefined

        if (next?.animation !== record.socialPose?.animation || next?.facing !== record.socialPose?.facing) {
          record.socialPose = next
          changed = true
        }
      }

      if (changed) {syncPresentation()}
    },
    activeAnimationCount(): number {
      let count = 0

      for (const record of records.values()) {
        if (!reducedMotion && record.visual && (record.visual.isAnimationActive?.() ?? (record.moving || CONTINUOUS_WORKER_ANIMATIONS.has(presentedAnimation(record))))) {
          count += 1
        }
      }

      return count
    },
    aggregate(destination: DestinationId): AggregatePopulation | undefined {
      return aggregates.get(destination)
    },
    applyLodPolicy(
      resolveIndex: (key: EntityKey, position: Vec3, isSelected: boolean) => number,
      isNearby?: (key: EntityKey, position: Vec3, isSelected: boolean) => boolean
    ): void {
      if (disposed) {
        return
      }

      let changed = false

      for (const record of records.values()) {
        if (!hasPhysicalPlacement(record)) {
          const nextIndex = record.lodFloor

          if (record.lodIndex !== nextIndex) {
            record.lodIndex = nextIndex
            changed = true
          }

          if (record.nearby) {
            record.nearby = false
            changed = true
          }

          continue
        }

        const nextIndex = Math.max(
          record.lodFloor,
          Math.floor(resolveIndex(record.entity.key, record.position, record.entity.key === selected))
        )

        if (record.lodIndex !== nextIndex) {
          record.lodIndex = nextIndex
          changed = true
        }

        const nextNearby = !isOverflow(record)
          ? (isNearby?.(record.entity.key, record.position, record.entity.key === selected) ?? false)
          : false

        if (record.nearby !== nextNearby) {
          record.nearby = nextNearby
          changed = true
        }
      }

      if (refreshActiveKeys() || changed) {
        syncPresentation()
      }
    },
    dispose(): void {
      if (disposed) {
        return
      }

      disposed = true

      for (const record of records.values()) {
        rememberPosition(record)
        release(record)
      }

      for (const group of groups.values()) {
        group.dispose?.()
      }

      records.clear()
      groups.clear()
      groupCounts.clear()
      aggregates.clear()
    },
    entity(key: EntityKey): Readonly<RetainedEntity> | undefined {
      return records.get(key)
    },
    hasActiveAnimations(): boolean {
      return this.activeAnimationCount() > 0
    },
    instancedGroup(key: string): { count: number } | undefined {
      const group = groups.get(key)

      if (!group) {
        return undefined
      }

      return { count: groupCounts.get(key) ?? 0 }
    },
    navigationEntity(key: EntityKey): NavigationPresentationEntity | undefined {
      const record = records.get(key)

      if (!record || !hasPhysicalPlacement(record)) {
        return undefined
      }

      return {
        get animation() {
          return presentedAnimation(record)
        },
        set animation(value: string) {
          if (value === 'walk') {
            record.moving = true
          } else {
            record.animation = value
          }
        },
        get key() {
          return record.entity.key
        },
        get position() {
          return record.position
        },
        set position(value: Vec3) {
          record.position = copied(value)
        }
      }
    },
    reconcile(snapshot: LunarCitySnapshot): void {
      if (disposed) {
        return
      }

      const incomingKeys = new Set(snapshot.entities.keys())

      for (const [key, record] of records) {
        if (!incomingKeys.has(key)) {
          release(record)
          records.delete(key)
          options.positionState?.forget(key)
        }
      }

      for (const entity of snapshot.entities.values()) {
        const previous = records.get(entity.key)

        const restored = previous ? undefined : options.positionState?.read(entity.key)

        const initialPosition = previous?.position ?? resolvePosition({
          entity,
          reason: restored ? 'restore' : 'spawn',
          candidate: copied(restored?.position ?? entity.position)
        })

        if (!initialPosition) {
          options.positionState?.forget(entity.key)

          continue
        }

        const record: RetainedEntity = previous ?? {
          sourcePosition: restored?.sourcePosition ?? (entity.authority === 'authoritative' && entity.position ? copied(entity.position) : undefined),
          animation: entity.animation,
          authority: entity.authority,
          character: options.characterAssets
            ? characterPresentationForEntity(entity, options.characterAssets)
            : undefined,
          entity,
          lodFloor: entity.presentation?.placement.lodHint ?? 0,
          lodIndex: entity.presentation?.placement.lodHint ?? 0,
          moving: false,
          nearby: false,
          position: initialPosition,
          visual: undefined
        }

        const previousLodFloor = record.lodFloor
        record.entity = entity
        record.animation = entity.animation
        record.authority = entity.authority
        record.character = options.characterAssets
          ? characterPresentationForEntity(entity, options.characterAssets)
          : undefined
        record.lodFloor = entity.presentation?.placement.lodHint ?? 0

        if (record.lodIndex === previousLodFloor || record.lodIndex < record.lodFloor) {
          record.lodIndex = record.lodFloor
        }

        if (entity.authority === 'authoritative' && entity.position &&
          (record.sourcePosition?.x !== entity.position.x || record.sourcePosition?.y !== entity.position.y ||
            record.sourcePosition?.z !== entity.position.z)) {
          record.sourcePosition = copied(entity.position)
          const corrected = resolvePosition({ entity, reason: 'correction', candidate: copied(entity.position), previous: copied(record.position) })

          if (corrected) {record.position = corrected}
        }

        record.moving ||= entity.animation === 'walk'
        records.set(entity.key, record)
        rememberPosition(record, true)
        publishAnchor(record)
      }

      if (selected && !records.has(selected)) {
        selected = undefined
      }

      rankedKeys = [...records.keys()].sort()
      refreshActiveKeys()
      syncPresentation()
    },
    setMoving(key: EntityKey, moving: boolean): void {
      const record = records.get(key)

      if (!record || disposed) {
        return
      }

      record.moving = moving
      refreshActiveKeys()
      syncPresentation()
    },
    setPosition(key: EntityKey, position: Vec3): void {
      const record = records.get(key)

      if (!record || disposed) {
        return
      }

      record.position = copied(position)
      rememberPosition(record)
      record.visual?.setPosition?.(copied(record.position))
    },
    setSelection(key: EntityKey | undefined): void {
      selected = key
      refreshActiveKeys()
      syncPresentation()
    },
    syncMotion(): void {
      if (disposed) {
        return
      }

      for (const record of records.values()) {
        rememberPosition(record)
        record.visual?.setPosition?.(copied(record.position))

        const animation = presentedAnimation(record)

        if (!reducedMotion && record.visual && options.workerClips.has(animation)) {
          record.visual.setAnimation?.(animation)
        }
      }
    }
  }
}
