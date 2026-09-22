import type {
  BabylonGlowLayerLike,
  BabylonImportResultLike,
  BabylonMeshLike,
  BabylonNodeLike,
  BabylonShadowGeneratorLike,
  CameraControlState,
  CameraIntent,
  CharacterAssetManifest,
  DestinationId,
  EntityKey,
  LeaderAnimationState,
  LeaderId,
  LeaderStateClipMap,
  LunarCityIntent,
  LunarCityNodeMetadata,
  LunarCitySnapshot,
  LunarCityWorkerPickMetadata,
  LunarCityWorldHandle,
  LunarCityWorldModules,
  LunarEntity,
  ModelManifestEntry,
  QualityTier,
  ReviewLeaderAsset,
  Vec3,
  WorkerCharacterPresentation,
  WorldBounds,
  WorldManifestV2
} from '../model'

import { createArrivalSlots } from './arrival-slots'
import { createBuildingActivity } from './building-activity'
import {
  type CameraController,
  type CameraLike,
  type CameraPickTarget,
  createCameraController
} from './camera-controller'
import { createCharacterAnimations } from './character-animations'
import {
  applyLodSelection,
  createEntityRegistry,
  type EntityFocusMetadata,
  type EntityPresentationFactory,
  type EntityVisual,
  type InstancedEntityGroup,
  type InstancedEntityMember,
  type LodEntry,
  selectLodIndex
} from './entities'
import { createEntityPositionState } from './entity-position-state'
import { createInteriorNavigation } from './interior-navigation'
import { createLeaderLife, type LeaderLifeMode, type LeaderLifePresentation, measuredLeaderEnvelope, safeLeaderStrollPoints } from './leader-life'
import { observedLeaderLife } from './leader-observed-life'
import { createNavigationController, type NavigationQuery } from './navigation'
import { createOcclusionController, type OcclusionCandidate, type OcclusionSelection } from './occlusion'
import { createBabylonPerfAdapter } from './perf-adapter'
import { createProjectSiteRuntime } from './project-site-runtime'
import { animationDistanceUnits, applyQualitySettings, createQualityController } from './quality'
import { importReviewLeader, projectWorldPoint } from './review-assets'
import { createFrameScheduler } from './scheduler'
import { applyHistoricalWorkerFit, cloneWorkerHierarchy, fitHistoricalWorker, type HistoricalWorkerFit } from './worker-hierarchy'
import { createWorkerSocialRuntime } from './worker-social-runtime'
import { createRouteNavigationQuery } from './world-navigation'

const LEADER_STATES: readonly LeaderAnimationState[] = [
  'acknowledging',
  'idle',
  'listening',
  'talking',
  'thinking',
  'unavailable'
]

const CONTINUOUS_LEADER_STATES = new Set<LeaderAnimationState>(['idle', 'listening', 'talking', 'thinking', 'walk'])

/** `Scene.FOGMODE_LINEAR`, inlined so the fog needs no extra Babylon import. */
const FOG_MODE_LINEAR = 3
/** `ImageProcessingConfiguration.TONEMAPPING_ACES`, inlined for the same reason. */
const TONE_MAPPING_ACES = 1
/**
 * One 1024² cascade is enough for a settlement that fits inside the camera's
 * bounded 18–120 unit zoom, and keeps the shadow pass to a single low-cost
 * render of the near-LOD casters.
 */
const SHADOW_MAP_SIZE = 1024

const LEADER_IDS = [
  'owl',
  'fox',
  'badger',
  'otter',
  'bird',
  'stag',
  'elephant',
  'cat',
  'capybara',
  'lion',
  'beaver',
  'monkey'
] as const satisfies readonly LeaderId[]

export interface LunarCitySceneHandle {
  setInteriorBuilding(id?: string): boolean
  getInteriorBuildings(): readonly { id: string; title: string }[]
  getWorkerPresentation: NonNullable<LunarCityWorldHandle['getWorkerPresentation']>
  requestWorkerInteraction(key: EntityKey, gesture?: string): boolean
  getWorkerEncounters: NonNullable<LunarCityWorldHandle['getWorkerEncounters']>
  readonly leaderStateClips: ReadonlyMap<string, LeaderStateClipMap>
  applySnapshot(snapshot: LunarCitySnapshot): void
  dispatchCamera(intent: CameraIntent): void
  getEntityCameraOrder(): readonly EntityKey[]
  projectWorldPoint(point: Vec3): { x: number; y: number; visible: boolean } | undefined
  getCameraState(): CameraControlState
  getPerfSnapshot(): NonNullable<ReturnType<NonNullable<LunarCityWorldHandle['getPerfSnapshot']>>>
  pick(clientX: number, clientY: number): CameraPickTarget | undefined
  reviewWorkerClips: ReadonlyMap<string, Readonly<Record<string, string>>>
  setReviewWorkerAnimation(id: string, state: string): void
  setLeaderLifeMode(id: LeaderId, mode: LeaderLifeMode | 'automatic'): void
  getLeaderLife(id: LeaderId): LeaderLifePresentation | undefined
  setLeaderAnimation(leaderId: LeaderId, state: LeaderAnimationState): void
  setVisible(visible: boolean): void
  setQuality(tier: QualityTier): void
  setReducedMotion(reduced: boolean): void
  render(): void
  dispose(): void
}

interface FocusMetadata extends OcclusionSelection {
  focusEntityKey: EntityKey
}

interface PlacedModel {
  focus: FocusMetadata
  lods: readonly LodEntry[]
}

function staticFocusKey(kind: 'leader' | 'model', value: string): EntityKey {
  return `lunar-city:${kind}:${encodeURIComponent(value)}` as EntityKey
}

function samePoint(left: Vec3 | undefined, right: Vec3 | undefined): boolean {
  return left?.x === right?.x && left?.y === right?.y && left?.z === right?.z
}

interface BabylonQuaternion {
  w: number
  x: number
  y: number
  z: number
}

interface BabylonAnimationGroupLike {
  hasMotion?: boolean
  reset?(): void
  clone?(name: string, targetConverter?: (target: unknown) => unknown): BabylonAnimationGroupLike | null
  dispose?(): void
  isPlaying?: boolean
  name: string
  start?(loop?: boolean, speedRatio?: number, from?: number, to?: number): void
  stop?(): void
}

interface BabylonInstancedMeshLike extends BabylonMeshLike {
  createInstance?(name: string): BabylonNodeLike
}

function manifestRotationQuaternion(rotation: Vec3): BabylonQuaternion {
  // TransformNode.rotation is Babylon's y-x-z Euler convention: yaw(y),
  // pitch(x), and roll(z). Keep this in sync with Vector3.toQuaternion().
  const halfRoll = rotation.z * 0.5
  const halfPitch = rotation.x * 0.5
  const halfYaw = rotation.y * 0.5
  const sinRoll = Math.sin(halfRoll)
  const cosRoll = Math.cos(halfRoll)
  const sinPitch = Math.sin(halfPitch)
  const cosPitch = Math.cos(halfPitch)
  const sinYaw = Math.sin(halfYaw)
  const cosYaw = Math.cos(halfYaw)

  return {
    x: cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll,
    y: sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll,
    z: cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll,
    w: cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll
  }
}

export function transformManifestPoint(model: Pick<ModelManifestEntry, 'transform'>, local: Vec3): Vec3 {
  const quaternion = manifestRotationQuaternion(model.transform.rotation)
  const x2 = quaternion.x + quaternion.x
  const y2 = quaternion.y + quaternion.y
  const z2 = quaternion.z + quaternion.z
  const xx = quaternion.x * x2
  const xy = quaternion.x * y2
  const xz = quaternion.x * z2
  const yy = quaternion.y * y2
  const yz = quaternion.y * z2
  const zz = quaternion.z * z2
  const wx = quaternion.w * x2
  const wy = quaternion.w * y2
  const wz = quaternion.w * z2

  const scaled = {
    x: local.x * model.transform.scale.x,
    y: local.y * model.transform.scale.y,
    z: local.z * model.transform.scale.z
  }

  return {
    x: model.transform.position.x + (1 - (yy + zz)) * scaled.x + (xy - wz) * scaled.y + (xz + wy) * scaled.z,
    y: model.transform.position.y + (xy + wz) * scaled.x + (1 - (xx + zz)) * scaled.y + (yz - wx) * scaled.z,
    z: model.transform.position.z + (xz - wy) * scaled.x + (yz + wx) * scaled.y + (1 - (xx + yy)) * scaled.z
  }
}

export function worldBoundsFromModel(model: Pick<ModelManifestEntry, 'bounds' | 'transform'>): WorldBounds {
  const corners = [model.bounds.min.x, model.bounds.max.x].flatMap(x =>
    [model.bounds.min.y, model.bounds.max.y].flatMap(y =>
      [model.bounds.min.z, model.bounds.max.z].map(z => transformManifestPoint(model, { x, y, z }))
    )
  )

  return {
    min: {
      x: Math.min(...corners.map(point => point.x)),
      y: Math.min(...corners.map(point => point.y)),
      z: Math.min(...corners.map(point => point.z))
    },
    max: {
      x: Math.max(...corners.map(point => point.x)),
      y: Math.max(...corners.map(point => point.y)),
      z: Math.max(...corners.map(point => point.z))
    }
  }
}

function worldCameraAnchor(model: ModelManifestEntry): Vec3 {
  return transformManifestPoint(model, model.cameraAnchor)
}

function metadataRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function tagNode(node: BabylonNodeLike, lunarCity: LunarCityNodeMetadata): void {
  node.metadata = { ...metadataRecord(node.metadata), lunarCity }
}

function copiedPoint(point: Vec3 | undefined): Vec3 {
  return point ? { x: point.x, y: point.y, z: point.z } : { x: 0, y: 0, z: 0 }
}

function workerPickMetadata(
  entity: { identity: LunarEntity['identity']; key: EntityKey; position?: Vec3; variant?: string },
  model: Pick<ModelManifestEntry, 'occlusionGroup'>,
  variant: string | undefined,
  character?: WorkerCharacterPresentation
): LunarCityWorkerPickMetadata {
  return {
    cameraAnchor: copiedPoint(entity.position),
    ...(character ? { character } : {}),
    entityKey: entity.key,
    focusEntityKey: entity.key,
    identity: entity.identity,
    kind: 'worker',
    modelId: 'workers',
    occlusionGroup: model.occlusionGroup,
    selectable: true,
    variant
  }
}

function tagWorkerNode(node: BabylonNodeLike, metadata: LunarCityWorkerPickMetadata): void {
  tagNode(node, metadata)

  if ('isPickable' in node) {
    ;(node as BabylonMeshLike).isPickable = true
  }
}

function allImportedNodes(result: BabylonImportResultLike): readonly BabylonNodeLike[] {
  return [...result.transformNodes, ...result.meshes]
}

function findNode(result: BabylonImportResultLike, name: string): BabylonNodeLike | undefined {
  return allImportedNodes(result).find(node => node.name === name)
}

function placeModel(
  result: BabylonImportResultLike,
  model: ModelManifestEntry,
  modules: LunarCityWorldModules,
  scene: ConstructorParameters<LunarCityWorldModules['TransformNode']>[1]
): PlacedModel {
  const root = findNode(result, `${model.id}:root`)

  if (!root) {
    throw new Error(`Lunar City model ${model.id} is missing its runtime root`)
  }

  const placement = new modules.TransformNode(`lunar-city:placement:${model.id}`, scene)
  const { position, rotation, scale } = model.transform
  const cameraAnchor = worldCameraAnchor(model)
  const focusEntityKey = staticFocusKey('model', model.id)

  placement.position?.set(position.x, position.y, position.z)
  placement.rotation?.set(rotation.x, rotation.y, rotation.z)
  placement.scaling?.set(scale.x, scale.y, scale.z)
  placement.setPivotPoint?.(new modules.Vector3(model.pivot.x, model.pivot.y, model.pivot.z))

  const loaderConversionRoot = result.meshes.find(node => node.name === '__root__') ?? root

  loaderConversionRoot.parent = placement
  tagNode(placement, {
    cameraAnchor,
    focusEntityKey,
    kind: 'landmark',
    modelId: model.id,
    occlusionGroup: model.occlusionGroup,
    selectable: model.id !== 'terrain'
  })

  const lods: LodEntry[] = []

  for (const lod of model.lods) {
    const node = findNode(result, lod.node)

    if (!node) {
      throw new Error(`Lunar City model ${model.id} is missing LOD node ${lod.node}`)
    }

    tagNode(node, { distance: lod.distance, kind: 'lod', modelId: model.id })
    lods.push({ distance: lod.distance, node })
  }

  // glTF imports commonly enable both subtrees.  Lunar City never does: one
  // declared representation is active for every model from the first frame.
  applyLodSelection(lods, { distance: 0, lodAdvance: 0, selected: false })

  for (const mesh of result.meshes) {
    tagNode(mesh, {
      cameraAnchor,
      focusEntityKey,
      kind: 'landmark-mesh',
      modelId: model.id,
      occlusionGroup: model.occlusionGroup,
      selectable: model.id !== 'terrain'
    })
  }

  return { focus: { cameraAnchor, focusEntityKey, occlusionGroup: model.occlusionGroup }, lods }
}

function gltfExtras(node: BabylonNodeLike): Record<string, unknown> {
  const gltf = metadataRecord(metadataRecord(node.metadata).gltf)

  return metadataRecord(gltf.extras)
}

/** Decorative authored GLB nodes are optional detail, never a source of city truth. */
function isDecorationNode(node: BabylonNodeLike): boolean {
  const semantic = gltfExtras(node).semantic

  return typeof semantic === 'string' && /:(?:plants|bench|cyan-fixture|wheels|signal)$/u.test(semantic)
}

function readLeaderStateClips(node: BabylonNodeLike, leaderId: LeaderId): LeaderStateClipMap {
  const extras = gltfExtras(node)

  if (extras.leaderId !== leaderId) {
    throw new Error(`leader ${leaderId} GLB metadata has a mismatched identity`)
  }

  const clips = metadataRecord(extras.stateClips)

  const entries = LEADER_STATES.map(state => {
    const clip = clips[state]
    const expected = `leader:${leaderId}:${state}`

    if (clip !== expected) {
      throw new Error(`leader ${leaderId} is missing authoritative ${state} clip metadata`)
    }

    return [state, expected]
  })

  return Object.freeze(Object.fromEntries(entries)) as LeaderStateClipMap
}

interface StructuredLeader {
  id: LeaderId
  node: BabylonNodeLike
  stateClips: LeaderStateClipMap
}

function isLeaderId(value: unknown): value is LeaderId {
  return typeof value === 'string' && LEADER_IDS.some(leaderId => leaderId === value)
}

function readStructuredLeaders(
  result: BabylonImportResultLike,
  expectedIds: readonly LeaderId[]
): readonly StructuredLeader[] {
  const leaders = new Map<LeaderId, StructuredLeader>()

  for (const node of new Set(allImportedNodes(result))) {
    const extras = gltfExtras(node)

    if (!('leaderId' in extras) && !('stateClips' in extras)) {
      continue
    }

    if (!isLeaderId(extras.leaderId)) {
      throw new Error('leaders GLB contains state metadata without a recognized leaderId')
    }

    if (leaders.has(extras.leaderId)) {
      throw new Error(`leaders GLB contains duplicate structured identity for ${extras.leaderId}`)
    }

    leaders.set(extras.leaderId, {
      id: extras.leaderId,
      node,
      stateClips: readLeaderStateClips(node, extras.leaderId)
    })
  }

  return expectedIds.map(leaderId => {
    const leader = leaders.get(leaderId)

    if (!leader) {
      throw new Error(`leaders GLB is missing structured identity for ${leaderId}`)
    }

    return leader
  })
}

function belongsToLeader(node: BabylonNodeLike, leaderNode: BabylonNodeLike): boolean {
  const visited = new Set<BabylonNodeLike>()
  let current: BabylonNodeLike | null | undefined = node

  while (current && !visited.has(current)) {
    if (current === leaderNode) {
      return true
    }

    visited.add(current)
    current = current.parent
  }

  return false
}

export function leaderCameraAnchor(node: BabylonNodeLike, fallback: Vec3): Vec3 {
  const matrix = (node as BabylonNodeLike & { getWorldMatrix?(): { m: readonly number[] } }).getWorldMatrix?.().m

  return matrix &&
    [matrix[12], matrix[13], matrix[14]].every(value => typeof value === 'number' && Number.isFinite(value))
    ? { x: matrix[12]!, y: matrix[13]!, z: matrix[14]! }
    : { ...fallback }
}

function retainLeaderIdentityMetadata(
  result: BabylonImportResultLike,
  leaderStateClips: Map<string, LeaderStateClipMap>,
  model: ModelManifestEntry,
  cameraAnchor: Vec3,
  expectedIds: readonly LeaderId[]
): ReadonlyMap<LeaderId, Vec3> {
  const leaders = readStructuredLeaders(result, expectedIds)
  const anchors = new Map<LeaderId, Vec3>()

  for (const leader of leaders) {
    leaderStateClips.set(leader.id, leader.stateClips)
    anchors.set(leader.id, leaderCameraAnchor(leader.node, cameraAnchor))
  }

  for (const node of new Set(allImportedNodes(result))) {
    const leader = leaders.find(entry => belongsToLeader(node, entry.node))
    const mesh = result.meshes.find(candidate => candidate === node)

    if (!leader) {
      if (mesh) {
        mesh.isPickable = false
        tagNode(mesh, {
          cameraAnchor,
          focusEntityKey: staticFocusKey('model', model.id),
          kind: 'leader-shared-surface',
          modelId: 'leaders',
          occlusionGroup: model.occlusionGroup,
          selectable: false
        })
      }

      continue
    }

    if (mesh) {
      mesh.isPickable = true
    }

    tagNode(node, {
      cameraAnchor: anchors.get(leader.id)!,
      focusEntityKey: staticFocusKey('leader', leader.id),
      kind: 'leader',
      leaderId: leader.id,
      modelId: 'leaders',
      occlusionGroup: model.occlusionGroup,
      selectable: true,
      stateClips: leader.stateClips
    })
  }

  return anchors
}

function freezeStaticResources(
  result: BabylonImportResultLike,
  materials: readonly { freeze?(): void }[],
  model: ModelManifestEntry
): void {
  if (model.requiredClips.length > 0) {
    return
  }

  for (const mesh of result.meshes) {
    mesh.freezeWorldMatrix?.()
  }

  for (const material of materials) {
    material.freeze?.()
  }
}

function setNodePosition(node: BabylonNodeLike, position: Vec3): void {
  node.position?.set(position.x, position.y, position.z)
}

function workerAnimationGroups(result: BabylonImportResultLike): ReadonlyMap<string, BabylonAnimationGroupLike> {
  const groups = new Map<string, BabylonAnimationGroupLike>()

  for (const candidate of result.animationGroups) {
    if (
      candidate &&
      typeof candidate === 'object' &&
      'name' in candidate &&
      typeof (candidate as { name?: unknown }).name === 'string'
    ) {
      const group = candidate as BabylonAnimationGroupLike
      groups.set(group.name, group)
    }
  }

  return groups
}

function deterministicWorkerVariant(key: EntityKey, variants: readonly string[]): string | undefined {
  if (variants.length === 0) {
    return undefined
  }

  let hash = 0

  for (const character of key) {
    hash = (hash * 31 + character.codePointAt(0)!) >>> 0
  }

  return variants[hash % variants.length]
}

function applyIdentityAccent(node: BabylonNodeLike, accentCode: number): void {
  const low = accentCode & 0x1ff
  const high = (accentCode >>> 9) & 0x1ff
  const rotation = (low / 512) * Math.PI * 2
  const scaleX = 0.75 + high / 1024
  node.rotation?.set(0, 0, rotation)
  node.scaling?.set(scaleX, 1.5 - scaleX, 1)
}

function validActivationScale(value: unknown): value is readonly [number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(component => typeof component === 'number' && Number.isFinite(component) && component > 0)
  )
}

export function createBabylonEntityFactory(
  model: ModelManifestEntry,
  result: BabylonImportResultLike,
  modules: LunarCityWorldModules,
  scene: ConstructorParameters<LunarCityWorldModules['TransformNode']>[1],
  characterAssets?: CharacterAssetManifest,
  reviewedNear?: BabylonImportResultLike
): EntityPresentationFactory {
  const historicalFactory = reviewedNear
    ? createBabylonEntityFactory(model, result, modules, scene, characterAssets)
    : undefined

  const template = findNode(result, `${model.id}:root`)
  const sourceGroups = workerAnimationGroups(result)
  const variants = model.instancing?.variants ?? []
  const candidateMeshes = result.meshes.filter(mesh => mesh.name !== '__root__') as readonly BabylonInstancedMeshLike[]
  const sourceLodNodes = model.lods.map(lod => findNode(result, lod.node))

  const sourceVariantNodes = new Map(
    variants.flatMap(variant => {
      const node = findNode(result, `worker:variant:${variant}`)

      return node ? [[variant, node] as const] : []
    })
  )

  const sourceGroupKitNodes = new Map(
    allImportedNodes(result).flatMap(node => {
      const match = /^worker:group-kit:([^:]+)$/u.exec(node.name)

      return match?.[1] ? [[match[1], node] as const] : []
    })
  )

  const declaredSignatureNodes = new Map(
    characterAssets
      ? [
          ...Object.entries(characterAssets.physicalVariantRoots.body).map(
            ([id, node]) => [node, `body:${id}`] as const
          ),
          ...Object.entries(characterAssets.physicalVariantRoots.head).map(
            ([id, node]) => [node, `head:${id}`] as const
          ),
          ...Object.entries(characterAssets.physicalVariantRoots.palette).map(
            ([id, node]) => [node, `palette:${id}`] as const
          )
        ]
      : []
  )

  const sourceSignatureNodes = new Map(
    allImportedNodes(result).flatMap(node => {
      const id = declaredSignatureNodes.get(node.name)

      return id ? [[id, node] as const] : []
    })
  )

  const activationScaleFor = (node: BabylonNodeLike): readonly [number, number, number] => {
    const declared = characterAssets?.physicalVariantRoots.activationScale[node.name]
    const fromExtras = gltfExtras(node).activationScale
    const value = declared ? [declared.x, declared.y, declared.z] : fromExtras

    if (!validActivationScale(value)) {
      throw new Error(`Lunar City workers GLB is missing a valid activationScale for ${node.name}`)
    }

    return value
  }

  const sourceActivationScales = new Map<BabylonNodeLike, readonly [number, number, number]>()

  for (const node of [
    ...sourceVariantNodes.values(),
    ...sourceGroupKitNodes.values(),
    ...sourceSignatureNodes.values()
  ]) {
    sourceActivationScales.set(node, activationScaleFor(node))
  }

  if (characterAssets && sourceSignatureNodes.size !== declaredSignatureNodes.size) {
    const found = new Set(allImportedNodes(result).map(node => node.name))
    const missing = [...declaredSignatureNodes.keys()].filter(node => !found.has(node))

    throw new Error(
      `Lunar City workers GLB is missing manifest-declared physical signature root: ${missing.join(', ')}`
    )
  }

  const sourceAccentNodes = new Map(
    allImportedNodes(result).flatMap(node => {
      const suffix = characterAssets?.physicalVariantRoots.groupKit.identityAccentSuffix
      const prefix = 'worker:group-kit:'
      const ending = suffix ? `:${suffix}` : ':identity-accent'

      const kitId =
        node.name.startsWith(prefix) && node.name.endsWith(ending)
          ? node.name.slice(prefix.length, -ending.length)
          : undefined

      return kitId ? [[kitId, node] as const] : []
    })
  )

  if (characterAssets) {
    const allNodes = allImportedNodes(result)
    const suffixes = characterAssets.physicalVariantRoots.groupKit

    const requiredKitNodes = characterAssets.groupKits.flatMap(kit => [
      `worker:group-kit:${kit.kitId}`,
      `worker:group-kit:${kit.kitId}:${suffixes.silhouetteSuffix}`,
      `worker:group-kit:${kit.kitId}:${suffixes.emblemSuffix}`,
      `worker:group-kit:${kit.kitId}:${suffixes.identityAccentSuffix}`
    ])

    const counts = new Map<string, number>()

    for (const node of allNodes) {
      counts.set(node.name, (counts.get(node.name) ?? 0) + 1)
    }

    const invalid = requiredKitNodes.filter(node => counts.get(node) !== 1)

    if (invalid.length > 0) {
      throw new Error(
        `Lunar City workers GLB has missing or duplicate manifest-declared kit roots: ${invalid.join(', ')}`
      )
    }
  }

  const meshesForLod = (lodIndex: number): readonly BabylonInstancedMeshLike[] => {
    const lodRoot = sourceLodNodes[lodIndex] ?? sourceLodNodes[0]
    const matching = lodRoot ? candidateMeshes.filter(mesh => belongsToLeader(mesh, lodRoot)) : []

    return matching.length > 0 ? matching : candidateMeshes
  }

  const meshesForVariant = (meshes: readonly BabylonInstancedMeshLike[], variant: string | undefined) => {
    if (!variant || !sourceVariantNodes.has(variant)) {
      return meshes
    }

    const selectedVariant = sourceVariantNodes.get(variant)!

    return meshes.filter(
      mesh =>
        ![...sourceVariantNodes.values()].some(variantNode => belongsToLeader(mesh, variantNode)) ||
        belongsToLeader(mesh, selectedVariant)
    )
  }

  const meshesForCharacterKit = (meshes: readonly BabylonInstancedMeshLike[], kitId: string | undefined) => {
    if (!kitId || !sourceGroupKitNodes.has(kitId)) {
      return meshes.filter(
        mesh => ![...sourceGroupKitNodes.values()].some(groupKitNode => belongsToLeader(mesh, groupKitNode))
      )
    }

    const selectedKit = sourceGroupKitNodes.get(kitId)!

    return meshes.filter(
      mesh =>
        ![...sourceGroupKitNodes.values()].some(groupKitNode => belongsToLeader(mesh, groupKitNode)) ||
        belongsToLeader(mesh, selectedKit)
    )
  }

  if (!template) {
    throw new Error('Lunar City workers GLB is missing its runtime root')
  }

  // The imported GLB is a template only. Every observable worker below is a
  // clone or hardware instance of its genuine low-poly meshes, never a CSS
  // substitute or a node-name animation.
  template.setEnabled?.(false)

  return {
    createAnimated(entity, declaredVariant, character): EntityVisual {
      const anchor = new modules.TransformNode(`lunar-city:entity:${entity.key}`, scene)

      const reviewTemplate =
        reviewedNear && !character?.kitId
          ? (reviewedNear.transformNodes.find(node => !node.parent || node.parent.name?.startsWith('review-leader:')) ??
            reviewedNear.meshes.find(node => node.name === '__root__'))
          : undefined

      const usingReview = Boolean(reviewTemplate)

      const historicalVisual = usingReview
        ? historicalFactory!.createAnimated(entity, declaredVariant, character)
        : undefined

      let currentLod = 0
      historicalVisual?.setLod?.(-1)

      const metricRoot = usingReview ? undefined : new modules.TransformNode(`lunar-city:worker-metric:${entity.key}`, scene)

      if (metricRoot) {metricRoot.parent = anchor}
      const metricFits = new Map<number, HistoricalWorkerFit>()

      const clone = cloneWorkerHierarchy(
        reviewTemplate ?? template,
        metricRoot ?? anchor,
        usingReview ? workerAnimationGroups(reviewedNear!) : sourceGroups,
        entity.key
      )

      const variant = declaredVariant ?? deterministicWorkerVariant(entity.key, variants)
      const metadata = workerPickMetadata(entity, model, variant, character)
      anchor.metadata = {
        ...metadataRecord(anchor.metadata),
        ...(character ? { lunarCityCharacter: character } : {}),
        lunarCityWorkerVariant: variant
      }
      tagWorkerNode(anchor, metadata)
      tagWorkerNode(clone.root, metadata)

      for (const node of new Set(clone.nodeMap.values())) {
        tagWorkerNode(node, metadata)
      }

      clone.root.setEnabled?.(true)

      for (const [variantId, sourceNode] of sourceVariantNodes) {
        const cloneNode = clone.nodeMap.get(sourceNode)
        const scale = sourceActivationScales.get(sourceNode)

        if (cloneNode && scale) {
          cloneNode.scaling?.set(...(variantId === variant ? scale : ([0, 0, 0] as const)))
          cloneNode.setEnabled?.(variantId === variant)
        }
      }

      for (const [kitId, sourceNode] of sourceGroupKitNodes) {
        const cloneNode = clone.nodeMap.get(sourceNode)
        const scale = sourceActivationScales.get(sourceNode)

        if (cloneNode && scale) {
          cloneNode.scaling?.set(...(kitId === character?.kitId ? scale : ([0, 0, 0] as const)))
          cloneNode.setEnabled?.(kitId === character?.kitId)
        }
      }

      const enabledSignatureNodes = new Set(
        character?.signature
          ? [
              `body:${character.signature.body}`,
              `head:${character.signature.head}`,
              `palette:${character.signature.palette}`
            ]
          : []
      )

      for (const [signatureId, sourceNode] of sourceSignatureNodes) {
        const cloneNode = clone.nodeMap.get(sourceNode)
        const scale = sourceActivationScales.get(sourceNode)

        if (cloneNode && scale) {
          const active = enabledSignatureNodes.has(signatureId)
          cloneNode.scaling?.set(...(active ? scale : ([0, 0, 0] as const)))
          cloneNode.setEnabled?.(active)
        }
      }

      for (const [kitId, sourceNode] of sourceAccentNodes) {
        const accent = clone.nodeMap.get(sourceNode)
        const enabled = kitId === character?.kitId
        accent?.setEnabled?.(enabled)

        if (enabled && character?.accentCode !== undefined && accent) {
          applyIdentityAccent(accent, character.accentCode)
        }
      }

      if (!character?.kitId && character?.signature && character.accentCode !== undefined) {
        const neutralAccentSource = sourceSignatureNodes.get(`palette:${character.signature.palette}`)
        const neutralAccent = neutralAccentSource ? clone.nodeMap.get(neutralAccentSource) : undefined

        if (neutralAccent) {
          applyIdentityAccent(neutralAccent, character.accentCode)
        }
      }

      let active: BabylonAnimationGroupLike | undefined

      return {
        dispose() {
          historicalVisual?.dispose?.()
          active?.stop?.()
          active = undefined

          for (const group of clone.animations.values()) {
            group.stop?.()
            group.dispose?.()
          }

          clone.disposeSkeletons()
          clone.root.dispose?.()
          anchor.dispose?.()
        },
        setAnimation(clip) {
          if (usingReview && currentLod !== 0) {
            active?.stop?.()
            active = undefined
            historicalVisual?.setAnimation?.(clip)

            return
          }

          historicalVisual?.setStaticPose?.('idle')
          const next = clone.animations.get(clip)

          if (!next) {
            active?.stop?.()
            active = undefined
            clone.animations.get('idle')?.reset?.()

            return
          }

          if (next.hasMotion === false) {
            active?.stop?.()
            next.reset?.()
            next.stop?.()
            active = undefined

            return
          }

          const loop = ['idle','walk','work','talk','listen','think','queue','wait','heartbeat','rest'].includes(clip)

          if (active !== next) {
            active?.stop?.()
            active = next
            active.start?.(loop)
          } else if (loop && active.isPlaying !== true) {active.start?.(true)}
        },
        setLod(lodIndex) {
          if (usingReview) {
            currentLod = lodIndex
            clone.root.setEnabled?.(lodIndex === 0)
            historicalVisual?.setLod?.(lodIndex === 0 ? -1 : lodIndex)

            return
          }

          for (const [index, sourceNode] of sourceLodNodes.entries()) {
            const node = sourceNode ? clone.nodeMap.get(sourceNode) : undefined
            node?.setEnabled?.(index === lodIndex)
          }

          if (metricRoot && lodIndex >= 0 && (currentLod !== lodIndex || !metricFits.has(lodIndex))) {
            const fit = metricFits.get(lodIndex)

            if (fit) {applyHistoricalWorkerFit(metricRoot, fit)}
            else {
              const measured = fitHistoricalWorker(metricRoot, [...new Set(clone.nodeMap.values())])

              if (measured) {metricFits.set(lodIndex, measured)}
            }
          }

          currentLod = lodIndex
        },
        isAnimationActive() { return active?.isPlaying === true || (historicalVisual?.isAnimationActive?.() ?? false) },
        supportsAnimation(clip) { return clone.animations.has(clip) && clone.animations.get(clip)?.hasMotion !== false },
        setFacing(yaw) {
          anchor.rotation?.set(0, yaw, 0)
          historicalVisual?.setFacing?.(yaw)
        },
        setPosition(position) {
          historicalVisual?.setPosition?.(position)
          setNodePosition(anchor, position)
        },
        setStaticPose() {
          historicalVisual?.setStaticPose?.('idle')
          active?.stop?.()
          active = undefined
        }
      }
    },
    createInstancedGroup(groupKey): InstancedEntityGroup {
      const lodIndex = Number(/:lod:(\d+)$/u.exec(groupKey)?.[1] ?? 0)
      const variant = variants.find(candidate => groupKey.startsWith(`worker:${candidate}:`))
      const kitId = /:kit:([^:]+)(?::lod:\d+)?$/u.exec(groupKey)?.[1]
      const sourceMeshes = meshesForCharacterKit(meshesForVariant(meshesForLod(lodIndex), variant), kitId)
      const kitNode = kitId ? sourceGroupKitNodes.get(kitId) : undefined
      const accentNode = kitId ? sourceAccentNodes.get(kitId) : undefined

      const midKitMeshes =
        lodIndex === 1 && kitNode ? candidateMeshes.filter(mesh => belongsToLeader(mesh, kitNode)) : []

      const instanceSources = [
        ...sourceMeshes.map(mesh => ({
          accent: !kitId && lodIndex === 1 && /(?:^|:)face$/u.test(mesh.name),
          mesh
        })),
        ...midKitMeshes
          .filter(mesh => !sourceMeshes.includes(mesh))
          .map(mesh => ({ accent: accentNode ? belongsToLeader(mesh, accentNode) : false, mesh }))
      ]

      // Every member in this group shares the same selected source geometry and LOD.
      let instanceMetricFit: HistoricalWorkerFit | undefined

      const members = new Map<
        EntityKey,
        { instances: { accent: boolean; node: BabylonNodeLike }[]; root: BabylonNodeLike }
      >()

      const disposeMember = (member: {
        instances: { accent: boolean; node: BabylonNodeLike }[]
        root: BabylonNodeLike
      }): void => {
        for (const instance of member.instances) {
          instance.node.dispose?.()
        }

        member.root.dispose?.()
      }

      return {
        dispose() {
          for (const member of members.values()) {
            disposeMember(member)
          }

          members.clear()
        },
        sync(nextMembers: readonly InstancedEntityMember[]) {
          const nextKeys = new Set(nextMembers.map(member => member.key))

          for (const [key, member] of members) {
            if (!nextKeys.has(key)) {
              disposeMember(member)
              members.delete(key)
            }
          }

          for (const member of nextMembers) {
            let visual = members.get(member.key)

            if (!visual) {
              const root = new modules.TransformNode(`lunar-city:instance:${groupKey}:${member.key}`, scene)

              const metricRoot = new modules.TransformNode(`lunar-city:instance-metric:${member.key}`, scene)
              metricRoot.parent = root

              const instances = instanceSources.flatMap(({ accent, mesh }) => {
                const instance = mesh.createInstance?.(`lunar-city:instance-mesh:${member.key}:${mesh.name}`)

                if (instance) {
                  instance.parent = metricRoot

                  return [{ accent, node: instance }]
                }

                return []
              })

              if (instanceMetricFit) {applyHistoricalWorkerFit(metricRoot, instanceMetricFit)}
              else {instanceMetricFit = fitHistoricalWorker(metricRoot, instances.map(instance => instance.node))}

              visual = { instances, root }
              members.set(member.key, visual)
            }

            const metadata = workerPickMetadata(member, model, member.variant ?? variant, member.character)
            tagWorkerNode(visual.root, metadata)

            for (const instance of visual.instances) {
              tagWorkerNode(instance.node, metadata)

              if (instance.accent && member.character?.accentCode !== undefined) {
                applyIdentityAccent(instance.node, member.character.accentCode)
              }
            }

            setNodePosition(visual.root, member.position)
          }
        }
      }
    }
  }
}

function intersectsBounds(start: Vec3, end: Vec3, bounds: WorldBounds): boolean {
  let minimum = 0
  let maximum = 1

  for (const axis of ['x', 'y', 'z'] as const) {
    const origin = start[axis]
    const direction = end[axis] - origin
    const lower = bounds.min[axis]
    const upper = bounds.max[axis]

    if (direction === 0) {
      if (origin < lower || origin > upper) {
        return false
      }

      continue
    }

    const first = (lower - origin) / direction
    const second = (upper - origin) / direction
    minimum = Math.max(minimum, Math.min(first, second))
    maximum = Math.min(maximum, Math.max(first, second))

    if (minimum > maximum) {
      return false
    }
  }

  return true
}

interface MaterialAdapter {
  readonly source: NonNullable<BabylonMeshLike['material']>
  alpha?: number
  clone?(): MaterialAdapter | undefined
}

function materialAdapter(source: NonNullable<BabylonMeshLike['material']>): MaterialAdapter {
  return {
    source,
    get alpha() {
      return source.alpha
    },
    set alpha(value: number | undefined) {
      source.alpha = value
    },
    clone() {
      const cloned = source.clone?.('lunar-city:occlusion-isolated')

      return cloned && typeof cloned === 'object'
        ? materialAdapter(cloned as NonNullable<BabylonMeshLike['material']>)
        : undefined
    }
  }
}

function buildOcclusionCandidates(
  result: BabylonImportResultLike,
  model: ModelManifestEntry
): readonly OcclusionCandidate[] {
  if (!/(?:roof|wall)/iu.test(model.occlusionGroup)) {
    return []
  }

  const bounds = worldBoundsFromModel(model)

  return result.meshes.flatMap(mesh => {
    if (!mesh.material?.clone) {
      return []
    }

    const material = materialAdapter(mesh.material)

    return [
      {
        group: model.occlusionGroup,
        material,
        isolateMaterial: true,
        assignMaterial(isolated) {
          mesh.material = (isolated as MaterialAdapter).source
        },
        intersectsFocusRay(camera, selection) {
          return intersectsBounds(camera.position, selection.cameraAnchor, bounds)
        }
      }
    ]
  })
}

function cameraPosition(camera: CameraLike): Vec3 {
  const positional = camera as CameraLike & { globalPosition?: Vec3; position?: Vec3 }
  const position = positional.globalPosition ?? positional.position

  if (position && Number.isFinite(position.x) && Number.isFinite(position.y) && Number.isFinite(position.z)) {
    return { x: position.x, y: position.y, z: position.z }
  }

  return { x: camera.target.x, y: camera.target.y, z: camera.target.z }
}

function pickedCameraTarget(node: BabylonNodeLike | undefined): CameraPickTarget | undefined {
  const visited = new Set<BabylonNodeLike>()
  let current = node

  while (current && !visited.has(current)) {
    visited.add(current)
    const metadata = current.metadata?.lunarCity

    if (metadata && 'selectable' in metadata && metadata.selectable && 'focusEntityKey' in metadata) {
      return { kind: 'entity', entityKey: metadata.focusEntityKey }
    }

    current = current.parent ?? undefined
  }

  return undefined
}

export async function createWorldScene(
  engine: ConstructorParameters<LunarCityWorldModules['Scene']>[0],
  manifest: WorldManifestV2,
  emit: (intent: LunarCityIntent) => void,
  modules: LunarCityWorldModules,
  resolveAssetUrl: (uri: string) => string
): Promise<LunarCitySceneHandle> {
  const scene = new modules.Scene(engine)
  // Manifest placements and authored glTF terrain share a right-handed metre frame.
  scene.useRightHandedSystem = true
  const capturePerf = typeof window !== 'undefined' && window.__LUNAR_CITY_PERF__ !== undefined

  const perfAdapter = createBabylonPerfAdapter(
    scene,
    capturePerf && modules.SceneInstrumentation ? new modules.SceneInstrumentation(scene) : undefined
  )

  let lastFrameMs = 0
  let lastWorldUpdateMs = 0
  const worldUpdateTimestampsMs: number[] | undefined = capturePerf ? [] : undefined

  let atmosphere: ReturnType<NonNullable<LunarCityWorldModules['createAtmosphere']>> | undefined
  let interiorCutaway: ReturnType<NonNullable<LunarCityWorldModules['createInteriorCutaway']>> | undefined
  let interiorBuildingId: string | undefined
  let disposed = false
  let scheduler: ReturnType<typeof createFrameScheduler> | undefined
  let navigation: ReturnType<typeof createNavigationController> | undefined
  let entityRegistry: ReturnType<typeof createEntityRegistry> | undefined
  let occlusion: ReturnType<typeof createOcclusionController> | undefined
  let glowLayer: BabylonGlowLayerLike | undefined
  let shadowGenerator: BabylonShadowGeneratorLike | undefined
  const projectCompoundNodes = new Map<string, BabylonNodeLike>()
  const buildingActivity = createBuildingActivity()
  const characterAnimations = createCharacterAnimations(() => scheduler?.requestRender())
  const leaderNodes = new Map<LeaderId, BabylonNodeLike>()
  const leaderLife = createLeaderLife()
  const leaderOverrides = new Set<LeaderId>()
  let observedModels = new Set<LeaderId>()
  let leaderSnapshot: LunarCitySnapshot | undefined

  const updateObservedLeaderLife = (snapshot: LunarCitySnapshot): void => {
    leaderSnapshot = snapshot
    const projections = observedLeaderLife(snapshot, manifest.characterAssets.leaders.map(leader => leader.id))
    const current = new Set(projections.map(projection => projection.id))

    for (const id of observedModels) {
      if (!current.has(id) && !leaderOverrides.has(id)) {leaderLife.setMode(id, 'unavailable')}
    }

    for (const projection of projections) {
      if (!leaderOverrides.has(projection.id)) {leaderLife.setMode(projection.id, projection.mode)}
    }

    observedModels = current
  }

  const leaderQueries = new Map<string, NavigationQuery>()
  const leaderBindings: { asset: ReviewLeaderAsset; root: BabylonNodeLike; anchor: Vec3; canWalk: boolean; envelope?: {radius: number; height: number} }[] = []

  const disposeWorld = (): void => {
    if (disposed) {
      return
    }

    disposed = true
    interiorCutaway?.dispose()
    atmosphere?.dispose()
    buildingActivity.dispose()
    leaderLife.dispose()

    for (const query of leaderQueries.values()) {query.dispose?.()}
    characterAnimations.dispose()
    scheduler?.dispose()
    entityRegistry?.dispose()
    occlusion?.clear()
    navigation?.dispose()

    for (const node of projectCompoundNodes.values()) {
      node.dispose?.()
    }

    projectCompoundNodes.clear()
    glowLayer?.dispose?.()
    shadowGenerator?.dispose?.()
    perfAdapter.dispose()
    scene.dispose()
  }

  try {
    // The authored rooms use charcoal interiors with cyan/violet accents. A
    // warmer ambient floor keeps those PBR surfaces legible when the efficient
    // tier disables dynamic shadows, without adding another per-frame light.
    if (modules.Color4) {
      scene.clearColor = new modules.Color4(0.045, 0.028, 0.065, 1)
    }

    scene.ambientColor = new modules.Color3(0.46, 0.32, 0.28)
    const overview = manifest.camera.overview
    scene.activeCamera = new modules.ArcRotateCamera(
      'lunar-city:approved-overview',
      overview.alpha,
      overview.beta,
      overview.radius,
      new modules.Vector3(overview.target.x, overview.target.y, overview.target.z),
      scene
    )

    const keyLight = new modules.DirectionalLight('lunar-city:key-light', new modules.Vector3(-0.45, -1, 0.35), scene)

    keyLight.intensity = 1.5
    keyLight.diffuse = new modules.Color3(1, 0.86, 0.68)

    // A cool back-left rim light carves the approved silhouettes away from the
    // charcoal interiors, the same StarCraft-style two-tone read as the
    // reference art, without a second shadow map: it never toggles
    // shadowEnabled and stays on across every quality tier.
    const rimLight = new modules.DirectionalLight('lunar-city:rim-light', new modules.Vector3(0.6, -0.25, -0.55), scene)

    rimLight.intensity = 0.4
    rimLight.diffuse = new modules.Color3(0.55, 0.72, 0.95)
    rimLight.shadowEnabled = false

    // A soft top-down/ground-bounce fill keeps unlit faces out of pure black
    // once the key/rim pair adds contrast. HemisphericLight has no shadow map
    // and no specular term here, so it costs one extra ambient term per pixel.
    const fillLight = modules.HemisphericLight
      ? new modules.HemisphericLight('lunar-city:fill-light', new modules.Vector3(0, 1, 0), scene)
      : undefined

    if (fillLight) {
      fillLight.intensity = 0.65
      fillLight.diffuse = new modules.Color3(0.55, 0.64, 0.82)
      fillLight.groundColor = new modules.Color3(0.34, 0.24, 0.22)
      fillLight.specular = new modules.Color3(0, 0, 0)
      fillLight.shadowEnabled = false
    }

    // A low-resolution glow pass blooms the authored emissive materials
    // (signal-emissive, archive-emissive, the amber/green accent glows) so
    // they read as lit signage instead of flat colored panels. It rides the
    // existing `decorations` quality flag so the efficient tier's most
    // aggressive degradation step turns it off with everything else.
    glowLayer = modules.GlowLayer
      ? new modules.GlowLayer('lunar-city:glow', scene, { mainTextureRatio: 0.5 })
      : undefined

    if (glowLayer) {
      glowLayer.intensity = 0.42
    }

    // Distance haze. Without it the settlement reads as a diorama floating in
    // a void: every district is equally crisp, so the eye gets no depth cue
    // and the far rim of the crater sits visually on top of the near one.
    // Linear fog over the camera's own bounded zoom range costs nothing —
    // it is a per-pixel lerp the fixed-function path already runs.
    scene.fogMode = FOG_MODE_LINEAR
    scene.fogColor = new modules.Color3(0.09, 0.06, 0.08)
    scene.fogStart = overview.maxRadius * 0.7
    scene.fogEnd = overview.maxRadius * 2.4

    const imageProcessing = scene.imageProcessingConfiguration

    if (imageProcessing) {
      // ACES filmic tonemapping. The authored palette drives emissive values
      // above 1.0 for the signage; without a tonemapper those clip to flat
      // white and the whole image reads as untonemapped sRGB — the single
      // clearest "engine default" tell. ACES rolls the highlights off and is
      // the same curve the reference games grade through.
      imageProcessing.toneMappingEnabled = true
      imageProcessing.toneMappingType = TONE_MAPPING_ACES
      imageProcessing.contrast = 1.12
      imageProcessing.exposure = 1.25
      imageProcessing.vignetteEnabled = true
      imageProcessing.vignetteWeight = 1.6
      imageProcessing.vignetteColor = new modules.Color3(0.04, 0.02, 0.04)
    }

    // The quality tiers have always declared `dynamicShadows: 'near'`, but no
    // shadow generator existed, so the flag toggled nothing and every district
    // floated with no contact against the terrain. One soft-filtered map on
    // the key light grounds the whole settlement; it renders only while
    // `keyLight.shadowEnabled` is true, which the efficient tier keeps off.
    shadowGenerator = modules.ShadowGenerator ? new modules.ShadowGenerator(SHADOW_MAP_SIZE, keyLight) : undefined

    if (shadowGenerator) {
      shadowGenerator.usePercentageCloserFiltering = true
      shadowGenerator.filteringQuality = 0
      shadowGenerator.darkness = 0.42
      shadowGenerator.bias = 0.0018
      shadowGenerator.normalBias = 0.012
      shadowGenerator.transparencyShadow = false
    }

    atmosphere = modules.createAtmosphere?.(scene)

    const leaderStateClips = new Map<string, LeaderStateClipMap>()
    const reviewWorkerClips = new Map<string, Readonly<Record<string, string>>>()
    const camera = scene.activeCamera as CameraLike
    const focusAnchors = new Map<EntityKey, () => Vec3 | undefined>()
    const focusMetadata = new Map<EntityKey, () => EntityFocusMetadata | undefined>()
    const occlusionCandidates: OcclusionCandidate[] = []
    const decorationNodes: BabylonNodeLike[] = []
    const staticLods: Array<{ focus: FocusMetadata; lods: readonly LodEntry[] }> = []
    const exteriorMeshes = new Map<string, readonly BabylonMeshLike[]>()
    interiorCutaway = modules.createInteriorCutaway?.(scene, id =>
      (exteriorMeshes.get(id) ?? []).filter((mesh): mesh is BabylonMeshLike & { visibility: number } =>
        typeof (mesh as { visibility?: number }).visibility === 'number'))
    let reviewedNearWorker: BabylonImportResultLike | undefined
    let workerAsset: { model: ModelManifestEntry; result: BabylonImportResultLike } | undefined

    const cameraController: CameraController = createCameraController(camera, overview, manifest.camera.bounds, {
      focusAnchors,
      focusBeta: key =>
        manifest.reviewLeaderAssets?.some(asset => key === `lunar-city:leader:${asset.id}`) ||
        manifest.reviewWorkerAssets?.some(asset => key === `lunar-city:review-worker:${asset.id}`)
          ? 1.2
          : undefined,
      focusRadius: key => {
        const asset = manifest.reviewLeaderAssets?.find(entry => staticFocusKey('leader', entry.id) === key)

        const authoredHeight = manifest.characterAssets.leaders.find(
          entry => staticFocusKey('leader', entry.id) === key
        )?.heightMetres

        const workerHeight = manifest.reviewWorkerAssets?.find(
          entry => `lunar-city:review-worker:${entry.id}` === key
        )?.heightMetres

        const height = asset?.heightMetres ?? authoredHeight ?? workerHeight

        if (height) {
          return height * 3.5
        }

        const model = manifest.models.find(entry => staticFocusKey('model', entry.id) === key)

        return model && model.id !== 'terrain'
          ? Math.max(
              8,
              ...(['x', 'y', 'z'] as const).map(
                axis => (model.bounds.max[axis] - model.bounds.min[axis]) * model.transform.scale[axis] * (model.uri.includes('review-building-') ? 2.2 : 1.6)
              )
            )
          : undefined
      },
      followOffset: manifest.camera.followOffset
    })

    for (const model of manifest.models) {
      if (model.id === 'leaders' && manifest.reviewLeaderAssets?.length) {
        continue
      }

      const materialStart = scene.materials?.length ?? 0
      const result = await modules.ImportMeshAsync(resolveAssetUrl(model.uri), scene)
      exteriorMeshes.set(model.id, result.meshes)

      if (!['workers', 'leaders'].includes(model.id)) {buildingActivity.register(model.id, result.animationGroups)}
      const placed = placeModel(result, model, modules, scene)
      const focus = placed.focus

      focusAnchors.set(focus.focusEntityKey, () => focus.cameraAnchor)
      focusMetadata.set(focus.focusEntityKey, () => focus)
      staticLods.push({ focus, lods: placed.lods })
      occlusionCandidates.push(...buildOcclusionCandidates(result, model))
      decorationNodes.push(...allImportedNodes(result).filter(isDecorationNode))

      if (shadowGenerator) {
        const renderList = shadowGenerator.getShadowMap?.()?.renderList

        for (const mesh of result.meshes) {
          // Every surface receives; only real geometry casts. The terrain is
          // the ground plane the settlement sits on, so it receives without
          // casting a redundant shadow onto itself.
          mesh.receiveShadows = true

          if (model.id !== 'terrain' && (mesh.getTotalVertices?.() ?? 0) > 0) {
            renderList?.push(mesh)
          }
        }
      }

      if (model.id === 'leaders') {
        const leaderIds = manifest.characterAssets.leaders.map(leader => leader.id)

        const leaderCameraAnchors = retainLeaderIdentityMetadata(
          result,
          leaderStateClips,
          model,
          focus.cameraAnchor,
          leaderIds
        )

        for (const leader of readStructuredLeaders(result, leaderIds)) {
          leaderNodes.set(leader.id, leader.node)
        }

        const importedAnimationGroups = workerAnimationGroups(result)

        for (const leaderId of leaderIds) {
          const stateClips = leaderStateClips.get(leaderId)
          const groups = new Map<LeaderAnimationState, BabylonAnimationGroupLike>()

          if (stateClips) {
            for (const state of LEADER_STATES) {
              const clip = stateClips[state]
              const group = clip ? importedAnimationGroups.get(clip) : undefined

              if (group) {
                groups.set(state, group)
              }
            }
          }

          characterAnimations.register(leaderId, groups, CONTINUOUS_LEADER_STATES)
        }

        for (const leaderId of leaderIds) {
          const leaderFocusKey = staticFocusKey('leader', leaderId)
          const cameraAnchor = leaderCameraAnchors.get(leaderId) ?? focus.cameraAnchor
          focusAnchors.set(leaderFocusKey, () => cameraAnchor)
          focusMetadata.set(leaderFocusKey, () => ({ ...focus, cameraAnchor, focusEntityKey: leaderFocusKey }))
        }
      }

      if (model.id === 'workers') {
        workerAsset = { model, result }
      }

      freezeStaticResources(result, scene.materials?.slice(materialStart) ?? [], model)
    }

    for (const asset of manifest.reviewLeaderAssets ?? []) {
      const { anchor, focusEntityKey, root, stateClips, groups, result } = await importReviewLeader(
        asset,
        scene,
        modules,
        resolveAssetUrl,
        shadowGenerator
      )

      let radialExtent = 0

      const bounds = result.meshes.flatMap(mesh => {
        const measured = mesh as typeof mesh & {
          computeWorldMatrix?(force: boolean): unknown
          getPositionData?(applySkeleton: boolean, applyMorph: boolean): ArrayLike<number> | null
          getBoundingInfo?(): { boundingBox: {minimumWorld: Vec3; maximumWorld: Vec3} }
        }

        if ((mesh.getTotalVertices?.() ?? 0) === 0 || !measured.getBoundingInfo) {return []}
        measured.computeWorldMatrix?.(true)
        const vertices = measured.getPositionData?.(true, true) ?? mesh.getVerticesData?.('position')
        const matrix = mesh.getWorldMatrix?.().m

        if (vertices && matrix) {
          for (let offset = 0; offset < vertices.length; offset += 3) {
            const x = matrix[0]! * vertices[offset]! + matrix[4]! * vertices[offset + 1]! + matrix[8]! * vertices[offset + 2]! + matrix[12]! - asset.position.x
            const z = matrix[2]! * vertices[offset]! + matrix[6]! * vertices[offset + 1]! + matrix[10]! * vertices[offset + 2]! + matrix[14]! - asset.position.z
            radialExtent = Math.max(radialExtent, Math.hypot(x, z))
          }
        }

        const box = measured.getBoundingInfo().boundingBox

        return [{ minimum: box.minimumWorld, maximum: box.maximumWorld }]
      })

      leaderBindings.push({ asset, root, anchor, canWalk: groups.get('walk')?.hasMotion === true, envelope: measuredLeaderEnvelope(bounds, asset.heightMetres, radialExtent || undefined) })
      leaderStateClips.set(asset.id, stateClips)
      characterAnimations.register(asset.id, groups, CONTINUOUS_LEADER_STATES)
      leaderNodes.set(asset.id, root)
      focusAnchors.set(focusEntityKey, () => anchor)
      focusMetadata.set(focusEntityKey, () => ({ cameraAnchor: anchor, focusEntityKey, occlusionGroup: 'leaders' }))
    }

    const workerStates = [
      'idle',
      'walk',
      'talk',
      'listen',
      'work',
      'tool-use',
      'carry',
      'handoff',
      'queue',
      'wait',
      'blocked',
      'failed',
      'review',
      'triage',
      'heartbeat',
      'rest',
      'done'
    ]

    const continuousWorkerStates = new Set(
      workerStates.filter(state => !['handoff', 'blocked', 'failed', 'done'].includes(state))
    )

    for (const asset of manifest.reviewWorkerAssets ?? []) {
      const {
        anchor,
        focusEntityKey,
        stateClips,
        groups,
        result: importedWorker
      } = await importReviewLeader(
        asset,
        scene,
        modules,
        resolveAssetUrl,
        shadowGenerator,
        'review-worker',
        workerStates
      )

      if (
        asset.id === 'baseline' &&
        typeof window !== 'undefined' &&
        new URLSearchParams(window.location.search).get('lunarCityReviewWorkers') === 'baseline'
      ) {
        reviewedNearWorker = importedWorker
      }

      reviewWorkerClips.set(asset.id, stateClips as Readonly<Record<string, string>>)
      characterAnimations.register(`worker:${asset.id}`, groups, continuousWorkerStates)
      focusAnchors.set(focusEntityKey, () => anchor)
    }

    for (const decoration of manifest.externalDecorations ?? []) {
      const result = await modules.ImportMeshAsync(resolveAssetUrl(decoration.uri), scene)
      const root = new modules.TransformNode(`decoration:${decoration.id}`, scene)
      root.position?.set(decoration.position.x, decoration.position.y, decoration.position.z)
      root.rotation?.set(decoration.rotation.x, decoration.rotation.y, decoration.rotation.z)
      root.scaling?.set(decoration.scale.x, decoration.scale.y, decoration.scale.z)

      for (const node of allImportedNodes(result)) {
        if (!node.parent) {
          node.parent = root
        }
      }

      for (const mesh of result.meshes) {
        mesh.isPickable = false
      }

      decorationNodes.push(root)
    }

    const occlusionController = createOcclusionController(occlusionCandidates)
    occlusion = occlusionController
    const quality = createQualityController('efficient')

    const applyRuntimeQuality = (): void => {
      const settings = quality.settings()
      buildingActivity.setEnabled(settings.decorations)

      applyQualitySettings(engine, settings)
      // Continuous idle camera motion is reserved for balanced/detailed
      // presentation. Efficient mode must be able to park the scheduler when
      // the world has no real activity, which keeps the low-power path honest.
      cameraController.setIdleEnabled(settings.tier !== 'efficient')
      keyLight.shadowEnabled = settings.dynamicShadows === 'near'

      if (glowLayer) {
        glowLayer.intensity = settings.decorations ? 0.42 : 0
      }

      for (const node of decorationNodes) {
        node.setEnabled?.(settings.decorations)
      }
    }

    applyRuntimeQuality()

    if (!workerAsset) {
      throw new Error('Lunar City manifest has no workers model')
    }

    const workerClipNames = new Set(workerAnimationGroups(workerAsset.result).keys())

    const navigationQuery = await createRouteNavigationQuery(manifest.navigation, modules, scene, resolveAssetUrl, modules.interiorPlans)

    const leaderPositionState = createEntityPositionState({
      scope: `leaders:${resolveAssetUrl(manifest.navigation.meshUri)}:${manifest.navigation.sha256 ?? manifest.source.sha256}:${JSON.stringify([manifest.navigation.colliders, modules.interiorPlans, manifest.reviewLeaderAssets])}`,
      maxEntries: 8
    })

    const leaderHomes: Partial<Record<LeaderId, string>> = {
      owl: 'library', fox: 'research-lab', elephant: 'archive', cat: 'arts-studio',
      capybara: 'revenue', lion: 'council', beaver: 'engineering-workshop', monkey: 'publishing'
    }

    for (const { asset, root, anchor, canWalk, envelope } of leaderBindings) {
      const plan = modules.interiorPlans?.find(entry => entry.id === leaderHomes[asset.id] && entry.runtimePortalStatus === 'authored_approach')

      if (!plan?.worldAnchors) {continue}
      const anchors = plan.worldAnchors as Record<string, readonly number[]>

      const point = (name: string): Vec3 | undefined => {
        const value = anchors[name]

        return value?.length === 3 && value.every(Number.isFinite) ? { x: value[0], y: value[1], z: value[2] } : undefined
      }

      const home = point('home')
      const work = point('leaderWork') ?? point('work')

      if (!home || !work || !envelope) {continue}
      const envelopeKey = `${envelope.radius}:${envelope.height}`

      let query = canWalk ? leaderQueries.get(envelopeKey) : createInteriorNavigation(
        {computePath: () => undefined}, {positions: new Float32Array(), indices: new Uint32Array()},
        manifest.navigation.colliders ?? [], [plan], envelope
      )

      if (!query) {
        query = await createRouteNavigationQuery(manifest.navigation, modules, scene, resolveAssetUrl, modules.interiorPlans, envelope)
        leaderQueries.set(envelopeKey, query)
      }

      leaderLife.register({
        id: asset.id, position: home, home, work,
        strollPoints: [...(canWalk ? safeLeaderStrollPoints(query, home, Object.values(manifest.destinations)) : []), ...[point('rest'), point('entry')].filter((entry): entry is Vec3 => !!entry)],
        canWalk, query, positionState: leaderPositionState,
        setPosition(position) {
          root.position?.set(position.x, position.y, position.z)
          Object.assign(anchor, position, { y: position.y + asset.heightMetres * .55 })
        },
        setFacing: yaw => root.rotation?.set(0, yaw, 0),
        setTraveling: moving => characterAnimations.set(asset.id, moving ? 'walk' : 'idle')
      })
    }



    const entityRegistryController = createEntityRegistry({
      positionState: createEntityPositionState({ scope: `${resolveAssetUrl(manifest.navigation.meshUri)}:${manifest.navigation.sha256 ?? manifest.source.sha256}:${JSON.stringify(manifest.navigation.colliders ?? [])}:${JSON.stringify(modules.interiorPlans ?? [])}` }),
      resolvePosition: ({ candidate, previous, entity, reason }) => {
        if (!navigationQuery.resolvePosition) {return candidate}

        // Observation coordinates are source provenance, not teleport permission.
        if (previous && reason === 'correction') {return navigationQuery.resolvePosition(previous)}
        const grounded = navigationQuery.resolvePosition(candidate)

        if (grounded) {return grounded}

        if (reason === 'spawn') {
          const anchor = manifest.destinations[entity.destination]

          return anchor ? navigationQuery.resolvePosition(anchor) : undefined
        }

        return undefined
      },
      characterAssets: manifest.characterAssets,
      factory: createBabylonEntityFactory(
        workerAsset.model,
        workerAsset.result,
        modules,
        scene,
        manifest.characterAssets,
        reviewedNearWorker
      ),
      focusAnchors,
      focusMetadata,
      workerClips: workerClipNames
    })

    entityRegistry = entityRegistryController

    let projectTargets: ReadonlyMap<EntityKey, Vec3> = new Map()
    const arrivalSlots = createArrivalSlots(manifest)

    const navigationController = createNavigationController({
      targetFor: (key, destination, origin) => {
        const role = ({ lab: 'research-lab', review: 'review-office' } as Record<string,string>)[destination] ?? destination
        const plan = modules.interiorPlans?.find(candidate => candidate.id === role && candidate.runtimePortalStatus === 'authored_approach')
        const pickup = plan?.worldAnchors.jobPickup
        const target = pickup ? { x: pickup[0]!, y: pickup[1]!, z: pickup[2]! } : undefined

        // This is the presentation destination of an already assigned worker.
        // It never creates or claims a backend assignment.
        return target && navigationQuery.resolvePosition?.(target) ? target : arrivalSlots.target(key, destination, destination === 'project' ? (projectTargets.get(key) ?? origin) : origin)
      },
      destinations: manifest.destinations,
      query: navigationQuery,
      workerClips: workerClipNames
    })

    navigation = navigationController

    const projectSites = createProjectSiteRuntime(manifest.projectSlots, projectCompoundNodes, anchor => {
      const node = modules.createProjectMarker?.(`lunar-city:compound:${anchor.key}`, scene) ??
        new modules.TransformNode(`lunar-city:compound:${anchor.key}`, scene)

      setNodePosition(node, anchor.position)
      tagNode(node, { connectionId:anchor.connectionId, key:anchor.key, kind:'project-compound', projectId:anchor.projectId, selectable:false })

      return node
    }, revision => navigationController.setWalkabilityRevision(revision))

    const reconcileProjectCompounds = (snapshot: LunarCitySnapshot): void => {
      projectTargets = projectSites.update(snapshot).entityTargets
    }

    const destinationByEntity = new Map<EntityKey, DestinationId>()
    const authoritativeOriginByEntity = new Map<EntityKey, Vec3>()
    const activeNavigation = new Set<EntityKey>()
    let currentEntityKeys: readonly EntityKey[] = []
    let reducedMotion = false
    const social = createWorkerSocialRuntime(entityRegistryController, () => !reducedMotion && quality.settings().decorations)

    let lastWorkerSnapshot: LunarCitySnapshot | undefined

    const reconcileWorkers = (snapshot: LunarCitySnapshot): void => {
      const dynamicEntities = new Map(
        [...snapshot.entities].filter(([, entity]) => !(entity.identity.kind === 'kanban' && !entity.position))
      )

      currentEntityKeys = [...dynamicEntities.values()]
        .filter(
          entity =>
            entity.position !== undefined &&
            (entity.identity.kind !== 'profile' || entity.presentation?.placement.slot !== undefined)
        )
        .map(entity => entity.key)

      for (const key of destinationByEntity.keys()) {
        if (!dynamicEntities.has(key)) {
          navigationController.cancel(key)
          destinationByEntity.delete(key)
          authoritativeOriginByEntity.delete(key)
          activeNavigation.delete(key)
        }
      }

      entityRegistryController.reconcile({ ...snapshot, entities: dynamicEntities })

      for (const [key, entity] of dynamicEntities) {
        const previousDestination = destinationByEntity.get(key)
        const previousOrigin = authoritativeOriginByEntity.get(key)
        const authoritativeOrigin = entity.authority === 'authoritative' ? entity.position : undefined

        const hasOriginCorrection =
          authoritativeOrigin !== undefined && !samePoint(previousOrigin, authoritativeOrigin)

        destinationByEntity.set(key, entity.destination)

        if (authoritativeOrigin) {
          authoritativeOriginByEntity.set(key, { ...authoritativeOrigin })
        }

        navigationController.updateArrivalAnimation(key, entity.animation)

        if (entity.authority !== 'authoritative') {
          destinationByEntity.delete(key)
          authoritativeOriginByEntity.delete(key)
          navigationController.cancel(key)
          activeNavigation.delete(key)
          entityRegistryController.setMoving(key, false)

          continue
        }

        const presentation = entityRegistryController.navigationEntity(key)

        if (
          presentation &&
          (previousDestination !== entity.destination || hasOriginCorrection) &&
          navigationController.move(presentation, entity.destination, entity.animation) &&
          navigationController.isMoving(key)
        ) {
          activeNavigation.add(key)
          entityRegistryController.setMoving(key, true)
        } else if (previousDestination !== entity.destination || hasOriginCorrection) {
          activeNavigation.delete(key)
          entityRegistryController.setMoving(key, false)
        }
      }

    }

    const applyOcclusion = (): void => {
      const focusedEntityKey = cameraController.getState().focusedEntityKey
      const selection = focusedEntityKey ? focusMetadata.get(focusedEntityKey)?.() : undefined

      occlusionController.update({ position: cameraPosition(camera) }, selection)
      interiorCutaway?.refreshExteriorVisibility()
    }

    const schedulerController = createFrameScheduler({
      captureMetrics: capturePerf,
      onFrame(frame) {
        const startedAt = typeof performance === 'undefined' ? Date.now() : performance.now()
        const previousCameraState = cameraController.getState()
        cameraController.update(frame.elapsedMs)
        const cameraState = cameraController.getState()

        if (
          previousCameraState.focusedEntityKey !== cameraState.focusedEntityKey ||
          previousCameraState.following !== cameraState.following
        ) {
          entityRegistryController.setSelection(cameraState.focusedEntityKey)
          const focusedLeaderId = cameraState.focusedEntityKey?.match(/^lunar-city:leader:(.+)$/u)?.[1]

          for (const [leaderId, node] of leaderNodes) {
            // Preserve metre-scale assets under focus. Legacy fixtures retain
            // their original emphasis; reviewed models use camera framing.
            const metricAsset =
              manifest.reviewLeaderAssets?.some(asset => asset.id === leaderId) ||
              manifest.characterAssets.leaders.some(asset => asset.id === leaderId && asset.heightMetres !== undefined)

            const emphasis = !metricAsset && leaderId === focusedLeaderId ? 1.14 : 1
            node.scaling?.set(emphasis, emphasis, emphasis)
          }

          emit({ kind: 'camera-state', state: cameraState })
        }

        const settings = quality.settings()
        const currentCameraPosition = cameraPosition(camera)

        const navigationActive = navigationController.tick(frame.elapsedMs)
        entityRegistryController.syncMotion()

        for (const key of [...activeNavigation]) {
          if (!navigationController.isMoving(key)) {
            activeNavigation.delete(key)
            entityRegistryController.setMoving(key, false)
            const destination=destinationByEntity.get(key)

            if(destination && navigationController.didArrive(key)) {buildingActivity.arrival(key,destination)}
          }
        }

        for (const staticModel of staticLods) {
          applyLodSelection(staticModel.lods, {
            distance: Math.hypot(
              currentCameraPosition.x - staticModel.focus.cameraAnchor.x,
              currentCameraPosition.y - staticModel.focus.cameraAnchor.y,
              currentCameraPosition.z - staticModel.focus.cameraAnchor.z
            ),
            lodAdvance: settings.lodAdvance,
            selected: cameraState.focusedEntityKey === staticModel.focus.focusEntityKey
          })
        }

        entityRegistryController.applyLodPolicy(
          (_key, position, isSelected) =>
            selectLodIndex(workerAsset.model.lods, {
              distance: Math.hypot(
                currentCameraPosition.x - position.x,
                currentCameraPosition.y - position.y,
                currentCameraPosition.z - position.z
              ),
              lodAdvance: settings.lodAdvance,
              selected: isSelected
            }),
          (_key, position, isSelected) =>
            isSelected ||
            Math.hypot(
              currentCameraPosition.x - position.x,
              currentCameraPosition.y - position.y,
              currentCameraPosition.z - position.z
            ) <= animationDistanceUnits(settings.animationDistance)
        )

        const leaderLifeActive = leaderLife.tick(frame.elapsedMs)
        const leaderDelay = leaderLife.nextWakeDelayMs()

        if (leaderDelay !== undefined) {schedulerController.requestVisualAt(frame.now + leaderDelay)}
        const socialActive = social.tick(frame.elapsedMs)
        const socialDelay = social.nextWakeDelay()

        if (socialDelay !== undefined) {schedulerController.requestVisualAt(frame.now + socialDelay)}
        atmosphere?.update(frame.elapsedMs, reducedMotion)

        if (atmosphere && !reducedMotion) {schedulerController.requestVisualAt(frame.now + 1000)}
        applyOcclusion()
        const renderStartedAt = typeof performance === 'undefined' ? Date.now() : performance.now()
        scene.render()
        const leaderAnimationActive = characterAnimations.isActive()
        const finishedAt = typeof performance === 'undefined' ? Date.now() : performance.now()
        lastFrameMs = Math.max(0, finishedAt - startedAt)
        lastWorldUpdateMs = Math.max(0, renderStartedAt - startedAt)
        worldUpdateTimestampsMs?.push(renderStartedAt)

        if (
          quality.noteFrame({ elapsedMs: Math.max(0, finishedAt - startedAt), interactive: frame.targetFps === 30 })
        ) {
          applyRuntimeQuality()
        }

        return (
          buildingActivity.tick() ||
          socialActive ||
          leaderLifeActive ||
          navigationActive ||
          activeNavigation.size > 0 ||
          cameraController.isTransitioning() ||
          cameraState.following ||
          cameraController.isIdleActive() ||
          cameraController.isIdlePending() ||
          entityRegistryController.hasActiveAnimations() ||
          leaderAnimationActive
        )
      },
      renderer: engine
    })

    scheduler = schedulerController

    schedulerController.bindRendererPauseState()

    await scene.whenReadyAsync()

    return {
      getInteriorBuildings: () => modules.interiorPlans?.map(({ id, title }) => ({ id, title })) ?? [],
      setInteriorBuilding(id) {
        if (disposed) {return false}
        const shown = interiorCutaway?.setBuilding(id) ?? false
        interiorBuildingId = shown ? id : undefined
        schedulerController.requestRender()

        return shown
      },
      leaderStateClips,
      getWorkerEncounters: social.encounters,
      getWorkerPresentation(key) { const record = entityRegistryController.entity(key);

 return record ? { position: { ...record.position }, moving: record.moving, animation: record.animation } : undefined },
      requestWorkerInteraction(key, gesture) {
        const started = social.request(key, gesture)

        if (started) {schedulerController.requestRender()}

        return started
      },
      applySnapshot(snapshot) {
        if (!disposed) {
          arrivalSlots.update(snapshot)
          buildingActivity.update(snapshot)
          updateObservedLeaderLife(snapshot)
          social.update(snapshot)
          reconcileProjectCompounds(snapshot)

          lastWorkerSnapshot = snapshot
          reconcileWorkers(snapshot)

          schedulerController.requestRender()
        }
      },
      projectWorldPoint(point) {
        return projectWorldPoint(point, scene.getTransformMatrix?.().m)
      },
      dispatchCamera(intent) {
        if (disposed) {
          return
        }

        cameraController.dispatch(intent)
        const cameraState = cameraController.getState()

        if (interiorBuildingId && cameraState.focusedEntityKey !== staticFocusKey('model', interiorBuildingId)) {
          interiorCutaway?.setBuilding(undefined)
          interiorBuildingId = undefined
        }

        entityRegistryController.setSelection(cameraState.focusedEntityKey)
        emit({ kind: 'camera-state', state: cameraState })
        schedulerController.noteInteraction(typeof performance === 'undefined' ? Date.now() : performance.now())
      },
      getCameraState() {
        return cameraController.getState()
      },
      getPerfSnapshot() {
        const schedulerMetrics = schedulerController.getMetrics()
        const babylon = perfAdapter.snapshot()
        const qualitySettings = quality.settings()

        return {
          ...babylon,
          activeAnimations: entityRegistryController.activeAnimationCount() + characterAnimations.activeCount(),
          cameraAlpha: camera.alpha,
          cameraBeta: camera.beta,
          cameraRadius: camera.radius,
          frameMs: lastFrameMs,
          frameTimestampsMs: schedulerMetrics.frameTimestampsMs,
          internalRenderScale: qualitySettings.renderScale,
          listeners: schedulerMetrics.listeners,
          rafs: schedulerMetrics.rafs,
          renderFrames: schedulerMetrics.renderFrames,
          targetFps: schedulerMetrics.targetFps,
          timers: schedulerMetrics.timers,
          qualityTier: qualitySettings.tier,
          worldUpdateMs: lastWorldUpdateMs,
          worldUpdateTimestampsMs: worldUpdateTimestampsMs ? [...worldUpdateTimestampsMs] : []
        }
      },
      getEntityCameraOrder() {
        const position = cameraPosition(camera)

        return [...currentEntityKeys].sort((left, right) => {
          const leftPosition = entityRegistryController.entity(left)?.position
          const rightPosition = entityRegistryController.entity(right)?.position

          const leftDistance = leftPosition
            ? Math.hypot(leftPosition.x - position.x, leftPosition.y - position.y, leftPosition.z - position.z)
            : Number.POSITIVE_INFINITY

          const rightDistance = rightPosition
            ? Math.hypot(rightPosition.x - position.x, rightPosition.y - position.y, rightPosition.z - position.z)
            : Number.POSITIVE_INFINITY

          return leftDistance - rightDistance || left.localeCompare(right)
        })
      },
      reviewWorkerClips,
      setReviewWorkerAnimation(id, state) {
        if (!disposed) {
          characterAnimations.set(`worker:${id}`, state)
        }
      },
      getLeaderLife(id) { const life = leaderLife.get(id);

 return life ? { ...life, manualOverride: leaderOverrides.has(id) } : undefined },
      setLeaderLifeMode(id, mode) {
        if (disposed) {return}

        if (mode === 'automatic') {
          leaderOverrides.delete(id)

          if (leaderSnapshot) {updateObservedLeaderLife(leaderSnapshot)}

          if (!observedModels.has(id)) {leaderLife.setMode(id, 'idle')}
        } else {
          leaderOverrides.add(id)
          leaderLife.setMode(id, mode)
        }

        schedulerController.requestRender()
      },
      setLeaderAnimation(leaderId, state) {
        if (!disposed) {
          leaderLife.hold(leaderId, state !== 'idle' && state !== 'walk')
          characterAnimations.set(leaderId, state)
        }
      },
      pick(clientX, clientY) {
        const target = pickedCameraTarget(scene.pick?.(clientX, clientY)?.pickedMesh)

        if (target) {
          emit({ kind: 'select-focus', entityKey: target.entityKey })
        } else {
          emit({ kind: 'clear-selection' })
        }

        return target
      },
      setQuality(tier) {
        quality.setTier(tier)
        applyRuntimeQuality()
        schedulerController.requestRender()
      },
      setReducedMotion(reduced) {
        if (disposed || reducedMotion === reduced) {
          return
        }

        reducedMotion = reduced
        leaderLife.setReducedMotion(reduced)
        cameraController.setReducedMotion(reduced)
        navigationController.setReducedMotion(reduced)
        entityRegistryController.setReducedMotion(reduced)

        if (reduced) {
          for (const key of [...activeNavigation]) {
            activeNavigation.delete(key)
            entityRegistryController.setMoving(key, false)
          }

          entityRegistryController.syncMotion()
        }

        if (!reduced && lastWorkerSnapshot) {
          destinationByEntity.clear()
          authoritativeOriginByEntity.clear()
          reconcileWorkers(lastWorkerSnapshot)
        }

        buildingActivity.setReducedMotion(reduced)
        social.tick(0)
        characterAnimations.setReducedMotion(reduced)

        schedulerController.requestRender()
      },
      setVisible(visible) {
        leaderLife.setPaused(!visible)
        schedulerController.setVisible(visible)
      },
      render() {
        if (!disposed) {
          schedulerController.requestRender()
          schedulerController.tick(typeof performance === 'undefined' ? Date.now() : performance.now())
        }
      },
      dispose() {
        disposeWorld()
      }
    }
  } catch (error) {
    disposeWorld()
    throw error
  }
}
