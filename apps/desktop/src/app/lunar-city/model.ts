import type { WorkerEncounter } from './worker-social'
import type { CutawayExteriorMesh, InteriorCutawayController } from './world/interior-cutaway'
import type { AuthoredInteriorPlan } from './world/interior-plan-data'
import type { LeaderLifeMode, LeaderLifePresentation } from './world/leader-life'
export type EntityKey = string & { readonly entityKey: unique symbol }
export type AuthorityState = 'authoritative' | 'partial' | 'stale' | 'unknown'
export type DestinationId =
  | 'bus'
  | 'council'
  | 'depot'
  | 'garden'
  | 'lab'
  | 'library'
  | 'project'
  | 'review'
  | 'triage'
  | 'unavailable'
  | 'unknown'
export type QualityTier = 'efficient' | 'balanced' | 'detailed'

export interface Vec3 {
  x: number
  y: number
  z: number
}

export type EntityIdentity =
  | { kind: 'profile'; connectionId: string; profile: string }
  | { kind: 'session'; connectionId: string; profile: string; sessionId: string }
  | { kind: 'subagent'; connectionId: string; profile: string; sessionId: string; subagentId: string }
  | {
      kind: 'kanban'
      connectionId: string
      profile: string
      board: string
      taskId: string
      runId?: string
      workerId?: string
    }

export interface SourceHealth {
  source: string
  authority: AuthorityState
  observedAt: number
  error?: string
}

export interface CameraLandmark {
  id: string
  alpha: number
  beta: number
  radius: number
  target: Vec3
  minBeta: number
  maxBeta: number
  minRadius: number
  maxRadius: number
}

export interface WorldBounds {
  min: Vec3
  max: Vec3
}

export interface ModelStatistics {
  animationClips: readonly string[]
  drawCalls: number
  materials: number
  meshes: number
  nodes: number
  textures: number
  triangles: number
  budget: {
    maxDrawCalls: number
    maxGpuMiB: number
    maxMaterials: number
    maxTextures: number
    maxTriangles: number
  }
  bytes: number
  extent: readonly [number, number, number]
  gpuMiB: number
  sha256: string
}

export interface ModelManifestEntry {
  id: string
  uri: string
  maxTriangles: number
  maxDrawCalls: number
  maxMaterials: number
  maxTextures: number
  maxGpuMiB: number
  requiredNodes: readonly string[]
  requiredClips: readonly string[]
  lods: readonly { distance: number; node: string }[]
  transform: { position: Vec3; rotation: Vec3; scale: Vec3 }
  pivot: Vec3
  bounds: WorldBounds
  anchors: Readonly<Record<string, Vec3>>
  cameraAnchor: Vec3
  occlusionGroup: string
  collision: { kind: string; navigationArea: string }
  materialSlots: readonly string[]
  instancing?: { eligible: boolean; variants: readonly string[] }
  statistics: ModelStatistics
}

export type NavigationCollider =
  | { id: string; kind: 'box'; center: Vec3; halfExtents: Vec3; rotationY: number }
  | { id: string; kind: 'cylinder'; center: Vec3; radius: number; height: number }

export interface NavigationManifest {
  colliders?: readonly NavigationCollider[]
  sha256?: string
  meshUri: string
  areas: readonly string[]
  links: readonly { from: Vec3; to: Vec3; bidirectional: boolean }[]
}

export interface TextureManifestEntry {
  bytes: number
  sha256: string
  source: string
  uri: string
}

export interface MaterialManifestEntry {
  id: string
  palette: string
  maxTextures: number
}

export interface ProjectSlotManifestEntry {
  id: string
  position: Vec3
  bounds: WorldBounds
  navigationLink: { from: Vec3; to: Vec3; bidirectional: boolean }
}

export interface WorkerCharacterSignature {
  body: string
  emblem?: string
  head: string
  palette: string
  silhouetteAccessory?: string
}

export interface WorkerCharacterKit {
  group: string
  kitId: string
  signature: WorkerCharacterSignature
}

export interface LeaderCharacterAsset {
  heightMetres?: number
  id: LeaderId
  silhouetteId: string
  species: string
  visualId: string
}

export interface CharacterAssetManifest {
  fleetIdentityFloor: number
  groupKits: readonly WorkerCharacterKit[]
  leaders: readonly LeaderCharacterAsset[]
  lodRepresentations: readonly {
    animated: boolean
    id: 'near' | 'mid' | 'far'
    representation: 'full' | 'reduced' | 'static-or-aggregate'
  }[]
  physicalVariantRoots: {
    activationScale: Readonly<Record<string, Vec3>>
    body: Readonly<Record<string, string>>
    groupKit: { emblemSuffix: string; identityAccentSuffix: string; silhouetteSuffix: string }
    head: Readonly<Record<string, string>>
    palette: Readonly<Record<string, string>>
  }
  sharedResourceStrategy: {
    animationClips: 'shared'
    gpuBuffers: 'shared'
    materials: 'shared'
    perProfile: { materials: 0; meshes: 0; skeletons: 0; textures: 0 }
    rig: string
    textureAtlas: string
  }
  workerVocabulary: {
    bodies: readonly string[]
    emblems: readonly string[]
    heads: readonly string[]
    palettes: readonly string[]
    silhouetteAccessories: readonly string[]
  }
}

export interface WorkerCharacterPresentation {
  /** Stable district slot encoded by the shared identity-accent transform. */
  accentCode?: number
  /** Exact-key-derived, resource-free visible accent token. */
  identityAccent?: string
  kitId?: string
  lod: 'near' | 'mid' | 'far'
  renderMode: 'animated' | 'instanced' | 'aggregate'
  signature?: WorkerCharacterSignature
  visibleSignature?: string
}

export interface QualityBudget {
  drawCalls: number
  visibleTriangles: number
  gpuMiB: number
}

export interface ExternalDecorationManifestEntry {
  id: string
  uri: string
  position: Vec3
  rotation: Vec3
  scale: Vec3
}

export interface ReviewCharacterAsset {
  id: string
  uri: string
  position: Vec3
  rotationY: number
  heightMetres: number
}

export interface ReviewLeaderAsset extends ReviewCharacterAsset {
  id: LeaderId
}

export interface WorldManifestV2 {
  reviewWorkerAssets?: readonly ReviewCharacterAsset[]
  reviewLeaderAssets?: readonly ReviewLeaderAsset[]
  externalDecorations?: readonly ExternalDecorationManifestEntry[]
  version: 2
  assetVersion: '2.0.0'
  source: { sha256: string }
  characterAssets: CharacterAssetManifest
  materials: readonly MaterialManifestEntry[]
  models: readonly ModelManifestEntry[]
  textures: readonly TextureManifestEntry[]
  camera: { overview: CameraLandmark; bounds: WorldBounds; followOffset: Vec3 }
  navigation: NavigationManifest
  destinations: Readonly<Record<string, Vec3>>
  projectSlots: readonly ProjectSlotManifestEntry[]
  qualityBudgets: {
    balancedOverview: QualityBudget
    balancedWorkerFocus: QualityBudget
  }
  generatedAssetPack: Readonly<Record<string, unknown>>
}

export interface LunarEntity {
  key: EntityKey
  identity: EntityIdentity
  observedAt: number
  authority: AuthorityState
  destination: DestinationId
  animation: string
  /** Exact upstream state used for command compatibility; never inferred from animation. */
  sourceState?: string
  /** Presentation position only; it never represents work progress. */
  position?: Vec3
  projectId?: string
  /** Declared worker colour/accessory variant, never a display-name identity. */
  variant?: string
  /** Bounded presentation metadata. It never participates in identity or command authority. */
  presentation?: LunarEntityPresentation
}

export interface LunarGroupMembership {
  id: string
  name: string
}

export interface LunarEntityPlacement {
  lodHint: number
  overflow: boolean
  primaryGroupId?: string
  /** Physical lattice slot; absent when bounded capacity is exhausted and the row is aggregate-only. */
  slot?: number
}

export interface LunarEntityPresentation {
  configuredTitle?: string
  groups: readonly LunarGroupMembership[]
  metadata: LunarPresentationMetadata
  placement: LunarEntityPlacement
  profileHandle?: string
  sourceLabel?: string
}

export interface LunarPresentationMetadata {
  observedAt?: number
  source: string
  state: 'fresh' | 'stale' | 'unavailable'
}

export interface LunarCitySnapshot {
  revision: number
  observedAt: number
  entities: ReadonlyMap<EntityKey, LunarEntity>
  sources: readonly SourceHealth[]
}

export type CameraIntent =
  | { kind: 'orbit'; deltaAlpha: number; deltaBeta: number }
  | { kind: 'pan'; deltaX: number; deltaZ: number }
  | { kind: 'zoom'; delta: number }
  | { kind: 'focus'; entityKey: EntityKey; follow: boolean }
  | { kind: 'clear-focus' }
  | { kind: 'return-to-city' }

export interface CameraControlState {
  focusedEntityKey: EntityKey | undefined
  following: boolean
}

export type LunarCityIntent =
  | { kind: 'camera-state'; state: CameraControlState }
  | { kind: 'clear-selection' }
  | { kind: 'select-focus'; entityKey: EntityKey }
  | { kind: 'select-landmark'; landmarkId: string }

export type LeaderAnimationState =
  'acknowledging' | 'idle' | 'listening' | 'talking' | 'thinking' | 'unavailable' | 'walk'
export type LeaderId =
  'owl' | 'fox' | 'badger' | 'otter' | 'bird' | 'stag' | 'elephant' | 'cat' | 'capybara' | 'lion' | 'beaver' | 'monkey'

export type LeaderStateClipMap = Readonly<Partial<Record<LeaderAnimationState, string>>>

export interface LunarCityLandmarkMetadata {
  cameraAnchor: Vec3
  focusEntityKey: EntityKey
  kind: 'landmark' | 'landmark-mesh'
  modelId: string
  occlusionGroup: string
  selectable: boolean
}

export interface LunarCityLodMetadata {
  distance: number
  kind: 'lod'
  modelId: string
}

export interface LunarCityLeaderPickMetadata {
  cameraAnchor: Vec3
  focusEntityKey: EntityKey
  kind: 'leader'
  leaderId: LeaderId
  modelId: 'leaders'
  occlusionGroup: string
  selectable: true
  stateClips: LeaderStateClipMap
}

/** Exact worker identity attached to every clone and hardware instance. */
export interface LunarCityWorkerPickMetadata {
  cameraAnchor: Vec3
  character?: WorkerCharacterPresentation
  entityKey: EntityKey
  focusEntityKey: EntityKey
  identity: EntityIdentity
  kind: 'worker'
  modelId: 'workers'
  occlusionGroup: string
  selectable: true
  variant?: string
}

/** Stable, non-interactive anchor for a manifest-slotted Kanban project compound. */
export interface LunarCityProjectCompoundMetadata {
  connectionId: string
  key: string
  kind: 'project-compound'
  projectId: string
  selectable: false
}

export interface LunarCitySharedLeaderSurfaceMetadata {
  cameraAnchor: Vec3
  focusEntityKey: EntityKey
  kind: 'leader-shared-surface'
  modelId: 'leaders'
  occlusionGroup: string
  selectable: false
}

export interface LunarCityReviewWorkerMetadata extends Omit<
  LunarCityLeaderPickMetadata,
  'kind' | 'leaderId' | 'modelId' | 'stateClips'
> {
  kind: 'review-worker'
  workerReviewId: string
  modelId: 'workers'
  stateClips: Readonly<Record<string, string>>
}

export type LunarCityNodeMetadata =
  | LunarCityReviewWorkerMetadata
  | LunarCityLandmarkMetadata
  | LunarCityLeaderPickMetadata
  | LunarCityLodMetadata
  | LunarCityProjectCompoundMetadata
  | LunarCitySharedLeaderSurfaceMetadata
  | LunarCityWorkerPickMetadata

export interface LunarCityWorldHandle {
  setLeaderLifeMode?(id: LeaderId, mode: LeaderLifeMode | 'automatic'): void
  getLeaderLife?(id: LeaderId): LeaderLifePresentation | undefined
  setInteriorBuilding?(id?: string): boolean
  getInteriorBuildings?(): readonly { id: string; title: string }[]
  getWorkerPresentation?(key: EntityKey): { position: Vec3; moving: boolean; animation: string } | undefined
  requestWorkerInteraction?(key: EntityKey, gesture?: string): boolean
  getWorkerEncounters?(): readonly WorkerEncounter[]
  readonly reviewWorkerClips?: ReadonlyMap<string, Readonly<Record<string, string>>>
  setReviewWorkerAnimation?(id: string, state: string): void
  readonly leaderStateClips: ReadonlyMap<string, LeaderStateClipMap>
  applySnapshot(snapshot: LunarCitySnapshot): void
  dispatchCamera(intent: CameraIntent): void
  getEntityCameraOrder(): readonly EntityKey[]
  projectWorldPoint?(point: Vec3): { x: number; y: number; visible: boolean } | undefined
  getCameraState(): CameraControlState
  getPerfSnapshot?(): {
    activeAnimations: number
    drawCalls: number
    entities: number
    frameMs: number
    frameTimestampsMs: readonly number[]
    listeners: number
    rafs: number
    renderFrames: number
    targetFps: 0 | 15 | 30
    textures: number
    timers: number
    visibleTriangles: number
    worldUpdateMs: number
    worldUpdateTimestampsMs: readonly number[]
  }
  /** Plays only a state clip declared by the selected leader's GLB metadata. */
  setLeaderAnimation(leaderId: LeaderId, state: LeaderAnimationState): void
  setQuality(tier: QualityTier): void
  setReducedMotion(reduced: boolean): void
  destroy(): void
}

export interface BabylonVector3Like {
  readonly x: number
  readonly y: number
  readonly z: number
}

export interface BabylonMutableVectorLike {
  x?: number
  y?: number
  z?: number
  set(x: number, y: number, z: number): void
}

export interface BabylonNodeLike {
  name: string
  metadata?: Record<string, unknown> & { lunarCity?: LunarCityNodeMetadata }
  parent?: BabylonNodeLike | null
  position?: BabylonMutableVectorLike
  rotation?: BabylonMutableVectorLike
  scaling?: BabylonMutableVectorLike
  clone?(name: string, parent?: BabylonNodeLike | null, doNotCloneChildren?: boolean): BabylonNodeLike | null
  dispose?(): void
  isEnabled?(): boolean
  setEnabled?(enabled: boolean): void
  setPivotPoint?(point: BabylonVector3Like): void
}

export interface BabylonMeshLike extends BabylonNodeLike {
  freezeWorldMatrix?(): void
  getIndices?(): readonly number[] | null
  getTotalVertices?(): number
  getVerticesData?(kind: string): readonly number[] | null
  getWorldMatrix?(): { m: readonly number[] }
  isPickable?: boolean
  receiveShadows?: boolean
  material?: {
    alpha?: number
    clone?(name: string): unknown
  } | null
}

export interface BabylonMaterialLike {
  freeze?(): void
}

export interface BabylonEngineLike {
  dispose(): void
  resize(): void
  setHardwareScalingLevel?(level: number): void
  stopRenderLoop?(): void
}

export interface BabylonSceneLike {
  useRightHandedSystem?: boolean
  _activeIndices?: { current?: number }
  activeCamera?: unknown
  ambientColor?: unknown
  clearColor?: unknown
  fogColor?: unknown
  fogDensity?: number
  fogEnd?: number
  fogMode?: number
  fogStart?: number
  imageProcessingConfiguration?: {
    contrast?: number
    exposure?: number
    toneMappingEnabled?: boolean
    toneMappingType?: number
    vignetteColor?: unknown
    vignetteEnabled?: boolean
    vignetteWeight?: number
  }
  materials?: readonly BabylonMaterialLike[]
  meshes?: readonly BabylonMeshLike[]
  textures?: readonly unknown[]
  dispose(): void
  render(): void
  whenReadyAsync(): Promise<void>
  getTransformMatrix?(): { m: ArrayLike<number> }
  pick?(x: number, y: number): { pickedMesh?: BabylonNodeLike } | undefined
}

/** Babylon's `ShadowGenerator`, narrowed to what the world scene drives. */
export interface BabylonShadowGeneratorLike {
  bias?: number
  darkness?: number
  normalBias?: number
  transparencyShadow?: boolean
  useExponentialShadowMap?: boolean
  usePercentageCloserFiltering?: boolean
  filteringQuality?: number
  getShadowMap?(): { renderList?: BabylonMeshLike[] | null } | null
  addShadowCaster?(mesh: BabylonMeshLike, includeDescendants?: boolean): unknown
  dispose?(): void
}

export interface BabylonImportResultLike {
  meshes: readonly BabylonMeshLike[]
  transformNodes: readonly BabylonNodeLike[]
  animationGroups: readonly unknown[]
}

export interface BabylonLightLike {
  intensity: number
  diffuse?: unknown
  shadowEnabled?: boolean
  specular?: unknown
}

export interface BabylonHemisphericLightLike extends BabylonLightLike {
  groundColor?: unknown
}

export interface BabylonGlowLayerLike {
  intensity: number
  dispose?(): void
}

export interface RecastPathLike {
  __destroy__?(): void
  delete?(): void
  destroy?(): void
  getPoint(index: number): BabylonVector3Like | undefined
  getPointCount(): number
}

export interface RecastNavMeshLike {
  build(
    positions: Float32Array,
    positionCount: number,
    indices: Uint32Array,
    indexCount: number,
    configuration: unknown
  ): void
  computePath(from: unknown, to: unknown): RecastPathLike
  destroy?(): void
}

export interface RecastWrapperLike {
  __destroy__?(): void
  delete?(): void
  destroy?(): void
}

export interface RecastConfigLike extends Record<string, unknown>, RecastWrapperLike {
  set_bmax?(index: number, value: number): void
  set_bmin?(index: number, value: number): void
}

export interface RecastRuntimeLike {
  NavMesh: new () => RecastNavMeshLike
  Vec3: new (x: number, y: number, z: number) => BabylonVector3Like & RecastWrapperLike
  rcConfig: new () => RecastConfigLike
}

export interface LunarCityWorldModules {
  interiorPlans?: readonly AuthoredInteriorPlan[]
  createInteriorCutaway?(scene: BabylonSceneLike, getExteriorMeshes: (id: string) => readonly CutawayExteriorMesh[]): InteriorCutawayController
  createAtmosphere?(scene: BabylonSceneLike): { update(elapsedMs: number, paused: boolean): void; dispose(): void }
  createProjectMarker?(name: string, scene: BabylonSceneLike): BabylonNodeLike
  Engine: new (
    canvas: HTMLCanvasElement,
    antialias: boolean,
    options: { alpha?: boolean; powerPreference: 'low-power'; preserveDrawingBuffer: false; stencil: false }
  ) => BabylonEngineLike
  Scene: new (engine: BabylonEngineLike) => BabylonSceneLike
  SceneInstrumentation?: new (scene: BabylonSceneLike) => { dispose(): void; drawCallsCounter?: { current?: number } }
  Vector3: new (x: number, y: number, z: number) => BabylonVector3Like
  Color3: new (red: number, green: number, blue: number) => unknown
  Color4?: new (red: number, green: number, blue: number, alpha: number) => unknown
  ArcRotateCamera: new (
    name: string,
    alpha: number,
    beta: number,
    radius: number,
    target: BabylonVector3Like,
    scene: BabylonSceneLike
  ) => unknown
  DirectionalLight: new (name: string, direction: BabylonVector3Like, scene: BabylonSceneLike) => BabylonLightLike
  HemisphericLight?: new (
    name: string,
    direction: BabylonVector3Like,
    scene: BabylonSceneLike
  ) => BabylonHemisphericLightLike
  GlowLayer?: new (
    name: string,
    scene: BabylonSceneLike,
    options?: { mainTextureRatio?: number }
  ) => BabylonGlowLayerLike
  ShadowGenerator?: new (mapSize: number, light: BabylonLightLike) => BabylonShadowGeneratorLike
  TransformNode: new (name: string, scene: BabylonSceneLike) => BabylonNodeLike
  ImportMeshAsync(source: string, scene: BabylonSceneLike, options?: { pluginExtension?: string }): Promise<BabylonImportResultLike>
  createRecastNavigation?(): Promise<RecastRuntimeLike>
}
