export type WorldAssetKind = 'animation' | 'building' | 'character' | 'prop' | 'road' | 'terrain' | 'texture'
export type WorldCollection = 'Buildings' | 'Characters' | 'Lighting' | 'Props' | 'Roads' | 'Terrain'
export type WorldLod = 'hero' | 'high' | 'medium' | 'low'
export type WorldTextureResolution = '2k' | '4k'

export interface WorldTextureSlot {
  resolution: WorldTextureResolution
  slot: 'albedo' | 'emissive' | 'metallicRoughness' | 'normal'
  uri: string
}

export interface WorldAssetBinding {
  eventKinds?: string[]
  homeBuilding?: string
  objectNames: string[]
  role?: string
}

export interface WorldAssetManifestEntry {
  bindings?: WorldAssetBinding[]
  collection: WorldCollection
  id: string
  instanced?: boolean
  kind: WorldAssetKind
  lod: WorldLod[]
  role?: string
  textureSet?: string
  visualTags?: string[]
}

export interface WorldAnimationManifestEntry {
  actorKinds: Array<'child' | 'dispatcher' | 'leader' | 'worker'>
  clip: string
  loop: boolean
  tags: string[]
}

export interface WorldCollectionManifestEntry {
  id: WorldCollection | 'Character Asset Library'
  render: boolean
  purpose: string
}

export interface WorldTextureManifestEntry {
  id: string
  maxResolution: WorldTextureResolution
  slots: WorldTextureSlot[]
}

export interface WorldValidationManifest {
  blenderScript: string
  metadata: string
  requires: string[]
}

export interface WorldGenerated3DAssetManifestEntry {
  id: string
  kind: 'building' | 'child' | 'leader' | 'prop' | 'vehicle' | 'worker'
  mesh: string
  pbrStatus: 'needs_rebake' | 'source_materials'
  sourceReferenceCrop: string
  status: 'imported' | 'missing'
}

export interface WorldGenerated3DManifest {
  assetCount: number
  assets: WorldGenerated3DAssetManifestEntry[]
  blend: string
  glb: string
  importedCount: number
  missingCount: number
  preview: string
  productionEligibility: 'rejected_for_production'
  productionUse: 'reference_only'
  rejectionReason: string
  sourceManifest: string
}

export interface WorldProductionAssetPipeline {
  highPolyMasterFirst: boolean
  masking: 'required_before_image_to_3d_generation'
  productionSource: 'full_resolution_high_poly_master_assets'
  rejectedSources: Array<'raw_scene_crop_image_to_3d'>
  retopology: 'derive_smart_low_poly_lods_from_master'
  textureBake: 'bake_2k_default_4k_hero_pbr_from_master'
}

export interface WorldAssetManifest {
  animationClips: WorldAnimationManifestEntry[]
  assets: WorldAssetManifestEntry[]
  collections: WorldCollectionManifestEntry[]
  glb: string
  generated3dBoardGlb?: string
  generated3dBoardPreview?: string
  generated3dManifest?: string
  heroAssetGlb?: string
  heroAssetManifest?: string
  heroAssetPreview?: string
  masterAssetManifest?: string
  masterAssetMaskManifest?: string
  masterAssetMaskReviewPreview?: string
  masterAssetRejectedCandidates?: string
  profileManifest: string
  productionAssetPipeline: WorldProductionAssetPipeline
  renderProfile: string
  scene: string
  schemaVersion: 2
  source: string
  textures: WorldTextureManifestEntry[]
  units: 'meters'
  validation: WorldValidationManifest
}

const BUILDINGS: Array<[id: string, role: string, textureSet: string, tags: string[]]> = [
  ['library', 'knowledge', 'library-facade', ['books', 'violet', 'leader-home']],
  ['research-lab', 'research', 'research-facade', ['lab', 'cyan', 'leader-home']],
  ['arts-studio', 'creative', 'studio-facade', ['art', 'green', 'leader-home']],
  ['council-hall', 'governance', 'council-facade', ['council', 'violet', 'leader-home']],
  ['engineering-workshop', 'engineering', 'engineering-facade', ['tools', 'cyan', 'leader-home']],
  ['triage-clinic', 'medical', 'clinic-facade', ['medical', 'amber', 'leader-home']],
  ['review-office', 'review', 'review-facade', ['review', 'violet', 'leader-home']],
  ['archive', 'archive', 'archive-facade', ['archive', 'violet', 'leader-home']]
]

const LEADERS: Array<[id: string, role: string, homeBuilding: string, tags: string[]]> = [
  ['leader-knowledge', 'knowledge', 'library', ['leader', 'robe', 'violet']],
  ['leader-research', 'research', 'research-lab', ['leader', 'lab-coat', 'cyan']],
  ['leader-creative', 'creative', 'arts-studio', ['leader', 'artist-smock', 'green']],
  ['leader-governance', 'governance', 'council-hall', ['leader', 'formal', 'violet']],
  ['leader-engineering', 'engineering', 'engineering-workshop', ['leader', 'utility-rig', 'cyan']],
  ['leader-medical', 'medical', 'triage-clinic', ['leader', 'medic-rig', 'amber']],
  ['leader-review', 'review', 'review-office', ['leader', 'inspector', 'violet']],
  ['leader-archive', 'archive', 'archive', ['leader', 'archivist', 'violet']]
]

const WORKERS: Array<[id: string, role: string, homeBuilding: string, tags: string[]]> = [
  ['worker-audit', 'audit', 'review-office', ['worker', 'inspection', 'methodical']],
  ['worker-operations', 'operations', 'engineering-workshop', ['worker', 'operations', 'protective']],
  ['worker-release', 'release', 'triage-clinic', ['worker', 'release', 'bold']],
  ['worker-research', 'research', 'research-lab', ['worker', 'research', 'curious']],
  ['worker-review', 'review', 'review-office', ['worker', 'review', 'methodical']],
  ['worker-support', 'support', 'council-hall', ['worker', 'support', 'social']]
]

const CHILDREN: Array<[id: string, role: string, homeBuilding: string, tags: string[]]> = [
  ['child-curious', 'child', 'break-garden', ['child', 'curious']],
  ['child-social', 'child', 'break-garden', ['child', 'social']],
  ['child-bold', 'child', 'break-garden', ['child', 'bold']],
  ['child-cautious', 'child', 'break-garden', ['child', 'cautious']]
]

const ANIMATION_CLIPS = [
  ['idle', true, ['idle']],
  ['walk', true, ['travel']],
  ['work', true, ['task.work']],
  ['carry', true, ['delivery']],
  ['inspect', false, ['inspect']],
  ['repair', false, ['repair', 'recover']],
  ['talk', false, ['dialogue']],
  ['wait', true, ['waiting']],
  ['panic', false, ['crisis']],
  ['celebrate', false, ['celebration']],
  ['rest', true, ['resting']],
  ['return', true, ['returning']]
] as const

function pbrSlots(id: string, resolution: WorldTextureResolution): WorldTextureSlot[] {
  return [
    { resolution, slot: 'albedo', uri: `lunar-city/textures/${id}/albedo.png` },
    { resolution, slot: 'normal', uri: `lunar-city/textures/${id}/normal.png` },
    { resolution, slot: 'metallicRoughness', uri: `lunar-city/textures/${id}/metallic-roughness.png` },
    { resolution, slot: 'emissive', uri: `lunar-city/textures/${id}/emissive.png` }
  ]
}

export const LUNAR_CITY_ASSET_MANIFEST: WorldAssetManifest = {
  animationClips: ANIMATION_CLIPS.map(([clip, loop, tags]) => ({
    actorKinds: ['child', 'dispatcher', 'leader', 'worker'],
    clip,
    loop,
    tags: [...tags]
  })),
  assets: [
    {
      collection: 'Terrain',
      id: 'terrain-colony-basin',
      kind: 'terrain',
      lod: ['high', 'medium', 'low'],
      textureSet: 'lunar-regolith',
      visualTags: ['concave', 'grounded', 'collision-source']
    },
    {
      collection: 'Roads',
      id: 'road-network-primary',
      instanced: true,
      kind: 'road',
      lod: ['high', 'medium', 'low'],
      textureSet: 'road-composite',
      visualTags: ['terrain-conforming', 'city-planning']
    },
    ...BUILDINGS.map(([id, role, textureSet, visualTags]) => ({
      bindings: [{ homeBuilding: id, objectNames: [`${id}_shell`, `${id}_entry`, `${id}_sign`], role }],
      collection: 'Buildings' as const,
      id,
      instanced: false,
      kind: 'building' as const,
      lod: ['hero', 'high', 'medium', 'low'] as WorldLod[],
      role,
      textureSet,
      visualTags
    })),
    ...LEADERS.map(([id, role, homeBuilding, visualTags]) => ({
      bindings: [{ homeBuilding, objectNames: [`${id}_body`, `${id}_head`], role }],
      collection: 'Characters' as const,
      id,
      kind: 'character' as const,
      lod: ['hero', 'high', 'low'] as WorldLod[],
      role,
      textureSet: 'leader-suits',
      visualTags
    })),
    ...WORKERS.map(([id, role, homeBuilding, visualTags]) => ({
      bindings: [{ homeBuilding, objectNames: [`${id}_body`, `${id}_head`], role }],
      collection: 'Characters' as const,
      id,
      instanced: true,
      kind: 'character' as const,
      lod: ['high', 'medium', 'low'] as WorldLod[],
      role,
      textureSet: 'worker-suits',
      visualTags
    })),
    ...CHILDREN.map(([id, role, homeBuilding, visualTags]) => ({
      bindings: [{ homeBuilding, objectNames: [`${id}_body`, `${id}_head`], role }],
      collection: 'Characters' as const,
      id,
      instanced: true,
      kind: 'character' as const,
      lod: ['medium', 'low'] as WorldLod[],
      role,
      textureSet: 'child-suits',
      visualTags
    })),
    {
      bindings: [{ objectNames: ['dispatcher-cube'], role: 'dispatcher' }],
      collection: 'Characters',
      id: 'dispatcher-cube',
      kind: 'character',
      lod: ['hero', 'high', 'low'],
      role: 'dispatcher',
      textureSet: 'dispatcher-glass',
      visualTags: ['dispatcher', 'companion', 'new-task']
    },
    {
      collection: 'Props',
      id: 'break-garden',
      instanced: true,
      kind: 'prop',
      lod: ['high', 'medium', 'low'],
      textureSet: 'garden-biome',
      visualTags: ['resting', 'garden', 'children']
    }
  ],
  collections: [
    { id: 'Terrain', purpose: 'Ground, collision, crater detail, and terrain height authority.', render: true },
    { id: 'Roads', purpose: 'Terrain-conforming travel network and route highlights.', render: true },
    { id: 'Buildings', purpose: 'Role-specific skinned structures and interiors.', render: true },
    { id: 'Characters', purpose: 'Runtime leader, worker, child, and dispatcher instances.', render: true },
    { id: 'Props', purpose: 'Garden, signage, tools, transport, repair, and celebration props.', render: true },
    { id: 'Lighting', purpose: 'Baked and preview lighting, camera, and skybox controls.', render: true },
    { id: 'Character Asset Library', purpose: 'Hidden source prototypes for runtime instancing.', render: false }
  ],
  glb: 'lunar-city/lunar-city-baseline.glb',
  generated3dBoardGlb: 'lunar-city/generated-3d/lunar-city-generated-assets-board.glb',
  generated3dBoardPreview: 'lunar-city/generated-3d/lunar-city-generated-assets-board.png',
  generated3dManifest: 'lunar-city/generated-3d/generated-assets-metadata.json',
  heroAssetGlb: 'lunar-city/hero-assets/lunar-city-hero-assets.glb',
  heroAssetManifest: 'lunar-city/hero-assets/hero-assets-manifest.json',
  heroAssetPreview: 'lunar-city/hero-assets/lunar-city-hero-assets.png',
  masterAssetManifest: 'lunar-city/master-assets/master-asset-manifest.json',
  masterAssetMaskManifest: 'lunar-city/master-assets/masks/mask-manifest.json',
  masterAssetMaskReviewPreview: 'lunar-city/master-assets/masks/mask-review-contact-sheet.png',
  masterAssetRejectedCandidates: 'lunar-city/master-assets/rejected-candidates.json',
  profileManifest: 'lunar-city/profile-assets.json',
  productionAssetPipeline: {
    highPolyMasterFirst: true,
    masking: 'required_before_image_to_3d_generation',
    productionSource: 'full_resolution_high_poly_master_assets',
    rejectedSources: ['raw_scene_crop_image_to_3d'],
    retopology: 'derive_smart_low_poly_lods_from_master',
    textureBake: 'bake_2k_default_4k_hero_pbr_from_master'
  },
  renderProfile: 'desktop-interactive',
  scene: 'lunar-city-baseline',
  schemaVersion: 2,
  source: 'procedural-blender-retopology-baseline',
  textures: [
    { id: 'lunar-regolith', maxResolution: '2k', slots: pbrSlots('lunar-regolith', '2k') },
    { id: 'road-composite', maxResolution: '2k', slots: pbrSlots('road-composite', '2k') },
    ...BUILDINGS.map(([, , textureSet]) => ({ id: textureSet, maxResolution: '4k' as const, slots: pbrSlots(textureSet, '4k') })),
    { id: 'leader-suits', maxResolution: '4k', slots: pbrSlots('leader-suits', '4k') },
    { id: 'worker-suits', maxResolution: '2k', slots: pbrSlots('worker-suits', '2k') },
    { id: 'child-suits', maxResolution: '2k', slots: pbrSlots('child-suits', '2k') },
    { id: 'dispatcher-glass', maxResolution: '2k', slots: pbrSlots('dispatcher-glass', '2k') },
    { id: 'garden-biome', maxResolution: '2k', slots: pbrSlots('garden-biome', '2k') }
  ],
  units: 'meters',
  validation: {
    blenderScript: 'scripts/generate_lunar_city_baseline.py',
    metadata: 'lunar-city/lunar-city-scene-metadata.json',
    requires: [
      'terrain_anchors_valid',
      'roads_conform_to_terrain',
      'buildings_do_not_overlap',
      'buildings_touch_ground',
      'collections_present',
      'lods_present',
      'textures_declared'
    ]
  }
}
