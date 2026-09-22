import type {
  CameraLandmark,
  CharacterAssetManifest,
  ExternalDecorationManifestEntry,
  LeaderId,
  MaterialManifestEntry,
  ModelManifestEntry,
  ModelStatistics,
  NavigationManifest,
  ProjectSlotManifestEntry,
  QualityBudget,
  TextureManifestEntry,
  Vec3,
  WorldBounds,
  WorldManifestV2
} from './model'

const LEADER_IDS = new Set([
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
])

const LOD_REPRESENTATIONS = new Map([
  ['near', ['full', true]],
  ['mid', ['reduced', false]],
  ['far', ['static-or-aggregate', false]]
] as const)

const CHARACTER_GROUPS = new Set([
  'Acceptance & Release',
  'Archive and Acquisition',
  'Arts Studio',
  'CI Repair Triage',
  'Community Intake',
  'Content Studio',
  'Control Plane Incidents',
  'Core Runtime & UX Repairs',
  'Data & Performance Repairs',
  'Editorial Desk',
  'Engineering Guild',
  'Federation Council',
  'Knowledge Commons',
  'Memory Stewardship',
  'Operations and Release',
  'PR Merge Train',
  'Research Lab',
  'Research Review Board',
  'Upstream Hermes Maintenance'
])

export const APPROVED_SOURCE_SHA256 = '248e8d40946b08b9f74f4b2ddd0ba17e4f17fd054260189972164c5d6ca70590'
const APPROVED_SOURCE_FILENAME = 'moon-settlement-approved.jpg'
const MODEL_URI = /^models\/[a-z0-9][a-z0-9._-]*\.glb$/i
const TEXTURE_URI = /^textures\/[a-z0-9][a-z0-9._-]*\.(?:jpe?g|ktx2|png|webp)$/i

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${path} must be an object`)
  }

  return value as Record<string, unknown>
}

function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array`)
  }

  return value
}

function string(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`)
  }

  return value
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${path} must be a finite number`)
  }

  return value
}

function natural(value: unknown, path: string): number {
  const parsed = finite(value, path)

  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${path} must be a non-negative integer`)
  }

  return parsed
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${path} must be a boolean`)
  }

  return value
}

function strings(value: unknown, path: string): readonly string[] {
  return array(value, path).map((item, index) => string(item, `${path}[${index}]`))
}

function vec3(value: unknown, path: string): Vec3 {
  const items = array(value, path)

  if (items.length !== 3) {
    throw new Error(`${path} must contain exactly three values`)
  }

  return {
    x: finite(items[0], `${path}[0]`),
    y: finite(items[1], `${path}[1]`),
    z: finite(items[2], `${path}[2]`)
  }
}

function bounds(value: unknown, path: string): WorldBounds {
  const item = record(value, path)

  return { min: vec3(item.min, `${path}.min`), max: vec3(item.max, `${path}.max`) }
}

function decodedUri(value: string): string {
  let result = value

  for (let index = 0; index < 3; index += 1) {
    try {
      const decoded = decodeURIComponent(result)

      if (decoded === result) {
        break
      }

      result = decoded
    } catch {
      break
    }
  }

  return result
}

function isApprovedSourceUri(value: string): boolean {
  const normalized = decodedUri(value).replaceAll('\\', '/').toLowerCase()
  const pathname = normalized.split(/[?#]/, 1)[0] ?? normalized

  return pathname.endsWith(`/${APPROVED_SOURCE_FILENAME}`) || pathname === APPROVED_SOURCE_FILENAME
}

function scanRuntimeUris(value: unknown, path = 'manifest'): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => scanRuntimeUris(item, `${path}[${index}]`))

    return
  }

  if (!value || typeof value !== 'object') {
    return
  }

  for (const [key, item] of Object.entries(value)) {
    const itemPath = `${path}.${key}`

    if (key.toLowerCase().endsWith('uri') && typeof item === 'string' && isApprovedSourceUri(item)) {
      throw new Error(`approved source cannot be a runtime asset (${itemPath})`)
    }

    scanRuntimeUris(item, itemPath)
  }
}

function runtimeAssetUri(value: unknown, path: string, kind: 'model' | 'texture'): string {
  const uri = string(value, path)

  if (isApprovedSourceUri(uri)) {
    throw new Error('approved source cannot be a runtime asset')
  }

  const pattern = kind === 'model' ? MODEL_URI : TEXTURE_URI

  if (!pattern.test(uri) || uri.includes('..') || uri.includes('\\') || /^[a-z][a-z\d+.-]*:/i.test(uri)) {
    throw new Error(`${path} must be a relative v2 ${kind} asset`)
  }

  return uri
}

function statistics(value: unknown, path: string): ModelStatistics {
  const item = record(value, path)
  const budget = record(item.budget, `${path}.budget`)
  const extent = array(item.extent, `${path}.extent`)

  if (extent.length !== 3) {
    throw new Error(`${path}.extent must contain exactly three values`)
  }

  return {
    animationClips: strings(item.animationClips, `${path}.animationClips`),
    drawCalls: natural(item.drawCalls, `${path}.drawCalls`),
    materials: natural(item.materials, `${path}.materials`),
    meshes: natural(item.meshes, `${path}.meshes`),
    nodes: natural(item.nodes, `${path}.nodes`),
    textures: natural(item.textures, `${path}.textures`),
    triangles: natural(item.triangles, `${path}.triangles`),
    budget: {
      maxDrawCalls: natural(budget.maxDrawCalls, `${path}.budget.maxDrawCalls`),
      maxGpuMiB: finite(budget.maxGpuMiB, `${path}.budget.maxGpuMiB`),
      maxMaterials: natural(budget.maxMaterials, `${path}.budget.maxMaterials`),
      maxTextures: natural(budget.maxTextures, `${path}.budget.maxTextures`),
      maxTriangles: natural(budget.maxTriangles, `${path}.budget.maxTriangles`)
    },
    bytes: natural(item.bytes, `${path}.bytes`),
    extent: [
      finite(extent[0], `${path}.extent[0]`),
      finite(extent[1], `${path}.extent[1]`),
      finite(extent[2], `${path}.extent[2]`)
    ],
    gpuMiB: finite(item.gpuMiB, `${path}.gpuMiB`),
    sha256: string(item.sha256, `${path}.sha256`)
  }
}

function modelEntry(value: unknown, index: number): ModelManifestEntry {
  const path = `models[${index}]`
  const item = record(value, path)
  const transform = record(item.transform, `${path}.transform`)
  const anchorsValue = record(item.anchors, `${path}.anchors`)

  const anchors = Object.fromEntries(
    Object.entries(anchorsValue).map(([name, anchor]) => [name, vec3(anchor, `${path}.anchors.${name}`)])
  )

  const collision = record(item.collision, `${path}.collision`)
  const instancingValue = item.instancing === undefined ? undefined : record(item.instancing, `${path}.instancing`)

  const lods = array(item.lods, `${path}.lods`)
    .map((lod, lodIndex) => {
      const entry = record(lod, `${path}.lods[${lodIndex}]`)

      return {
        distance: finite(entry.distance, `${path}.lods[${lodIndex}].distance`),
        node: string(entry.node, `${path}.lods[${lodIndex}].node`)
      }
    })
    .sort((left, right) => left.distance - right.distance)

  if (new Set(lods.map(lod => lod.distance)).size !== lods.length) {
    throw new Error(`${path}.lods LOD distances must be distinct`)
  }

  return {
    id: string(item.id, `${path}.id`),
    uri: runtimeAssetUri(item.uri, `${path}.uri`, 'model'),
    maxTriangles: natural(item.maxTriangles, `${path}.maxTriangles`),
    maxDrawCalls: natural(item.maxDrawCalls, `${path}.maxDrawCalls`),
    maxMaterials: natural(item.maxMaterials, `${path}.maxMaterials`),
    maxTextures: natural(item.maxTextures, `${path}.maxTextures`),
    maxGpuMiB: finite(item.maxGpuMiB, `${path}.maxGpuMiB`),
    requiredNodes: strings(item.requiredNodes, `${path}.requiredNodes`),
    requiredClips: strings(item.requiredClips, `${path}.requiredClips`),
    lods: Object.freeze(lods),
    transform: {
      position: vec3(transform.position, `${path}.transform.position`),
      rotation: vec3(transform.rotation, `${path}.transform.rotation`),
      scale: vec3(transform.scale, `${path}.transform.scale`)
    },
    pivot: vec3(item.pivot, `${path}.pivot`),
    bounds: bounds(item.bounds, `${path}.bounds`),
    anchors,
    cameraAnchor: vec3(item.cameraAnchor, `${path}.cameraAnchor`),
    occlusionGroup: string(item.occlusionGroup, `${path}.occlusionGroup`),
    collision: {
      kind: string(collision.kind, `${path}.collision.kind`),
      navigationArea: string(collision.navigationArea, `${path}.collision.navigationArea`)
    },
    materialSlots: strings(item.materialSlots, `${path}.materialSlots`),
    ...(instancingValue
      ? {
          instancing: {
            eligible: boolean(instancingValue.eligible, `${path}.instancing.eligible`),
            variants: strings(instancingValue.variants, `${path}.instancing.variants`)
          }
        }
      : {}),
    statistics: statistics(item.statistics, `${path}.statistics`)
  }
}

function cameraLandmark(value: unknown, path: string): CameraLandmark {
  const item = record(value, path)

  return {
    id: string(item.id, `${path}.id`),
    alpha: finite(item.alpha, `${path}.alpha`),
    beta: finite(item.beta, `${path}.beta`),
    radius: finite(item.radius, `${path}.radius`),
    target: vec3(item.target, `${path}.target`),
    minBeta: finite(item.minBeta, `${path}.minBeta`),
    maxBeta: finite(item.maxBeta, `${path}.maxBeta`),
    minRadius: finite(item.minRadius, `${path}.minRadius`),
    maxRadius: finite(item.maxRadius, `${path}.maxRadius`)
  }
}

function navigation(value: unknown): NavigationManifest {
  const item = record(value, 'navigation')

  return {
    ...(item.colliders ? { colliders: array(item.colliders, 'navigation.colliders').map((value, index) => {
      const path = `navigation.colliders[${index}]`, collider = record(value, path)
      const id = string(collider.id, `${path}.id`), center = vec3(collider.center, `${path}.center`)

      if (collider.kind === 'box') {
        const halfExtents = vec3(collider.halfExtents, `${path}.halfExtents`)

        if (Object.values(halfExtents).some(value => value <= 0)) {throw Error(`${path} must have positive extents`)}

        return { id, kind: 'box' as const, center, halfExtents, rotationY: finite(collider.rotationY, `${path}.rotationY`) }
      }

      if (collider.kind !== 'cylinder') {throw Error(`${path} has unsupported collider kind`)}
      const radius = finite(collider.radius, `${path}.radius`), height = finite(collider.height, `${path}.height`)

      if (radius <= 0 || height <= 0) {throw Error(`${path} must have positive dimensions`)}

      return { id, kind: 'cylinder' as const, center, radius, height }
    }) } : {}),
    meshUri: runtimeAssetUri(item.meshUri, 'navigation.meshUri', 'model'),
    ...(item.sha256 ? { sha256: string(item.sha256, 'navigation.sha256') } : {}),
    areas: strings(item.areas, 'navigation.areas'),
    links: array(item.links, 'navigation.links').map((link, index) => {
      const entry = record(link, `navigation.links[${index}]`)

      return {
        from: vec3(entry.from, `navigation.links[${index}].from`),
        to: vec3(entry.to, `navigation.links[${index}].to`),
        bidirectional: boolean(entry.bidirectional, `navigation.links[${index}].bidirectional`)
      }
    })
  }
}

function texture(value: unknown, index: number): TextureManifestEntry {
  const path = `textures[${index}]`
  const item = record(value, path)

  return {
    bytes: natural(item.bytes, `${path}.bytes`),
    sha256: string(item.sha256, `${path}.sha256`),
    source: string(item.source, `${path}.source`),
    uri: runtimeAssetUri(item.uri, `${path}.uri`, 'texture')
  }
}

function externalDecoration(value: unknown, index: number): ExternalDecorationManifestEntry {
  const path = `externalDecorations[${index}]`
  const item = record(value, path)

  return {
    id: string(item.id, `${path}.id`),
    uri: runtimeAssetUri(item.uri, `${path}.uri`, 'model'),
    position: vec3(item.position, `${path}.position`),
    rotation: vec3(item.rotation, `${path}.rotation`),
    scale: vec3(item.scale, `${path}.scale`)
  }
}

function material(value: unknown, index: number): MaterialManifestEntry {
  const path = `materials[${index}]`
  const item = record(value, path)

  return {
    id: string(item.id, `${path}.id`),
    palette: string(item.palette, `${path}.palette`),
    maxTextures: natural(item.maxTextures, `${path}.maxTextures`)
  }
}

function projectSlot(value: unknown, index: number): ProjectSlotManifestEntry {
  const path = `projectSlots[${index}]`
  const item = record(value, path)
  const link = record(item.navigationLink, `${path}.navigationLink`)

  return {
    id: string(item.id, `${path}.id`),
    position: vec3(item.position, `${path}.position`),
    bounds: bounds(item.bounds, `${path}.bounds`),
    navigationLink: {
      from: vec3(link.from, `${path}.navigationLink.from`),
      to: vec3(link.to, `${path}.navigationLink.to`),
      bidirectional: boolean(link.bidirectional, `${path}.navigationLink.bidirectional`)
    }
  }
}

function qualityBudget(value: unknown, path: string): QualityBudget {
  const item = record(value, path)

  return {
    drawCalls: natural(item.drawCalls, `${path}.drawCalls`),
    visibleTriangles: natural(item.visibleTriangles, `${path}.visibleTriangles`),
    gpuMiB: finite(item.gpuMiB, `${path}.gpuMiB`)
  }
}

function characterAssets(value: unknown): CharacterAssetManifest {
  const path = 'characterAssets'
  const item = record(value, path)
  const vocabularyValue = record(item.workerVocabulary, `${path}.workerVocabulary`)

  const vocabulary = {
    bodies: strings(vocabularyValue.bodies, `${path}.workerVocabulary.bodies`),
    emblems: strings(vocabularyValue.emblems, `${path}.workerVocabulary.emblems`),
    heads: strings(vocabularyValue.heads, `${path}.workerVocabulary.heads`),
    palettes: strings(vocabularyValue.palettes, `${path}.workerVocabulary.palettes`),
    silhouetteAccessories: strings(
      vocabularyValue.silhouetteAccessories,
      `${path}.workerVocabulary.silhouetteAccessories`
    )
  }

  const vocabularySets = {
    body: new Set(vocabulary.bodies),
    emblem: new Set(vocabulary.emblems),
    head: new Set(vocabulary.heads),
    palette: new Set(vocabulary.palettes),
    silhouetteAccessory: new Set(vocabulary.silhouetteAccessories)
  }

  const physicalRootsValue = record(item.physicalVariantRoots, `${path}.physicalVariantRoots`)

  const physicalRoots = (field: 'body' | 'head' | 'palette', ids: readonly string[]) => {
    const entries = record(physicalRootsValue[field], `${path}.physicalVariantRoots.${field}`)

    const result = Object.fromEntries(
      Object.entries(entries).map(([id, node]) => {
        if (!ids.includes(id)) {
          throw new Error(`${path}.physicalVariantRoots.${field}.${id} is not declared in workerVocabulary`)
        }

        const nodeName = string(node, `${path}.physicalVariantRoots.${field}.${id}`)

        if (!/^worker:[a-z0-9][a-z0-9:-]*$/u.test(nodeName)) {
          throw new Error(`${path}.physicalVariantRoots.${field}.${id} must be a worker node name`)
        }

        return [id, nodeName]
      })
    )

    if (Object.keys(result).length !== ids.length || new Set(Object.values(result)).size !== ids.length) {
      throw new Error(`${path}.physicalVariantRoots.${field} must map every vocabulary id exactly once`)
    }

    return result
  }

  const leaders = array(item.leaders, `${path}.leaders`).map((value, index) => {
    const leader = record(value, `${path}.leaders[${index}]`)
    const id = string(leader.id, `${path}.leaders[${index}].id`)

    if (!LEADER_IDS.has(id)) {
      throw new Error(`${path}.leaders[${index}].id is not a declared leader`)
    }

    return {
      id: id as CharacterAssetManifest['leaders'][number]['id'],
      ...(leader.heightMetres === undefined
        ? {}
        : { heightMetres: finite(leader.heightMetres, `${path}.leaders[${index}].heightMetres`) }),
      silhouetteId: string(leader.silhouetteId, `${path}.leaders[${index}].silhouetteId`),
      species: string(leader.species, `${path}.leaders[${index}].species`),
      visualId: string(leader.visualId, `${path}.leaders[${index}].visualId`)
    }
  })

  for (const field of ['id', 'silhouetteId', 'species', 'visualId'] as const) {
    if (new Set(leaders.map(leader => leader[field])).size !== leaders.length) {
      throw new Error(`${path}.leaders must have distinct ${field}`)
    }
  }

  if (leaders.length === 0) {
    throw new Error(`${path}.leaders must contain at least one leader`)
  }

  const groupKits = array(item.groupKits, `${path}.groupKits`).map((value, index) => {
    const kitPath = `${path}.groupKits[${index}]`
    const kit = record(value, kitPath)
    const signatureValue = record(kit.signature, `${kitPath}.signature`)

    const signature = {
      body: string(signatureValue.body, `${kitPath}.signature.body`),
      emblem: string(signatureValue.emblem, `${kitPath}.signature.emblem`),
      head: string(signatureValue.head, `${kitPath}.signature.head`),
      palette: string(signatureValue.palette, `${kitPath}.signature.palette`),
      silhouetteAccessory: string(signatureValue.silhouetteAccessory, `${kitPath}.signature.silhouetteAccessory`)
    }

    for (const [field, declared] of Object.entries(vocabularySets)) {
      if (!declared.has(signature[field as keyof typeof signature])) {
        throw new Error(`${kitPath}.signature.${field} is not declared in ${path}.workerVocabulary`)
      }
    }

    const group = string(kit.group, `${kitPath}.group`)

    if (!CHARACTER_GROUPS.has(group)) {
      throw new Error(`${kitPath}.group is not an exact configured Hermes group`)
    }

    return {
      group,
      kitId: string(kit.kitId, `${kitPath}.kitId`),
      signature
    }
  })

  if (
    groupKits.length !== 19 ||
    new Set(groupKits.map(kit => kit.group)).size !== CHARACTER_GROUPS.size ||
    new Set(groupKits.map(kit => kit.kitId)).size !== groupKits.length
  ) {
    throw new Error(`${path}.groupKits must contain 19 distinct kit ids`)
  }

  const completeSignatures = groupKits.map(kit => JSON.stringify(kit.signature))

  if (new Set(completeSignatures).size !== completeSignatures.length) {
    throw new Error(`${path}.groupKits characterAssets signatures must be distinct`)
  }

  const strategyValue = record(item.sharedResourceStrategy, `${path}.sharedResourceStrategy`)
  const perProfileValue = record(strategyValue.perProfile, `${path}.sharedResourceStrategy.perProfile`)

  const zero = (field: string): 0 => {
    const value = natural(perProfileValue[field], `${path}.sharedResourceStrategy.perProfile.${field}`)

    if (value !== 0) {
      throw new Error(`${path}.sharedResourceStrategy.perProfile.${field} must equal 0`)
    }

    return 0
  }

  const shared = (field: string): 'shared' => {
    if (strategyValue[field] !== 'shared') {
      throw new Error(`${path}.sharedResourceStrategy.${field} must equal shared`)
    }

    return 'shared'
  }

  const lodRepresentations = array(item.lodRepresentations, `${path}.lodRepresentations`).map((value, index) => {
    const lodPath = `${path}.lodRepresentations[${index}]`
    const lod = record(value, lodPath)
    const id = string(lod.id, `${lodPath}.id`)
    const expected = LOD_REPRESENTATIONS.get(id as 'near' | 'mid' | 'far')
    const representation = string(lod.representation, `${lodPath}.representation`)
    const animated = boolean(lod.animated, `${lodPath}.animated`)

    if (!expected || expected[0] !== representation || expected[1] !== animated) {
      throw new Error(`${lodPath} is not a supported low-power representation`)
    }

    return {
      animated,
      id: id as 'near' | 'mid' | 'far',
      representation: representation as 'full' | 'reduced' | 'static-or-aggregate'
    }
  })

  if (new Set(lodRepresentations.map(lod => lod.id)).size !== 3) {
    throw new Error(`${path}.lodRepresentations must declare near, mid, and far exactly once`)
  }

  const fleetIdentityFloor = natural(item.fleetIdentityFloor, `${path}.fleetIdentityFloor`)

  if (fleetIdentityFloor < 128) {
    throw new Error(`${path}.fleetIdentityFloor must cover at least 128 exact profiles`)
  }

  const bodyRoots = physicalRoots('body', vocabulary.bodies)
  const headRoots = physicalRoots('head', vocabulary.heads)
  const paletteRoots = physicalRoots('palette', vocabulary.palettes)

  const requiredActivationRoots = [
    ...Object.values(bodyRoots),
    ...Object.values(headRoots),
    ...Object.values(paletteRoots),
    ...groupKits.map(kit => `worker:group-kit:${kit.kitId}`)
  ]

  const activationScaleValue = record(
    physicalRootsValue.activationScale,
    `${path}.physicalVariantRoots.activationScale`
  )

  const activationScale = Object.fromEntries(
    Object.entries(activationScaleValue).map(([node, value]) => {
      const scale = vec3(value, `${path}.physicalVariantRoots.activationScale.${node}`)

      if (scale.x <= 0 || scale.y <= 0 || scale.z <= 0) {
        throw new Error(`${path}.physicalVariantRoots.activationScale.${node} must be nonzero on every axis`)
      }

      return [node, Object.freeze(scale)]
    })
  )

  if (
    Object.keys(activationScale).length !== requiredActivationRoots.length ||
    requiredActivationRoots.some(node => activationScale[node] === undefined)
  ) {
    throw new Error(
      `${path}.physicalVariantRoots.activationScale must declare every physical variant root exactly once`
    )
  }

  return {
    fleetIdentityFloor,
    groupKits,
    leaders,
    lodRepresentations,
    physicalVariantRoots: {
      activationScale: Object.freeze(activationScale),
      body: bodyRoots,
      groupKit: {
        emblemSuffix: string(
          record(physicalRootsValue.groupKit, `${path}.physicalVariantRoots.groupKit`).emblemSuffix,
          `${path}.physicalVariantRoots.groupKit.emblemSuffix`
        ),
        identityAccentSuffix: string(
          record(physicalRootsValue.groupKit, `${path}.physicalVariantRoots.groupKit`).identityAccentSuffix,
          `${path}.physicalVariantRoots.groupKit.identityAccentSuffix`
        ),
        silhouetteSuffix: string(
          record(physicalRootsValue.groupKit, `${path}.physicalVariantRoots.groupKit`).silhouetteSuffix,
          `${path}.physicalVariantRoots.groupKit.silhouetteSuffix`
        )
      },
      head: headRoots,
      palette: paletteRoots
    },
    sharedResourceStrategy: {
      animationClips: shared('animationClips'),
      gpuBuffers: shared('gpuBuffers'),
      materials: shared('materials'),
      perProfile: {
        materials: zero('materials'),
        meshes: zero('meshes'),
        skeletons: zero('skeletons'),
        textures: zero('textures')
      },
      rig: string(strategyValue.rig, `${path}.sharedResourceStrategy.rig`),
      textureAtlas: runtimeAssetUri(
        strategyValue.textureAtlas,
        `${path}.sharedResourceStrategy.textureAtlas`,
        'texture'
      )
    },
    workerVocabulary: vocabulary
  }
}

export function assertWorldManifestRuntimeAssets(
  manifest: Pick<WorldManifestV2, 'models' | 'navigation' | 'textures'>
): void {
  scanRuntimeUris(manifest)
  manifest.models.forEach((model, index) => runtimeAssetUri(model.uri, `models[${index}].uri`, 'model'))
  runtimeAssetUri(manifest.navigation.meshUri, 'navigation.meshUri', 'model')
  manifest.textures.forEach((entry, index) => runtimeAssetUri(entry.uri, `textures[${index}].uri`, 'texture'))
}

export function parseWorldManifest(value: unknown): WorldManifestV2 {
  scanRuntimeUris(value)
  const root = record(value, 'manifest')

  if (root.version !== 2) {
    throw new Error('version must equal 2')
  }

  if (root.assetVersion !== '2.0.0') {
    throw new Error('assetVersion must equal 2.0.0')
  }

  const source = record(root.source, 'source')

  if (source.sha256 !== APPROVED_SOURCE_SHA256) {
    throw new Error('approved source digest mismatch')
  }

  const models = array(root.models, 'models').map(modelEntry)
  const modelIds = new Set<string>()

  for (const model of models) {
    if (modelIds.has(model.id)) {
      throw new Error(`model id ${model.id} is duplicated`)
    }

    modelIds.add(model.id)
  }

  const cameraValue = record(root.camera, 'camera')
  const destinationsValue = record(root.destinations, 'destinations')
  const budgets = record(root.qualityBudgets, 'qualityBudgets')

  const externalDecorations =
    root.externalDecorations === undefined
      ? undefined
      : array(root.externalDecorations, 'externalDecorations').map(externalDecoration)

  if (externalDecorations) {
    const decorationIds = new Set<string>()

    for (const decoration of externalDecorations) {
      if (decorationIds.has(decoration.id)) {
        throw new Error(`external decoration id ${decoration.id} is duplicated`)
      }

      decorationIds.add(decoration.id)
    }
  }

  const reviewLeaderAssets =
    root.reviewLeaderAssets === undefined
      ? undefined
      : array(root.reviewLeaderAssets, 'reviewLeaderAssets').map((value, index) => {
          const entry = record(value, `reviewLeaderAssets[${index}]`)
          const id = string(entry.id, 'reviewLeaderAssets.id')

          if (!LEADER_IDS.has(id)) {
            throw new Error('reviewLeaderAssets has an unknown leader')
          }

          const heightMetres = finite(entry.heightMetres, 'reviewLeaderAssets.heightMetres')

          if (heightMetres <= 0) {
            throw new Error('reviewLeaderAssets height must be positive')
          }

          return {
            id: id as LeaderId,
            uri: runtimeAssetUri(entry.uri, 'reviewLeaderAssets.uri', 'model'),
            position: vec3(entry.position, 'reviewLeaderAssets.position'),
            rotationY: finite(entry.rotationY, 'reviewLeaderAssets.rotationY'),
            heightMetres
          }
        })

  if (reviewLeaderAssets && new Set(reviewLeaderAssets.map(asset => asset.id)).size !== reviewLeaderAssets.length) {
    throw new Error('Duplicate review leader')
  }

  const reviewWorkerAssets =
    root.reviewWorkerAssets === undefined
      ? undefined
      : array(root.reviewWorkerAssets, 'reviewWorkerAssets').map((value, index) => {
          const entry = record(value, `reviewWorkerAssets[${index}]`)
          const id = string(entry.id, 'reviewWorkerAssets.id')

          const heightMetres = finite(entry.heightMetres, 'reviewWorkerAssets.heightMetres')

          if (heightMetres <= 0) {
            throw new Error('reviewWorkerAssets height must be positive')
          }

          return {
            id,
            uri: runtimeAssetUri(entry.uri, 'reviewWorkerAssets.uri', 'model'),
            position: vec3(entry.position, 'reviewWorkerAssets.position'),
            rotationY: finite(entry.rotationY, 'reviewWorkerAssets.rotationY'),
            heightMetres
          }
        })

  if (reviewWorkerAssets && new Set(reviewWorkerAssets.map(asset => asset.id)).size !== reviewWorkerAssets.length) {
    throw new Error('Duplicate review leader')
  }

  const manifest: WorldManifestV2 = {
    ...(reviewLeaderAssets ? { reviewLeaderAssets } : {}),
    ...(reviewWorkerAssets ? { reviewWorkerAssets } : {}),
    version: 2,
    assetVersion: '2.0.0',
    source: { sha256: APPROVED_SOURCE_SHA256 },
    characterAssets: characterAssets(root.characterAssets),
    materials: array(root.materials, 'materials').map(material),
    models,
    textures: array(root.textures, 'textures').map(texture),
    ...(externalDecorations ? { externalDecorations: Object.freeze(externalDecorations) } : {}),
    camera: {
      overview: cameraLandmark(cameraValue.overview, 'camera.overview'),
      bounds: bounds(cameraValue.bounds, 'camera.bounds'),
      followOffset: vec3(cameraValue.followOffset, 'camera.followOffset')
    },
    navigation: navigation(root.navigation),
    destinations: Object.fromEntries(
      Object.entries(destinationsValue).map(([name, position]) => [name, vec3(position, `destinations.${name}`)])
    ),
    projectSlots: array(root.projectSlots, 'projectSlots').map(projectSlot),
    qualityBudgets: {
      balancedOverview: qualityBudget(budgets.balancedOverview, 'qualityBudgets.balancedOverview'),
      balancedWorkerFocus: qualityBudget(budgets.balancedWorkerFocus, 'qualityBudgets.balancedWorkerFocus')
    },
    generatedAssetPack: { ...record(root.generatedAssetPack, 'generatedAssetPack') }
  }

  assertWorldManifestRuntimeAssets(manifest)

  return manifest
}

export async function loadWorldManifest(url: string, signal?: AbortSignal): Promise<WorldManifestV2> {
  const response = await fetch(url, { signal })

  if (!response.ok) {
    throw new Error(`Lunar City manifest request failed: ${response.status} ${response.statusText}`)
  }

  return parseWorldManifest(await response.json())
}
