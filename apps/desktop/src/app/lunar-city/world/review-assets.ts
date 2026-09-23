import type {
  BabylonNodeLike,
  BabylonSceneLike,
  BabylonShadowGeneratorLike,
  EntityKey,
  LeaderAnimationState,
  LeaderId,
  LunarCityWorldModules,
  ReviewCharacterAsset,
  Vec3
} from '../model'

interface ReviewAnimationGroup {
  name: string
  isPlaying?: boolean
  hasMotion?: boolean
  targetedAnimations?: readonly { target?: unknown; animation: { getKeys(): readonly { value: unknown }[] } }[]
  start?(loop?: boolean): void
  stop?(): void
}

const CHAT_STATES: readonly LeaderAnimationState[] = [
  'acknowledging',
  'idle',
  'listening',
  'talking',
  'thinking',
  'unavailable',
  'walk'
]

function keyComponents(value: unknown): readonly number[] | undefined {
  if (typeof value === 'number') {
    return [value]
  }

  if (value && typeof value === 'object' && 'asArray' in value && typeof value.asArray === 'function') {
    return value.asArray()
  }

  return Array.isArray(value) ? value : undefined
}

/** Constant exported tracks are held poses and do not need a permanent frame loop. */
export function importedClipHasMotion(
  group: ReviewAnimationGroup,
  skinTargets?: ReadonlySet<unknown>
): boolean | undefined {
  if (!group.targetedAnimations?.length) {
    return undefined
  }

  let unknown = false

  for (const { animation, target } of group.targetedAnimations) {
    if (skinTargets && !skinTargets.has(target)) {continue}

    const keys = animation.getKeys(),
      first = keyComponents(keys[0]?.value)

    if (!first) {
      unknown = true

      continue
    }

    for (const key of keys.slice(1)) {
      const values = keyComponents(key.value)

      if (!values) {
        unknown = true

        continue
      }

      if (values.some((value, index) => Math.abs(value - first[index]!) > 1e-6)) {
        return true
      }
    }
  }

  return unknown ? undefined : false
}

/** The imported groups are the only source of animation availability. */
export function reviewAnimationGroups(
  id: string,
  imported: readonly unknown[],
  states: readonly string[] = CHAT_STATES,
  skinTargets?: ReadonlySet<unknown>
) {
  const groups = new Map<string, ReviewAnimationGroup>()
  const stateClips: Record<string, string> = {}

  for (const value of imported) {
    if (value && typeof value === 'object' && 'stop' in value && typeof value.stop === 'function') {
      value.stop()
    }
  }

  for (const state of states) {
    const group = imported.find(
      value =>
        value &&
        typeof value === 'object' &&
        'name' in value &&
        'start' in value &&
        typeof value.start === 'function' &&
        (value.name === `leader:${id}:${state}` || value.name === state)
    ) as ReviewAnimationGroup | undefined

    if (group) {
      group.hasMotion = importedClipHasMotion(group, skinTargets)
      groups.set(state, group)
      stateClips[state] = group.name
    }
  }

  return { groups, stateClips: Object.freeze(stateClips) }
}

interface SkinBone {
  getIndex(): number
  getTransformNode(): { parent?: unknown } | null
  getParent(): SkinBone | null
}
interface SkinMesh {
  skeleton?: { bones: SkinBone[] }
  getVerticesData?(kind: string): ArrayLike<number> | null
}

/** Unweighted control bones cannot deform the displayed skin. Include their weighted descendants' ancestors. */
export function weightedSkinTargets(meshes: readonly SkinMesh[]): ReadonlySet<unknown> | undefined {
  const targets = new Set<unknown>()
  let hasSkin = false

  for (const mesh of meshes) {
    if (!mesh.skeleton || !mesh.getVerticesData) {continue}
    hasSkin = true

    for (const suffix of ['', 'Extra']) {
      const indices = mesh.getVerticesData(`matricesIndices${suffix}`)
      const weights = mesh.getVerticesData(`matricesWeights${suffix}`)

      if (!indices || !weights) {continue}
      const used = new Set<number>()

      for (let i = 0; i < weights.length; i++) {if (weights[i]! > 1e-6) {used.add(indices[i]!)}}

      for (const bone of mesh.skeleton.bones) {
        if (!used.has(bone.getIndex())) {continue}

        for (let ancestor: SkinBone | null = bone; ancestor; ancestor = ancestor.getParent()) {
          const node = ancestor.getTransformNode()

          if (node) {targets.add(node)}
        }

        let node: { parent?: unknown } | null = bone.getTransformNode()

        while (node) {
          targets.add(node)
          node = node.parent as { parent?: unknown } | null
        }
      }
    }
  }

  return hasSkin ? targets : undefined
}

/** Review geometry keeps its authored metre scale and only its real animation groups. */
export async function importReviewLeader(
  asset: ReviewCharacterAsset,
  scene: BabylonSceneLike,
  modules: LunarCityWorldModules,
  resolveAssetUrl: (uri: string) => string,
  shadows?: BabylonShadowGeneratorLike,
  kind: 'leader' | 'review-worker' = 'leader',
  states?: readonly string[]
) {
  const result = await modules.ImportMeshAsync(resolveAssetUrl(asset.uri), scene)
  const root = new modules.TransformNode(`review-leader:${asset.id}`, scene)
  root.position?.set(asset.position.x, asset.position.y, asset.position.z)
  root.rotation?.set(0, asset.rotationY, 0)
  const nodes: readonly BabylonNodeLike[] = [...new Set([...result.meshes, ...result.transformNodes])]

  for (const node of nodes) {
    if (!node.parent) {
      node.parent = root
    }
  }

  const anchor: Vec3 = { ...asset.position, y: asset.position.y + asset.heightMetres * 0.55 }
  const focusEntityKey = `lunar-city:${kind}:${asset.id}` as EntityKey

  const { groups, stateClips } = reviewAnimationGroups(
    asset.id,
    result.animationGroups,
    states,
    weightedSkinTargets(result.meshes as readonly SkinMesh[])
  )

  const identity =
    kind === 'leader'
      ? { kind, leaderId: asset.id as LeaderId, modelId: 'leaders' as const }
      : { kind, workerReviewId: asset.id, modelId: 'workers' as const }

  for (const node of nodes) {
    node.metadata = {
      ...node.metadata,
      lunarCity: {
        cameraAnchor: anchor,
        focusEntityKey,
        ...identity,
        occlusionGroup: 'leaders',
        selectable: true,
        stateClips
      }
    }
  }

  for (const mesh of result.meshes) {
    mesh.isPickable = true
    mesh.receiveShadows = true
    shadows?.getShadowMap?.()?.renderList?.push(mesh)
  }

  return { anchor, focusEntityKey, root, stateClips, groups, result }
}

/** Project into overlay percentages using the same view-projection matrix as the scene. */
export function projectWorldPoint(point: Vec3, m?: ArrayLike<number>) {
  if (!m) {
    return undefined
  }

  const w = point.x * m[3]! + point.y * m[7]! + point.z * m[11]! + m[15]!

  if (w <= 0) {
    return { x: 0, y: 0, visible: false }
  }

  const x = (point.x * m[0]! + point.y * m[4]! + point.z * m[8]! + m[12]!) / w
  const y = (point.x * m[1]! + point.y * m[5]! + point.z * m[9]! + m[13]!) / w
  const z = (point.x * m[2]! + point.y * m[6]! + point.z * m[10]! + m[14]!) / w

  return { x: (x + 1) * 50, y: (1 - y) * 50, visible: Math.abs(x) <= 1 && Math.abs(y) <= 1 && z >= 0 && z <= 1 }
}
