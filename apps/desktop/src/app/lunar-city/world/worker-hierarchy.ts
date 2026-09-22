import type { BabylonNodeLike } from '../model'
interface BoneLike {
  getTransformNode?(): BabylonNodeLike | null
  linkTransformNode?(node: BabylonNodeLike): void
}
interface SkeletonLike {
  bones: BoneLike[]
  clone(name: string): SkeletonLike
  dispose(): void
}
interface SkinnedNode extends BabylonNodeLike {
  skeleton?: SkeletonLike | null
}
interface BabylonAnimationGroupLike {
  name: string
  hasMotion?: boolean
  reset?(): void
  clone?(name: string, converter?: (target: unknown) => unknown): BabylonAnimationGroupLike | null
  start?(loop?: boolean): void
  stop?(): void
  dispose?(): void
}
interface BabylonHierarchyNodeLike extends BabylonNodeLike {
  instantiateHierarchy?(
    parent?: BabylonNodeLike | null,
    options?: unknown,
    onNewNodeCreated?: (source: unknown, clone: BabylonNodeLike) => void
  ): BabylonNodeLike | null
}

export function cloneWorkerHierarchy(
  template: BabylonNodeLike,
  parent: BabylonNodeLike,
  sourceGroups: ReadonlyMap<string, BabylonAnimationGroupLike>,
  name: string
): {
  animations: ReadonlyMap<string, BabylonAnimationGroupLike>
  nodeMap: ReadonlyMap<unknown, BabylonNodeLike>
  root: BabylonNodeLike
  disposeSkeletons(): void
} {
  const nodeMap = new Map<unknown, BabylonNodeLike>()
  const hierarchy = template as BabylonHierarchyNodeLike

  const root =
    hierarchy.instantiateHierarchy?.(parent, { doNotInstantiate: true }, (source, clone) =>
      nodeMap.set(source, clone)
    ) ??
    template.clone?.(`lunar-city:worker-clone:${name}`, parent) ??
    parent

  const skeletons = new Map<SkeletonLike, SkeletonLike>()

  for (const [source, clonedNode] of nodeMap) {
    const skeleton = (source as SkinnedNode).skeleton

    if (!skeleton) {continue}
    let copied = skeletons.get(skeleton)

    if (!copied) {
      copied = skeleton.clone(`lunar-city:worker-skeleton:${name}`)
      skeletons.set(skeleton, copied)
      skeleton.bones.forEach((bone, index) => {
        const cloneBone = copied!.bones[index]!
        nodeMap.set(bone, cloneBone as unknown as BabylonNodeLike)
        const target = bone.getTransformNode?.()

        if (target) {
          const clonedTarget = nodeMap.get(target)

          if (!clonedTarget) {throw new Error(`Worker skeleton target was not cloned: ${name}`)}
          cloneBone.linkTransformNode?.(clonedTarget)
        }
      })
    }

    ;(clonedNode as SkinnedNode).skeleton = copied
  }

  const animations = new Map<string, BabylonAnimationGroupLike>()

  for (const [name, group] of sourceGroups) {
    const cloned = group.clone?.(`lunar-city:worker:${name}`, target => nodeMap.get(target) ?? target)

    if (cloned) {
      cloned.hasMotion = group.hasMotion
      animations.set(name, cloned)
    }
  }

  return {
    animations,
    nodeMap,
    root,
    disposeSkeletons() {
      for (const skeleton of skeletons.values()) {skeleton.dispose()}
    }
  }
}

export interface HistoricalWorkerFit { scale: number; offsetY: number }
interface MeasurableNode extends BabylonNodeLike {
  computeWorldMatrix?(force?: boolean): unknown
  getWorldMatrix?(): { m: ArrayLike<number> }
  getPositionData?(applySkeleton?: boolean, applyMorph?: boolean): ArrayLike<number> | null
  getVerticesData?(kind: string): ArrayLike<number> | null
}

export function applyHistoricalWorkerFit(root: BabylonNodeLike, fit: HistoricalWorkerFit): void {
  root.scaling?.set(fit.scale, fit.scale, fit.scale)
  root.position?.set(0, fit.offsetY, 0)
}

/** Measure selected, enabled geometry in its actual rest skin pose; never use hidden variant bounds. */
export function fitHistoricalWorker(
  root: BabylonNodeLike,
  nodes: readonly BabylonNodeLike[],
  heightMetres = 1.2
): HistoricalWorkerFit | undefined {
  const wrapper = root as MeasurableNode
  root.scaling?.set(1, 1, 1)
  root.position?.set(0, 0, 0)
  wrapper.computeWorldMatrix?.(true)
  const originY = wrapper.getWorldMatrix?.().m[13] ?? 0
  let minY = Infinity, maxY = -Infinity

  for (const candidate of nodes) {
    const mesh = candidate as MeasurableNode

    if (mesh.isEnabled?.() === false) {continue}
    mesh.computeWorldMatrix?.(true)
    const matrix = mesh.getWorldMatrix?.().m
    const positions = mesh.getPositionData?.(true, true) ?? mesh.getVerticesData?.('position')

    if (!matrix || !positions) {continue}

    for (let i = 0; i < positions.length; i += 3) {
      const y = matrix[1]! * positions[i]! + matrix[5]! * positions[i + 1]! +
        matrix[9]! * positions[i + 2]! + matrix[13]! - originY

      if (!Number.isFinite(y)) {continue}
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
    }
  }

  if (!(maxY > minY)) {return undefined}
  const scale = heightMetres / (maxY - minY)
  const fit = { scale, offsetY: -minY * scale }
  applyHistoricalWorkerFit(root, fit)

  return fit
}
