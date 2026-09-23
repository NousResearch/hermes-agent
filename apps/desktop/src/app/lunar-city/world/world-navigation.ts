import type {
  BabylonImportResultLike,
  BabylonMeshLike,
  BabylonNodeLike,
  LunarCityWorldModules,
  RecastConfigLike,
  RecastRuntimeLike,
  Vec3,
  WorldBounds,
  WorldManifestV2
} from '../model'
import { RuntimeAssetIntegrityError } from '../runtime-asset-integrity'

import { createInteriorNavigation, type NavigableInteriorPlan } from './interior-navigation'
import { createMovementSafety } from './movement-safety'
import { createRecastNavigationQuery, disposeRecastWrapper, type NavigationQuery } from './navigation'

function navigationPointKey(point: Vec3): string {
  return `${point.x},${point.y},${point.z}`
}

/**
 * A fail-closed traversal boundary for the manifest's declared links. Recast
 * may replace this query when its route-local navmesh is ready, but no worker
 * is ever sent on an inferred straight line in the meantime.
 */
export function createManifestNavigationQuery(
  manifest: Pick<WorldManifestV2, 'navigation'>['navigation']
): NavigationQuery {
  const points = new Map<string, Vec3>()
  const edges = new Map<string, string[]>()

  const addEdge = (from: Vec3, to: Vec3): void => {
    const fromKey = navigationPointKey(from)
    const toKey = navigationPointKey(to)
    points.set(fromKey, { ...from })
    points.set(toKey, { ...to })
    const adjacent = edges.get(fromKey) ?? []
    adjacent.push(toKey)
    edges.set(fromKey, adjacent)
  }

  for (const link of manifest.links) {
    addEdge(link.from, link.to)

    if (link.bidirectional) {
      addEdge(link.to, link.from)
    }
  }

  return {
    computePath(from, to) {
      const start = navigationPointKey(from)
      const destination = navigationPointKey(to)

      if (!points.has(start) || !points.has(destination)) {
        return undefined
      }

      const previous = new Map<string, string | undefined>([[start, undefined]])
      const pending = [start]

      while (pending.length > 0) {
        const current = pending.shift()!

        if (current === destination) {
          break
        }

        for (const adjacent of edges.get(current) ?? []) {
          if (!previous.has(adjacent)) {
            previous.set(adjacent, current)
            pending.push(adjacent)
          }
        }
      }

      if (!previous.has(destination)) {
        return undefined
      }

      const path: Vec3[] = []
      let current: string | undefined = destination

      while (current !== undefined) {
        path.push({ ...points.get(current)! })
        current = previous.get(current)
      }

      return path.reverse()
    }
  }
}

interface NavigationGeometry {
  bounds: WorldBounds
  indices: Uint32Array
  positions: Float32Array
}

function transformNavigationVertex(mesh: BabylonMeshLike, x: number, y: number, z: number): Vec3 | undefined {
  const values = mesh.getWorldMatrix?.().m

  if (!values) {
    return { x, y, z }
  }

  if (values.length !== 16 || !values.every(Number.isFinite)) {
    return undefined
  }

  const w = values[3]! * x + values[7]! * y + values[11]! * z + values[15]!

  if (!Number.isFinite(w) || w === 0) {
    return undefined
  }

  return {
    x: (values[0]! * x + values[4]! * y + values[8]! * z + values[12]!) / w,
    y: (values[1]! * x + values[5]! * y + values[9]! * z + values[13]!) / w,
    z: (values[2]! * x + values[6]! * y + values[10]! * z + values[14]!) / w
  }
}

/** Extracts valid, indexed geometry only; malformed navigation assets fail closed. */
function navigationGeometry(result: BabylonImportResultLike): NavigationGeometry | undefined {
  const positions: number[] = []
  const indices: number[] = []

  for (const mesh of result.meshes) {
    const meshPositions = mesh.getVerticesData?.('position')
    const meshIndices = mesh.getIndices?.()

    if (!meshPositions || !meshIndices || meshPositions.length === 0 || meshPositions.length % 3 !== 0) {
      continue
    }

    const vertexCount = meshPositions.length / 3

    if (
      !Array.from(meshPositions).every(Number.isFinite) ||
      !Array.from(meshIndices).every(index => Number.isSafeInteger(index) && index >= 0 && index < vertexCount)
    ) {
      continue
    }

    const transformed = [] as Vec3[]

    for (let index = 0; index < meshPositions.length; index += 3) {
      const point = transformNavigationVertex(
        mesh,
        meshPositions[index]!,
        meshPositions[index + 1]!,
        meshPositions[index + 2]!
      )

      if (!point) {
        transformed.length = 0

        break
      }

      transformed.push(point)
    }

    if (transformed.length !== vertexCount) {
      continue
    }

    const vertexOffset = positions.length / 3

    for (const point of transformed) {
      positions.push(point.x, point.y, point.z)
    }

    indices.push(...Array.from(meshIndices, index => vertexOffset + index))
  }

  if (positions.length === 0 || indices.length === 0) {
    return undefined
  }

  const xs = positions.filter((_value, index) => index % 3 === 0)
  const ys = positions.filter((_value, index) => index % 3 === 1)
  const zs = positions.filter((_value, index) => index % 3 === 2)

  return {
    bounds: {
      min: { x: Math.min(...xs), y: Math.min(...ys), z: Math.min(...zs) },
      max: { x: Math.max(...xs), y: Math.max(...ys), z: Math.max(...zs) }
    },
    indices: Uint32Array.from(indices),
    positions: Float32Array.from(positions)
  }
}

function disposeNavigationImport(result: BabylonImportResultLike): void {
  for (const node of new Set<BabylonNodeLike>([...result.meshes, ...result.transformNodes])) {
    node.setEnabled?.(false)
    node.dispose?.()
  }
}

function recastConfig(Runtime: RecastRuntimeLike, bounds: WorldBounds): RecastConfigLike {
  const config = new Runtime.rcConfig()

  if (!config.set_bmin || !config.set_bmax) {
    throw new Error('Lunar City Recast runtime does not expose indexed bounds setters')
  }

  for (const [index, axis] of ['x', 'y', 'z'].entries() as Iterable<[number, keyof Vec3]>) {
    config.set_bmin(index, bounds.min[axis])
    config.set_bmax(index, bounds.max[axis])
  }

  // These are Recast voxelization inputs, not hand-authored city coordinates;
  // the actual navigable extent is extracted from the declared navigation GLB.
  Object.assign(config, {
    ch: 0.01, // Centimetre vertical cells keep 1.2m workers on the road surface.
    cs: 0.2,
    detailSampleDist: 6,
    detailSampleMaxError: 1,
    maxEdgeLen: 12,
    maxSimplificationError: 1.3,
    maxVertsPerPoly: 6,
    mergeRegionArea: 20,
    minRegionArea: 8,
    walkableClimb: 20, // Preserve 0.2m step clearance at 0.01m voxel height.
    walkableHeight: 200, // Preserve 2m clearance at 0.01m voxel height.
    walkableRadius: 2, // 0.4m clearance at 0.2m cell width.
    walkableSlopeAngle: 45
  })

  return config
}

/**
 * Builds a route-local Recast query from the manifest navigation GLB. The
 * imported geometry is released immediately after the navmesh has been built;
 * declared links remain a fail-closed fallback if Recast/WASM cannot start.
 */
export async function createRouteNavigationQuery(
  navigation: Pick<WorldManifestV2, 'navigation'>['navigation'],
  modules: Pick<LunarCityWorldModules, 'ImportMeshAsync' | 'createRecastNavigation'>,
  scene: Parameters<LunarCityWorldModules['ImportMeshAsync']>[1],
  resolveAssetUrl: (uri: string) => string,
  interiorPlans: readonly NavigableInteriorPlan[] = [],
  envelope?: { radius: number; height: number }
): Promise<NavigationQuery> {
  const fallback = createManifestNavigationQuery(navigation)

  const closed: NavigationQuery = { computePath: () => undefined, resolvePosition: () => undefined, canTraverse: () => false }

  let imported: BabylonImportResultLike | undefined
  let releaseNavMesh: (() => void) | undefined

  try {
    imported = await modules.ImportMeshAsync(resolveAssetUrl(navigation.meshUri), scene)
    const geometry = navigationGeometry(imported)

    if (!geometry) {
      return closed
    }

    const safety = createMovementSafety(geometry, navigation.colliders, envelope)

    if (!modules.createRecastNavigation) {return createInteriorNavigation({ ...fallback, ...safety },geometry,navigation.colliders??[],interiorPlans,envelope)}
    const Runtime = await modules.createRecastNavigation()
    const navMesh = new Runtime.NavMesh()
    releaseNavMesh = () => navMesh.destroy?.()
    const configuration = recastConfig(Runtime, geometry.bounds)

    try {
      navMesh.build(
        geometry.positions,
        geometry.positions.length / 3,
        geometry.indices,
        geometry.indices.length,
        configuration
      )
    } finally {
      disposeRecastWrapper(configuration)
    }

    const query = createRecastNavigationQuery(navMesh, (x, y, z) => new Runtime.Vec3(x, y, z))

    // The query now owns navMesh and its idempotent destroy lifecycle.
    releaseNavMesh = undefined

    return createInteriorNavigation({ ...query, ...safety },geometry,navigation.colliders??[],interiorPlans,envelope)
  } catch (error) {
    releaseNavMesh?.()

    if (error instanceof RuntimeAssetIntegrityError) {throw error}

    return closed
  } finally {
    if (imported) {
      disposeNavigationImport(imported)
    }
  }
}
