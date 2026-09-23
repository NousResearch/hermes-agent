import type { NavigationCollider, Vec3 } from '../model'

interface GroundGeometry { positions: Float32Array | Float64Array; indices: Uint32Array }
export interface ActorEnvelope { radius: number; height: number }
const WORKER_ENVELOPE: ActorEnvelope = { radius: .35, height: 1.2 }
const HEIGHT_TOLERANCE = .25

function colliderBlocks(from: Vec3, to: Vec3, collider: NavigationCollider, envelope: ActorEnvelope): boolean {
  const bottom = collider.center.y - (collider.kind === 'box' ? collider.halfExtents.y : collider.height / 2)
  const top = collider.center.y + (collider.kind === 'box' ? collider.halfExtents.y : collider.height / 2)

  if (Math.max(from.y, to.y) + envelope.height <= bottom || Math.min(from.y, to.y) >= top) {return false}

  if (collider.kind === 'cylinder') {
    const dx = to.x - from.x, dz = to.z - from.z
    const length = dx * dx + dz * dz
    const t = length ? Math.max(0, Math.min(1, ((collider.center.x - from.x) * dx + (collider.center.z - from.z) * dz) / length)) : 0

    return Math.hypot(from.x + dx * t - collider.center.x, from.z + dz * t - collider.center.z) < collider.radius + envelope.radius
  }

  const cosine = Math.cos(collider.rotationY), sine = Math.sin(collider.rotationY)
  const local = (point: Vec3) => ({ x: (point.x - collider.center.x) * cosine - (point.z - collider.center.z) * sine, z: (point.x - collider.center.x) * sine + (point.z - collider.center.z) * cosine })
  const a = local(from), b = local(to)
  let low = 0, high = 1

  for (const axis of ['x', 'z'] as const) {
    const extent = collider.halfExtents[axis] + envelope.radius, delta = b[axis] - a[axis]

    if (Math.abs(delta) < 1e-10) { if (Math.abs(a[axis]) > extent) {return false;}

 continue }

    const first = (-extent - a[axis]) / delta, second = (extent - a[axis]) / delta
    low = Math.max(low, Math.min(first, second)); high = Math.min(high, Math.max(first, second))

    if (low > high) {return false}
  }

  return true
}

/** Ground comes from the verified navigation triangles; scenery never invents a floor. */
export function createMovementSafety(geometry: GroundGeometry, colliders: readonly NavigationCollider[] = [], envelope: ActorEnvelope = WORKER_ENVELOPE) {
  if (![envelope.radius, envelope.height].every(value => Number.isFinite(value) && value > 0)) {throw new Error('Invalid actor envelope')}
  const { positions, indices } = geometry

  const triangles = Array.from({ length: indices.length / 3 }, (_, index) => {
    const points = [0, 1, 2].map(offset => {
      const vertex = indices[index * 3 + offset]! * 3

      return { x: positions[vertex]!, y: positions[vertex + 1]!, z: positions[vertex + 2]! }
    })

    return points as [Vec3, Vec3, Vec3]
  })

  const cells = new Map<string, number[]>()
  const cellSize = 4
  triangles.forEach((points, index) => {
    for (let x = Math.floor(Math.min(...points.map(p => p.x)) / cellSize); x <= Math.floor(Math.max(...points.map(p => p.x)) / cellSize); x++) {
      for (let z = Math.floor(Math.min(...points.map(p => p.z)) / cellSize); z <= Math.floor(Math.max(...points.map(p => p.z)) / cellSize); z++) {
        const key = `${x},${z}`, values = cells.get(key) ?? []
        values.push(index); cells.set(key, values)
      }
    }
  })

  const ground = (point: Vec3): Vec3 | undefined => {
    let best: number | undefined

    for (const index of cells.get(`${Math.floor(point.x / cellSize)},${Math.floor(point.z / cellSize)}`) ?? []) {
      const [a, b, c] = triangles[index]!
      const determinant = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z)

      if (Math.abs(determinant) < 1e-10) {continue}
      const u = ((b.z - c.z) * (point.x - c.x) + (c.x - b.x) * (point.z - c.z)) / determinant
      const v = ((c.z - a.z) * (point.x - c.x) + (a.x - c.x) * (point.z - c.z)) / determinant

      if (u < -1e-5 || v < -1e-5 || u + v > 1.00001) {continue}
      const y = u * a.y + v * b.y + (1 - u - v) * c.y

      if (Math.abs(y - point.y) <= HEIGHT_TOLERANCE && (best === undefined || Math.abs(y - point.y) < Math.abs(best - point.y))) {best = y}
    }

    return best === undefined ? undefined : { x: point.x, y: best, z: point.z }
  }

  const resolvePosition = (point: Vec3): Vec3 | undefined => {
    if (![point.x, point.y, point.z].every(Number.isFinite)) {return undefined}
    const grounded = ground(point)

    if (!grounded || colliders.some(collider => colliderBlocks(grounded, grounded, collider, envelope))) {return undefined}

    return grounded
  }

  const canTraverse = (from: Vec3, to: Vec3): boolean => {
    if (![from.x,from.y,from.z,to.x,to.y,to.z].every(Number.isFinite) || !ground(from) || !ground(to)) {return false}

    if (colliders.some(collider => colliderBlocks(from, to, collider, envelope))) {return false}

    if (Math.hypot(from.x - to.x, from.y - to.y, from.z - to.z) < 1e-10) {return true}
    // Clip the entire segment against triangle interiors, then require continuous
    // coverage. Point sampling can miss narrow floor gaps and is not sufficient.
    const candidates = new Set<number>()

    for (let x = Math.floor(Math.min(from.x, to.x) / cellSize); x <= Math.floor(Math.max(from.x, to.x) / cellSize); x++) {
      for (let z = Math.floor(Math.min(from.z, to.z) / cellSize); z <= Math.floor(Math.max(from.z, to.z) / cellSize); z++) {
        for (const index of cells.get(`${x},${z}`) ?? []) {candidates.add(index)}
      }
    }

    const coverage: Array<[number, number]> = []

    for (const index of candidates) {
      const [a, b, c] = triangles[index]!
      const determinant = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z)

      if (Math.abs(determinant) < 1e-10) {continue}

      const weights = (point: Vec3) => {
        const u = ((b.z-c.z)*(point.x-c.x)+(c.x-b.x)*(point.z-c.z))/determinant
        const v = ((c.z-a.z)*(point.x-c.x)+(a.x-c.x)*(point.z-c.z))/determinant
        const w = 1-u-v, offset = u*a.y+v*b.y+w*c.y-point.y

        return [u,v,w,HEIGHT_TOLERANCE-offset,HEIGHT_TOLERANCE+offset]
      }

      const first = weights(from), last = weights(to)
      let low = 0, high = 1

      for (let constraint = 0; constraint < first.length; constraint++) {
        const value = first[constraint]!, delta = last[constraint]! - value

        if (Math.abs(delta) < 1e-12) { if (value < -1e-8) { high = -1;

 break } }
        else if (delta > 0) {low = Math.max(low, (-1e-8-value)/delta)}
        else {high = Math.min(high, (-1e-8-value)/delta)}
      }

      if (low <= high) {coverage.push([low,high])}
    }

    coverage.sort((left,right)=>left[0]-right[0])
    let covered = 0

    for (const [start,end] of coverage) {
      if (start > covered + 1e-7) {return false}
      covered = Math.max(covered,end)

      if (covered >= 1-1e-7) {return true}
    }

    return false
  }

  return { resolvePosition, canTraverse }
}
