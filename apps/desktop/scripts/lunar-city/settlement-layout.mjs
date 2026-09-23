// Metres. Preserve the authored district relationships, with room for card-derived footprints.
export const HORIZONTAL_SCALE = 1
export const BUILDING_SCALE = 0.75
const district = (id, x, y, z, diameter) => ({
  id,
  position: [x * 0.87 * BUILDING_SCALE, 0.55, z * 0.87 * BUILDING_SCALE],
  diameter: diameter * BUILDING_SCALE
})
export const DISTRICT_LAYOUT = [
  district('library', -44.8, 0.8, -28.8, 25),
  district('research-lab', 40, 1.1, -35.2, 23),
  district('depot', -63, 0.45, 20, 24),
  district('review-office', 90, 0.7, 42, 24),
  district('triage', 16, 0.4, 42, 21),
  district('garden', -12.8, 0.25, 54.4, 24),
  district('council', 50, 0.35, 58, 44),
  district('bus', 0, 0.55, -14, 23),
  district('arts-studio', -46.4, 0.7, -1.6, 20),
  district('engineering-workshop', -46.4, 0.6, 48, 26),
  district('release-gatehouse', 0, 0.5, 19.2, 23),
  district('archive', 65, 0.5, -10, 35),
  district('revenue', -6, 0.7, -37, 31),
  district('publishing', 22, 0.55, -64, 19),
  district('project-inner', -8, 0.4, 88, 21),
  district('project-outer', 100, 0.5, 82, 21)
]
export const ROUTES = [
  [0, 1],
  [0, 7],
  [1, 7],
  [2, 5],
  [3, 4],
  [4, 5],
  [4, 6],
  [7, 4],
  [8, 0],
  [8, 2],
  [9, 5],
  [9, 10],
  [10, 4],
  [11, 6],
  [11, 3],
  [12, 0],
  [12, 1],
  [12, 7],
  [13, 12],
  [14, 5],
  [15, 6]
]
export const DECK_OFFSET = 1.65
export const localDistrictPoints = DISTRICT_LAYOUT.map(({ position: [x, y, z] }) => [
  x / HORIZONTAL_SCALE,
  y,
  z / HORIZONTAL_SCALE
])
export function arrivalPoint(district) {
  const [x, y, z] = district.position
  return [x, y + DECK_OFFSET, z]
}

const footprints = {
  library: [22, 22],
  'research-lab': [18, 18],
  depot: [24, 20],
  'review-office': [26, 20],
  triage: [14, 12],
  garden: [28, 24],
  council: [39, 39],
  bus: [10, 6],
  'arts-studio': [16, 16],
  'engineering-workshop': [22, 22],
  'release-gatehouse': [22, 18],
  archive: [31, 31],
  revenue: [27, 27],
  publishing: [14, 16],
  'project-inner': [16, 16],
  'project-outer': [16, 16]
}
for (const key of Object.keys(footprints)) footprints[key] = footprints[key].map(value => value * BUILDING_SCALE)
export const ROAD_WIDTH = 2.25
export const ROAD_LEVEL = 2.2
export function districtYaw(district) {
  return Math.atan2(-district.position[0], -district.position[2])
}
export function doorwayPoint(district) {
  const yaw = districtYaw(district)
  const distance = footprints[district.id][1] / 2 + ROAD_WIDTH / 2 + 0.35
  return [
    district.position[0] + Math.sin(yaw) * distance,
    district.position[1] + DECK_OFFSET,
    district.position[2] + Math.cos(yaw) * distance
  ]
}
// Oriented footprint tests preserve the actual entrance/front direction.
export const obstacles = DISTRICT_LAYOUT.filter(
  d => !['project-inner', 'project-outer'].includes(d.id)
).map(d => ({
  id: d.id,
  x: d.position[0],
  z: d.position[2],
  yaw: districtYaw(d),
  halfWidth: footprints[d.id][0] / 2 + ROAD_WIDTH / 2 + 0.15,
  halfDepth: footprints[d.id][1] / 2 + ROAD_WIDTH / 2 + 0.15
}))
function local(point, obstacle) {
  const x = point[0] - obstacle.x,
    z = point[2] - obstacle.z,
    c = Math.cos(obstacle.yaw),
    s = Math.sin(obstacle.yaw)
  return [x * c - z * s, x * s + z * c]
}
export function segmentBlocked(from, to, obstacle) {
  const a = local(from, obstacle),
    b = local(to, obstacle)
  let lower = 0,
    upper = 1
  for (const [axis, extent] of [
    [0, obstacle.halfWidth - 0.00001],
    [1, obstacle.halfDepth - 0.00001]
  ]) {
    const delta = b[axis] - a[axis]
    if (Math.abs(delta) < 1e-10) {
      if (Math.abs(a[axis]) >= extent) return false
      continue
    }
    const t1 = (-extent - a[axis]) / delta,
      t2 = (extent - a[axis]) / delta
    lower = Math.max(lower, Math.min(t1, t2))
    upper = Math.min(upper, Math.max(t1, t2))
    if (lower >= upper) return false
  }
  return upper > 1e-8 && lower < 1 - 1e-8
}
const visible = (a, b) => !obstacles.some(o => segmentBlocked(a, b, o))
const distance = (a, b) => Math.hypot(a[0] - b[0], a[2] - b[2])
function corners(obstacle) {
  const c = Math.cos(obstacle.yaw),
    s = Math.sin(obstacle.yaw)
  return [
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1]
  ]
    .map(([sx, sz]) => {
      const x = sx * (obstacle.halfWidth + 0.025),
        z = sz * (obstacle.halfDepth + 0.025)
      return [obstacle.x + x * c + z * s, ROAD_LEVEL, obstacle.z - x * s + z * c]
    })
    .filter(p => visible(p, p))
}
function routeBetween(from, to) {
  const points = [from, to, ...obstacles.flatMap(corners)]
  const distances = points.map(() => Infinity),
    previous = points.map(() => -1),
    visited = new Set()
  distances[0] = 0
  while (visited.size < points.length) {
    let current = -1
    for (let i = 0; i < points.length; i++)
      if (!visited.has(i) && (current < 0 || distances[i] < distances[current])) current = i
    if (current === 1 || current < 0 || !Number.isFinite(distances[current])) break
    visited.add(current)
    for (let next = 0; next < points.length; next++) {
      if (visited.has(next) || !visible(points[current], points[next])) continue
      const candidate = distances[current] + distance(points[current], points[next])
      if (candidate < distances[next]) {
        distances[next] = candidate
        previous[next] = current
      }
    }
  }
  if (!Number.isFinite(distances[1])) throw new Error(`No collision-free district approach from ${from} to ${to}`)
  const result = []
  for (let cursor = 1; cursor >= 0; cursor = previous[cursor]) result.unshift(points[cursor])
  // Constant-height trunk roads meet gentle entry ramps, so crossings share a surface.
  const ramp = (a, b, height) => {
    const length = distance(a, b),
      t = Math.min(0.4, Math.max(2, Math.abs(a[1] - height) * 12) / length)
    return [a[0] + (b[0] - a[0]) * t, height, a[2] + (b[2] - a[2]) * t]
  }
  return [
    result[0],
    ramp(result[0], result[1], ROAD_LEVEL),
    ...result.slice(1, -1),
    ramp(result.at(-1), result.at(-2), ROAD_LEVEL),
    result.at(-1)
  ]
}
export const PEDESTRIAN_ROUTES = ROUTES.map(([from, to]) => ({
  from: DISTRICT_LAYOUT[from].id,
  to: DISTRICT_LAYOUT[to].id,
  points: routeBetween(doorwayPoint(DISTRICT_LAYOUT[from]), doorwayPoint(DISTRICT_LAYOUT[to]))
}))
