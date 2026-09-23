import { DISTRICT_LAYOUT, districtYaw } from '../settlement-layout.mjs'
export const SEA_LEVEL = 1.48
export const smooth = (a, b, x) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}
const beaver = DISTRICT_LAYOUT.find(d => d.id === 'engineering-workshop')
const [bx, , bz] = beaver.position,
  yaw = districtYaw(beaver)
export const WATERWORKS_PORTS = {
  rear: [bx - Math.sin(yaw) * 8.184, 2.24, bz - Math.cos(yaw) * 8.184],
  front: [bx + Math.sin(yaw) * 8.184, SEA_LEVEL, bz + Math.cos(yaw) * 8.184]
}
const controls = [
  [-49, 2.24, 51],
  WATERWORKS_PORTS.rear,
  [bx, 2.24, bz],
  WATERWORKS_PORTS.front,
  [-21, SEA_LEVEL, 15],
  [-17, SEA_LEVEL, 3],
  [-5, SEA_LEVEL, 1],
  [10, SEA_LEVEL, 0],
  [20, SEA_LEVEL, 11],
  [32, SEA_LEVEL, 12],
  [47, SEA_LEVEL, 12],
  [61, SEA_LEVEL, 8],
  [87, SEA_LEVEL, -12]
]
// Catmull-Rom gives continuous banks; preserve exact waterworks entry/exit controls.
export const RIVER_POINTS = []
for (let i = 0; i < controls.length - 1; i++) {
  const p0 = controls[Math.max(0, i - 1)],
    p1 = controls[i],
    p2 = controls[i + 1],
    p3 = controls[Math.min(controls.length - 1, i + 2)]
  const count = Math.ceil(Math.hypot(p2[0] - p1[0], p2[2] - p1[2]) / 1.4)
  for (let n = 0; n < count; n++) {
    const t = n / count,
      t2 = t * t,
      t3 = t2 * t
    const xz = [0, 2].map(
      k =>
        0.5 *
        (2 * p1[k] +
          (-p0[k] + p2[k]) * t +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 +
          (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)
    )
    RIVER_POINTS.push([xz[0], p1[1] + (p2[1] - p1[1]) * smooth(0, 1, t), xz[1]])
  }
}
RIVER_POINTS.push(controls.at(-1))
export function riverSample(x, z) {
  let distance = Infinity,
    height = SEA_LEVEL
  for (let i = 1; i < RIVER_POINTS.length; i++) {
    const a = RIVER_POINTS[i - 1],
      b = RIVER_POINTS[i],
      dx = b[0] - a[0],
      dz = b[2] - a[2]
    const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[2]) * dz) / (dx * dx + dz * dz || 1)))
    const d = Math.hypot(x - a[0] - t * dx, z - a[2] - t * dz)
    if (d < distance) {
      distance = d
      height = a[1] + t * (b[1] - a[1])
    }
  }
  return { distance, height }
}
export function coastDistance(x, z) {
  const px = x - 3,
    pz = z - 3,
    angle = Math.atan2(pz, px)
  const radius = 61 + 7 * Math.sin(angle * 3 + 0.6) + 4 * Math.cos(angle * 5 - 0.8) + 2 * Math.sin(angle * 9)
  let distance = radius - Math.hypot(px, pz)
  for (const d of DISTRICT_LAYOUT)
    distance = Math.max(distance, d.diameter * 0.72 + 10 - Math.hypot(x - d.position[0], z - d.position[2]))
  // The headwater lake occupies a wooded headland; it is not a second sea inlet.
  distance = Math.max(distance, 14 - Math.hypot(x + 49, z - 51))
  return distance
}
export function groundHeight(x, z) {
  const sample = riverSample(x, z),
    bank = smooth(2.65, 6.7, sample.distance)
  const inland = 1.96 + 0.06 * Math.sin(x * 0.22) * Math.cos(z * 0.19)
  const hill =
    Math.max(inland, sample.height + 0.4) * (1 - smooth(6, 14, sample.distance)) +
    inland * smooth(6, 14, sample.distance)
  const channel = sample.height - 0.26 + bank * (hill - sample.height + 0.26)
  const coast = 1.06 + smooth(-1, 8, coastDistance(x, z)) * (channel - 1.06)
  return coast
}
