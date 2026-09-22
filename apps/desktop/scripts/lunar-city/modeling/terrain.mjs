import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { RIVER_POINTS, SEA_LEVEL, coastDistance, groundHeight, riverSample, smooth } from './island-shape.mjs'
import { beamBetween, box, cylinder, group, sphere, torus } from './primitives.mjs'
import { Color3, Mesh, PBRMaterial, VertexBuffer } from './babylon.mjs'
import {
  DISTRICT_LAYOUT,
  HORIZONTAL_SCALE,
  PEDESTRIAN_ROUTES,
  ROAD_WIDTH,
  obstacles,
  segmentBlocked,
  districtYaw
} from '../settlement-layout.mjs'

function distanceToSegment(x, z, a, b) {
  const dx = b[0] - a[0],
    dz = b[2] - a[2],
    t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[2]) * dz) / (dx * dx + dz * dz || 1)))
  return Math.hypot(x - a[0] - t * dx, z - a[2] - t * dz)
}
const segments = PEDESTRIAN_ROUTES.flatMap(r => r.points.slice(1).map((p, i) => [r.points[i], p]))
function reserved(x, z, extra = 0) {
  return (
    obstacles.some(o =>
      segmentBlocked([x, 0, z], [x, 0, z], { ...o, halfWidth: o.halfWidth + extra, halfDepth: o.halfDepth + extra })
    ) ||
    DISTRICT_LAYOUT.some(d => Math.hypot(x - d.position[0], z - d.position[2]) < d.diameter / 2 + extra) ||
    segments.some(([a, b]) => distanceToSegment(x, z, a, b) < ROAD_WIDTH / 2 + 1.4 + extra)
  )
}
function materials(scene) {
  const make = (name, hex, rough = 0.9) => {
    const m = new PBRMaterial('terrain:' + name, scene)
    m.albedoColor = Color3.FromHexString(hex)
    m.roughness = rough
    m.metallic = 0
    return m
  }
  return {
    ground: make('meadow', '#ffffff'),
    grass: make('grass', '#718854'),
    leaf: make('leaves', '#426548'),
    leafLight: make('leaves-light', '#759257'),
    dirt: make('earth', '#7a6048'),
    gravel: make('gravel', '#a29a83'),
    road: make('road', '#626770'),
    paving: make('paving', '#beb8a3'),
    water: make('river', '#397e8b', 0.25),
    bark: make('bark', '#625346')
  }
}
const districtSoils = {
  library: [0.25, 0.38, 0.19],
  garden: [0.29, 0.44, 0.23],
  archive: [0.28, 0.36, 0.23],
  'research-lab': [0.52, 0.52, 0.45],
  'engineering-workshop': [0.47, 0.37, 0.27],
  depot: [0.45, 0.34, 0.24],
  revenue: [0.48, 0.41, 0.3],
  council: [0.35, 0.43, 0.27],
  'arts-studio': [0.46, 0.34, 0.28],
  publishing: [0.38, 0.4, 0.25]
}
function districtGround(x, z) {
  const nearest = DISTRICT_LAYOUT.map(d => ({ d, distance: Math.hypot(x - d.position[0], z - d.position[2]) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 3)
  const weights = nearest.map(n => 1 / Math.max(5, n.distance) ** 2),
    total = weights.reduce((a, b) => a + b, 0)
  return [0, 1, 2].map(i =>
    nearest.reduce((sum, n, j) => sum + ((districtSoils[n.d.id] ?? [0.35, 0.39, 0.26])[i] * weights[j]) / total, 0)
  )
}
function ground(scene, parent, mat, step) {
  const positions = [],
    colors = [],
    indices = [],
    normals = []
  const minX = -95,
    maxX = 98,
    minZ = -87,
    maxZ = 94
  const nx = Math.ceil((maxX - minX) / step),
    nz = Math.ceil((maxZ - minZ) / step)
  for (let iz = 0; iz <= nz; iz++)
    for (let ix = 0; ix <= nx; ix++) {
      const x = minX + (ix * (maxX - minX)) / nx,
        z = minZ + (iz * (maxZ - minZ)) / nz,
        y = groundHeight(x, z)
      positions.push(x, y, z)
      normals.push(0, 1, 0)
      const noise = 0.5 + 0.5 * Math.sin(x * 0.19 + Math.sin(z * 0.14)) * Math.cos(z * 0.21)
      const bank = 1 - smooth(3, 8, riverSample(x, z).distance)
      const dust = reserved(x, z, 1) ? 0.7 : 0.15 + noise * 0.18
      const green = districtGround(x, z).map(v => v * (0.85 + 0.3 * noise)),
        earth = [0.47, 0.37, 0.27]
      const t = Math.max(bank, dust)
      const beach = 1 - smooth(2, 9, coastDistance(x, z)),
        sand = [0.68, 0.58, 0.39]
      colors.push(...green.map((v, i) => (v * (1 - t) + earth[i] * t) * (1 - beach) + sand[i] * beach), 1)
    }
  for (let z = 0; z < nz; z++)
    for (let x = 0; x < nx; x++) {
      const a = z * (nx + 1) + x,
        b = a + 1,
        c = a + nx + 1,
        d = c + 1
      indices.push(a, b, c, b, d, c)
    }
  const mesh = new Mesh(parent.name + ':continuous-ground', scene)
  mesh.parent = parent
  mesh.material = mat
  mesh.setVerticesData(VertexBuffer.PositionKind, positions)
  mesh.setVerticesData(VertexBuffer.UVKind, positions.flatMap((value,i)=>i%3===1?[]:[value*.1]))
  mesh.setVerticesData(VertexBuffer.NormalKind, normals)
  mesh.setVerticesData(VertexBuffer.ColorKind, colors)
  mesh.setIndices(indices)
  mesh.isPickable = false
}
function river(scene, parent, m) {
  const positions = [],
    normals = [],
    indices = []
  for (let i = 0; i < RIVER_POINTS.length; i++) {
    const p = RIVER_POINTS[i],
      a = RIVER_POINTS[Math.max(0, i - 1)],
      b = RIVER_POINTS[Math.min(RIVER_POINTS.length - 1, i + 1)]
    const dx = b[0] - a[0],
      dz = b[2] - a[2],
      length = Math.hypot(dx, dz),
      width = 2.65 + 0.18 * Math.sin(i * 0.3)
    for (const side of [-1, 1]) {
      positions.push(p[0] + ((side * dz) / length) * width, p[1] + 0.015 * smooth(0, 5, coastDistance(p[0], p[2])) - 0.03 * (1 - smooth(-3, 0, coastDistance(p[0], p[2]))), p[2] - ((side * dx) / length) * width)
      normals.push(0, 1, 0)
    }
    if (i) {
      const a = (i - 1) * 2,
        b = a + 1,
        c = i * 2,
        d = c + 1
      indices.push(a, b, c, b, d, c)
    }
  }
  const mesh = new Mesh(parent.name + ':river-channel', scene)
  mesh.parent = parent
  mesh.material = m.water
  m.water.backFaceCulling = false
  mesh.setVerticesData(VertexBuffer.PositionKind, positions)
  mesh.setVerticesData(VertexBuffer.UVKind, positions.flatMap((value,i)=>i%3===1?[]:[value*.1]))
  mesh.setVerticesData(VertexBuffer.NormalKind, normals)
  mesh.setIndices(indices)
  mesh.isPickable = false
  box(scene, parent.name + ':sea', {
    width: 900,
    depth: 900,
    height: 0.05,
    position: [0, SEA_LEVEL - 0.025, 0],
    material: m.water,
    parent
  })
  cylinder(scene, parent.name + ':headwater-pool', {
    diameter: 10,
    height: 0.06,
    position: [RIVER_POINTS[0][0], RIVER_POINTS[0][1] - 0.02, RIVER_POINTS[0][2]],
    material: m.water,
    parent,
    tessellation: 24
  })
}
const roadSurfaces = JSON.parse(readFileSync(new URL('../road-surfaces.json', import.meta.url), 'utf8'))
const roadHash = createHash('sha256')
  .update(JSON.stringify({ routes: PEDESTRIAN_ROUTES, width: ROAD_WIDTH }))
  .digest('hex')
if (roadHash !== roadSurfaces.sourceSha256)
  throw new Error('Road geometry is stale: run build-road-surfaces.py with shapely==2.1.2')
function streets(scene, parent, m) {
  for (const [name, data] of Object.entries(roadSurfaces.layers)) {
    const mesh = new Mesh(parent.name + ':joined-' + name, scene)
    mesh.parent = parent
    mesh.material = m[name]
    mesh.setVerticesData(VertexBuffer.PositionKind, data.positions)
    mesh.setVerticesData(VertexBuffer.UVKind, data.positions.flatMap((value,i)=>i%3===1?[]:[value*.1]))
    mesh.setVerticesData(
      VertexBuffer.NormalKind,
      data.positions.map((_, i) => (i % 3 === 1 ? 1 : 0))
    )
    mesh.setIndices(data.indices)
    mesh.isPickable = false
  }
}
function tree(scene, parent, m, x, z, size, index, detail) {
  const y = groundHeight(x, z)
  if (detail) scene.metadata.staticColliders.push({
    id: `tree:${index}`, kind: 'cylinder', center: [x, y + 1.2 * size, z],
    radius: 0.125 * size, height: 2.4 * size
  })
  cylinder(scene, parent.name + ':tree-trunk:' + index, {
    diameter: 0.25 * size,
    height: 2.4 * size,
    position: [x, y + 1.2 * size, z],
    material: m.bark,
    parent,
    tessellation: 6
  })
  for (let k = 0; k < (detail ? 2 : 1); k++)
    sphere(scene, parent.name + ':tree-crown:' + index + ':' + k, {
      diameter: (2.3 - k * 0.35) * size,
      segments: detail ? 3 : 2,
      scale: [1, 1.05, 0.9],
      position: [x + Math.sin(k * 2.4) * size * 0.6, y + (2.9 + k * 0.45) * size, z + Math.cos(k * 2.4) * size * 0.5],
      material: k % 2 ? m.leafLight : m.leaf,
      parent
    })
}
function landscape(scene, parent, m, detail) {
  let index = 0
  for (let i = 4; i < RIVER_POINTS.length - 4; i += 5)
    for (const side of [-1, 1]) {
      const p = RIVER_POINTS[i],
        a = RIVER_POINTS[i - 1],
        b = RIVER_POINTS[i + 1],
        dx = b[0] - a[0],
        dz = b[2] - a[2],
        length = Math.hypot(dx, dz)
      const x = p[0] + ((side * dz) / length) * 6.5,
        z = p[2] - ((side * dx) / length) * 6.5
      if (!reserved(x, z, 2) && coastDistance(x, z) > 5)
        tree(scene, parent, m, x, z, 0.8 + 0.25 * Math.sin(i) ** 2, index++, detail)
    }
  // Groves fill real gaps between blocks while keeping doorways and walking corridors clear.
  for (let z = -45; z < 58; z += 8.5)
    for (let x = -49; x < 68; x += 8.5) {
      const px = x + 2 * Math.sin(z * 0.7),
        pz = z + 2 * Math.cos(x * 0.6)
      if (reserved(px, pz, 3) || riverSample(px, pz).distance < 8 || coastDistance(px, pz) < 6) continue
      if (Math.sin(x * 1.31 + z * 0.71) > 0.1)
        tree(scene, parent, m, px, pz, 0.7 + 0.5 * (0.5 + 0.5 * Math.sin(x + z)), index++, detail)
      else if (detail)
        sphere(scene, parent.name + ':shrub:' + index++, {
          diameter: 1.8,
          segments: 3,
          scale: [1, 0.45, 1],
          position: [px, groundHeight(px, pz) + 0.35, pz],
          material: m.leafLight,
          parent
        })
    }
  if (detail)
    for (let angle = 0; angle < Math.PI * 2; angle += 0.055) {
      for (let radius = 45; radius < 89; radius += 2) {
        const x = 3 + Math.cos(angle) * radius,
          z = 3 + Math.sin(angle) * radius
        if (Math.abs(coastDistance(x, z) - 3) > 1.3 || reserved(x, z, 2)) continue
        sphere(scene, parent.name + ':shore-rock:' + index++, {
          diameter: 1 + 0.8 * Math.sin(angle * 7) ** 2,
          segments: 3,
          scale: [1, 0.65, 0.8],
          position: [x, groundHeight(x, z) + 0.2, z],
          material: m.gravel,
          parent
        })
      }
    }
}
export function buildTerrain(scene) {
  scene.metadata = { ...scene.metadata, staticColliders: [] }
  const root = group(scene, 'terrain:root'),
    m = materials(scene)
  for (const [lod, detail] of [
    ['near', true],
    ['far', false]
  ]) {
    const parent = group(scene, 'terrain:lod:' + lod, root)
    ground(scene, parent, m.ground, detail ? 2.5 : 6)
    river(scene, parent, m, !detail)
    streets(scene, group(scene, detail ? 'terrain:walkways' : 'terrain:far:walkways', parent), m, detail)
    for (const [index, d] of DISTRICT_LAYOUT.entries()) {
      const [x, , z] = d.position
      if (d.id !== 'engineering-workshop') {
        box(scene, `terrain:district-deck:${lod}:${index}`, {
          width: d.diameter,
          depth: d.diameter,
          height: 0.22,
          position: [x, 2.07, z],
          rotation: [0, districtYaw(d), 0],
          material: m.paving,
          parent
        })
      }
      // The decorative inset stays below the walking surface.
      if (d.id !== 'engineering-workshop')
        torus(scene, `terrain:district-ring:${lod === 'near' ? '' : lod + ':'}${index}`, {
          diameter: d.diameter - 0.5,
          thickness: 0.1,
          position: [x, 2.14, z],
          material: m.gravel,
          parent,
          tessellation: 12
        })
    }
    landscape(scene, parent, m, detail)
  }
  return root
}

export function buildNavigation(scene) {
  const root = group(scene, 'navigation:root')
  const walkable = group(scene, 'navigation:walkable', root)
  for (const [routeIndex, route] of PEDESTRIAN_ROUTES.entries()) {
    const points = route.points.map(([x, y, z]) => [x / HORIZONTAL_SCALE, y, z / HORIZONTAL_SCALE])
    for (const [index, point] of points.entries()) {
      cylinder(scene, `navigation:join:${routeIndex}:${index}`, {
        diameter: ROAD_WIDTH / HORIZONTAL_SCALE - 0.15,
        height: 0.1,
        material: 'charcoal-structure',
        parent: walkable,
        position: [point[0], point[1] - 0.05, point[2]],
        tessellation: 12
      })
      if (index === 0) continue
      const a = points[index - 1],
        b = point
      beamBetween(
        scene,
        `navigation:path:${routeIndex}:${index}`,
        [a[0], a[1] - 0.05, a[2]],
        [b[0], b[1] - 0.05, b[2]],
        { height: 0.1, width: ROAD_WIDTH / HORIZONTAL_SCALE - 0.2, material: 'charcoal-structure', parent: walkable }
      )
    }
  }
  return root
}
