import { createHash } from 'node:crypto'
import { writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { NullEngine, Scene, GLTF2Export } from './modeling/babylon.mjs'
import { buildTerrain, buildNavigation } from './modeling/terrain.mjs'
import { mergeLodMeshes } from './modeling/primitives.mjs'
import {
  BUILDING_SCALE,
  DISTRICT_LAYOUT,
  HORIZONTAL_SCALE,
  PEDESTRIAN_ROUTES,
  arrivalPoint,
  doorwayPoint,
  districtYaw
} from './settlement-layout.mjs'

async function exportGeometry(output, id, builder) {
  const engine = new NullEngine()
  const scene = new Scene(engine)
  scene.useRightHandedSystem = true
  try {
    const root = builder(scene)
    if (id === 'terrain') {
      for (const lod of ['near', 'far'])
        mergeLodMeshes(scene, scene.getTransformNodeByName(`terrain:lod:${lod}`), `terrain:${lod}`)
    } else mergeLodMeshes(scene, root, id)
    root.scaling.set(HORIZONTAL_SCALE, 1, HORIZONTAL_SCALE)
    for (const mesh of scene.meshes) mesh.computeWorldMatrix(true)
    const bounds = root.getHierarchyBoundingVectors(true)
    const statistics = {
      animationClips: scene.animationGroups.map(group => group.name),
      extent: bounds.max.subtract(bounds.min).asArray(),
      drawCalls: scene.meshes.length,
      materials: scene.materials.length,
      meshes: scene.meshes.length,
      nodes: scene.transformNodes.length + scene.meshes.length,
      textures: scene.textures.length,
      triangles: scene.meshes.reduce((sum, mesh) => sum + mesh.getTotalIndices() / 3, 0)
    }
    const exported = await GLTF2Export.GLBAsync(scene, id, {
      exportWithoutWaitingForScene: true,
      removeNoopRootNodes: false
    })
    const bytes = Buffer.from(await exported.glTFFiles[`${id}.glb`].arrayBuffer())
    const sha256 = createHash('sha256').update(bytes).digest('hex')
    const uri = `models/${id}-${sha256.slice(0, 12)}.glb`
    await writeFile(resolve(output, uri), bytes)
    return {
      uri,
      colliders: scene.metadata?.staticColliders ?? [],
      statistics: {
        ...statistics,
        sha256,
        bytes: bytes.length,
        gpuMiB: bytes.length / 1024 ** 2,
        gpuBytes: bytes.length,
        accessorBytes: bytes.length
      },
      bounds: { min: bounds.min.asArray(), max: bounds.max.asArray() }
    }
  } finally {
    scene.dispose()
    engine.dispose()
  }
}

export async function buildReviewEnvironment(manifest, output) {
  const terrain = await exportGeometry(output, 'terrain', buildTerrain)
  const navigation = await exportGeometry(output, 'navigation', buildNavigation)
  const terrainModel = manifest.models.find(model => model.id === 'terrain')
  terrain.statistics.budget = { ...terrainModel.statistics.budget, maxMaterials: 10, maxDrawCalls: 16 }
  terrainModel.maxDrawCalls = 16
  Object.assign(terrainModel, terrain)
  terrainModel.maxMaterials = Math.max(terrainModel.maxMaterials, terrain.statistics.materials)
  terrainModel.maxDrawCalls = Math.max(terrainModel.maxDrawCalls, terrain.statistics.drawCalls)
  manifest.navigation.colliders = terrain.colliders
  manifest.navigation.meshUri = navigation.uri
  manifest.navigation.sha256 = navigation.statistics.sha256
  manifest.navigation.links = PEDESTRIAN_ROUTES.flatMap(route =>
    route.points.slice(1).map((point, index) => ({ from: route.points[index], to: point, bidirectional: true }))
  )
  const aliases = { 'research-lab': 'lab', 'review-office': 'review' }
  for (const district of DISTRICT_LAYOUT) {
    const model = manifest.models.find(model => model.id === district.id)
    const point = arrivalPoint(district)
    if (model) {
      model.transform.position = point
      model.transform.scale = model.transform.scale.map(value => value * BUILDING_SCALE)
      model.transform.rotation = [0, districtYaw(district), 0]
    }
    manifest.destinations[aliases[district.id] ?? district.id] = doorwayPoint(district)
  }
  manifest.destinations.project = doorwayPoint(DISTRICT_LAYOUT.find(district => district.id === 'project-inner'))
  manifest.destinations.unavailable = manifest.destinations.bus
  // This reviewed terrain has two authored project pads; excess projects stay explicit overflow.
  manifest.projectSlots = manifest.projectSlots.slice(0, 2)
  for (const [index, slot] of manifest.projectSlots.entries()) {
    const point = arrivalPoint(
      DISTRICT_LAYOUT.find(district => district.id === (index ? 'project-outer' : 'project-inner'))
    )
    slot.position = point
    slot.bounds = { min: [point[0] - 8, point[1], point[2] - 8], max: [point[0] + 8, point[1] + 12, point[2] + 8] }
    slot.navigationLink = {
      from: doorwayPoint(DISTRICT_LAYOUT.find(district => district.id === (index ? 'project-outer' : 'project-inner'))),
      to: manifest.destinations[index ? 'council' : 'garden'],
      bidirectional: true
    }
  }
  Object.assign(manifest.camera.overview, { radius: 150, maxRadius: 200, target: [4, 3, 5] })
  manifest.camera.bounds = { min: [-120, -12, -100], max: [120, 45, 105] }
  for (const model of manifest.models.filter(model => !['workers', 'leaders'].includes(model.id))) {
    for (const lod of model.lods) if (lod.distance > 0) lod.distance = 360
  }
  const receipt = {
    units: 'metres',
    coordinates: 'right-handed Y up',
    horizontalExpansion: HORIZONTAL_SCALE,
    terrain,
    navigation,
    districts: DISTRICT_LAYOUT,
    buildingStatus: 'Eight card-derived building review candidates with distinct operational fixtures',
    landscape: 'Continuous district-colored ground, riverbanks, groves, gravel shoulders and paved streets'
  }
  await writeFile(resolve(output, 'environment-build.json'), JSON.stringify(receipt, null, 2) + '\n')
  return receipt
}
