import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'
import { BUILDING_SCALE, DISTRICT_LAYOUT, arrivalPoint, districtYaw } from './settlement-layout.mjs'

// All source-card buildings are selected only in the opt-in review pack.
// This replaces role entries; preserved historical assets remain in v2.
export const REVIEW_BUILDINGS = [
  ['owl', 'library', '', 16],
  ['fox', 'research-lab', '', 12],
  ['monkey', 'publishing', '', 12],
  ['cat', 'arts-studio', 'reconstruction-v2/', 10],
  ['elephant', 'archive', '', 24],
  ['capybara', 'revenue', '', 16],
  ['lion', 'council', '', 26],
  ['beaver', 'engineering-workshop', '', 14]
]

export async function buildReviewBuildings(manifest, assets, output, receipts) {
  const ids = manifest.models.map(model => model.id)
  if (new Set(ids).size !== ids.length) throw Error('Review pack contains duplicate model roles')
  for (const [species, id, variant, heightMetres] of REVIEW_BUILDINGS) {
    const source = `building-multiview-2026-09-07/${species}/${variant}building-review.glb`
    const bytes = await readFile(resolve(assets, source))
    const length = bytes.readUInt32LE(12)
    const gltf = JSON.parse(bytes.subarray(20, 20 + length).toString())
    if (gltf.nodes.some(node => node.matrix || node.translation || node.rotation || node.scale))
      throw Error(`Building bounds require transformed-node evaluation: ${species}`)
    const originalRoots = gltf.scenes[gltf.scene ?? 0].nodes
    const lod = gltf.nodes.length
    gltf.nodes.push({ name: `${id}:lod:near`, children: originalRoots })
    gltf.nodes.push({ name: `${id}:root`, children: [lod] })
    gltf.scenes[gltf.scene ?? 0].nodes = [lod + 1]
    let json = Buffer.from(JSON.stringify(gltf))
    json = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 32)])
    const chunks = bytes.subarray(20 + length)
    const header = Buffer.alloc(20)
    header.writeUInt32LE(0x46546c67, 0)
    header.writeUInt32LE(2, 4)
    header.writeUInt32LE(20 + json.length + chunks.length, 8)
    header.writeUInt32LE(json.length, 12)
    header.writeUInt32LE(0x4e4f534a, 16)
    const runtimeBytes = Buffer.concat([header, json, chunks])
    const sha256 = createHash('sha256').update(runtimeBytes).digest('hex')
    const uri = `models/review-building-${species}-${sha256.slice(0, 12)}.glb`
    await writeFile(resolve(output, uri), runtimeBytes)
    const primitives = gltf.meshes.flatMap(mesh => mesh.primitives)
    const positions = primitives.map(p => gltf.accessors[p.attributes.POSITION])
    const bounds = {
      min: [0, 1, 2].map(i => Math.min(...positions.map(p => p.min[i]))),
      max: [0, 1, 2].map(i => Math.max(...positions.map(p => p.max[i])))
    }
    if (Math.abs(bounds.max[1] - bounds.min[1] - heightMetres) > .005)
      throw Error(`Building source height differs from metre contract: ${species}`)
    const district = DISTRICT_LAYOUT.find(d => d.id === id)
    if (!district) throw Error(`Building role has no district: ${id}`)
    const model =
      manifest.models.find(m => m.id === id) ?? structuredClone(manifest.models.find(m => m.id === 'library'))
    if (!manifest.models.includes(model)) manifest.models.push(model)
    const triangles = primitives.reduce((n, p) => n + gltf.accessors[p.indices].count / 3, 0)
    const budget = {
      maxTriangles: Math.ceil(triangles),
      maxDrawCalls: primitives.length,
      maxMaterials: gltf.materials.length,
      maxTextures: gltf.textures?.length ?? 0,
      maxGpuMiB: Math.ceil(runtimeBytes.length / 1024 ** 2) * 2
    }
    const extent = bounds.max.map((v, i) => v - bounds.min[i])
    Object.assign(model, {
      id,
      uri,
      ...budget,
      requiredNodes: [`${id}:root`, `${id}:lod:near`],
      requiredClips: [],
      lods: [{ distance: 0, node: `${id}:lod:near` }],
      bounds,
      pivot: [0, 0, 0],
      transform: {
        position: arrivalPoint(district).map((v, i) => (i === 1 ? v - bounds.min[1] * BUILDING_SCALE : v)),
        rotation: [0, districtYaw(district), 0],
        scale: [BUILDING_SCALE, BUILDING_SCALE, BUILDING_SCALE]
      },
      cameraAnchor: [0, extent[1] * 0.55, 0],
      anchors: { foot: [0, 0, 0], interaction: [0, 0, 0], camera: [0, extent[1] * 0.55, 0] },
      materialSlots: gltf.materials.map((m, i) => m.name ?? `material-${i}`),
      occlusionGroup: `${id}-review`,
      statistics: {
        accessorBytes: runtimeBytes.length,
        animationClips: [],
        drawCalls: primitives.length,
        materials: gltf.materials.length,
        meshes: gltf.meshes.length,
        nodes: gltf.nodes.length,
        gpuBytes: runtimeBytes.length,
        textures: gltf.textures?.length ?? 0,
        triangles,
        gpuMiB: runtimeBytes.length / 1024 ** 2,
        budget,
        bytes: runtimeBytes.length,
        extent,
        sha256
      }
    })
    receipts.push({
      id: `building:${species}`,
      modelId: id,
      sourceHeightMetres: heightMetres,
      placedHeightMetres: heightMetres * BUILDING_SCALE,
      source,
      uri,
      sha256,
      sourceSha256: createHash('sha256').update(bytes).digest('hex'),
      status: 'review_only_pending_geometry_material_and_entrance_acceptance',
      boundsMetres: bounds,
      entrance: 'Path ends at the reserved forecourt; authored doorway not yet surveyed.'
    })
  }
}
