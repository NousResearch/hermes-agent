import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildReviewBuildings } from './build-review-buildings.mjs'
import { buildReviewEnvironment } from './build-review-environment.mjs'
import { DISTRICT_LAYOUT, PEDESTRIAN_ROUTES, doorwayPoint } from './settlement-layout.mjs'

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const assets = resolve(desktop, 'public/lunar-city')
const output = resolve(assets, 'v2-review')
const manifest = JSON.parse(await readFile(resolve(assets, 'v2/world-manifest.v2.json'), 'utf8'))
await mkdir(resolve(output, 'models'), { recursive: true })
await mkdir(resolve(output, 'textures'), { recursive: true })
const uris = new Set([
  ...manifest.models.map(model => model.uri),
  manifest.navigation.meshUri,
  ...manifest.textures.map(texture => texture.uri)
])
for (const uri of uris) await copyFile(resolve(assets, 'v2', uri), resolve(output, uri))
const roster = [
  ['owl', 'owl-librarian', 1.5, 'owl-detail-review.glb'],
  ['elephant', 'elephant-memory', 2.3],
  ['cat', 'cat-arts', 1.6, 'walk-contact-review/rigged-review.glb'],
  ['fox', 'fox-scientist', 1.75, 'material-review-v4.glb'],
  ['capybara', 'capybara-revenue', 1.7],
  ['lion', 'lion-steward', 2],
  ['beaver', 'beaver-architect', 1.6, 'material-review-v4.glb'],
  ['monkey', 'monkey-poet', 1.65, 'material-review-v4.glb']
]
const receipts = []
manifest.reviewLeaderAssets = []
manifest.camera.overview.minRadius = 3
manifest.camera.overview.radius = 110
manifest.camera.overview.beta = 0.82
for (const model of manifest.models.filter(model => !['workers', 'leaders'].includes(model.id))) {
  // Overview must retain actual district silhouettes and connected walkways.
  for (const lod of model.lods ?? []) if (lod.distance > 0) lod.distance *= 2
}
const leaderDistricts = ['library', 'archive', 'arts-studio', 'research-lab', 'revenue', 'council', 'engineering-workshop', 'publishing']
const staging = leaderDistricts.map(id => doorwayPoint(DISTRICT_LAYOUT.find(d => d.id === id)))
function workerStage(routeIndex) {
  const points = PEDESTRIAN_ROUTES[routeIndex].points
  const segments = points.slice(1).map((p, i) => [points[i], p])
  segments.sort(([a, b], [c, d]) => Math.hypot(d[0]-c[0], d[2]-c[2]) - Math.hypot(b[0]-a[0], b[2]-a[2]))
  const [a, b] = segments[0]
  return a.map((v, i) => (v + b[i]) / 2)
}

for (const [index, [id, folder, heightMetres, file = 'material-review.glb']] of roster.entries()) {
  const bytes = await readFile(resolve(assets, 'multiview-2026-09-07', folder, file))
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const uri = `models/review-${id}-${sha256.slice(0, 12)}.glb`
  receipts.push({ id, source: `${folder}/${file}`, uri, sha256, status: 'review_only' })
  await copyFile(resolve(assets, 'multiview-2026-09-07', folder, file), resolve(output, uri))
  manifest.reviewLeaderAssets.push({ id, uri, heightMetres, position: staging[index], rotationY: 0 })
}
manifest.characterAssets.leaders = roster.map(([id]) => ({
  id,
  species: id,
  silhouetteId: `card-${id}`,
  visualId: `leader:${id}`
}))
manifest.reviewWorkerAssets = []
for (const [id, position] of [['baseline', workerStage(0)], ['ci-repair-triage', workerStage(4)]]) {
  const source = `worker-multiview-2026-09-07/${id}/rigged-review.glb`
  const bytes = await readFile(resolve(assets, source))
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const uri = `models/review-worker-${id}-${sha256.slice(0, 12)}.glb`
  await copyFile(resolve(assets, source), resolve(output, uri))
  manifest.reviewWorkerAssets.push({ id, uri, heightMetres: 1.2, position, rotationY: 0 })
  receipts.push({ id: `worker:${id}`, source, uri, sha256, status: 'rig_review_only' })
}
await buildReviewEnvironment(manifest, output)
await buildReviewBuildings(manifest, assets, output, receipts)
// Collider bounds follow the final source bounds and placement, including scaled review assets.
for (const model of manifest.models) {
  if (['terrain', 'workers', 'leaders'].includes(model.id) || !model.bounds) continue
  const { position, scale, rotation } = model.transform
  const localCenter = model.bounds.min.map((v, i) => (v + model.bounds.max[i]) * .5 * scale[i])
  const halfExtents = model.bounds.min.map((v, i) => (model.bounds.max[i] - v) * .5 * Math.abs(scale[i]))
  if (halfExtents.some(v => !Number.isFinite(v) || v <= 0)) throw Error(`Invalid collider bounds: ${model.id}`)
  const yaw = rotation[1], c = Math.cos(yaw), s = Math.sin(yaw)
  manifest.navigation.colliders.push({
    id: `building:${model.id}`, kind: 'box', rotationY: yaw, halfExtents,
    center: [position[0] + c * localCenter[0] + s * localCenter[2], position[1] + localCenter[1], position[2] - s * localCenter[0] + c * localCenter[2]]
  })
}
manifest.generatedAssetPack.reviewStatus =
  'Review only: eight card-derived building candidates; distinct operational fixtures retained. Static model quality and final animation acceptance remain pending.'
await writeFile(resolve(output, 'world-manifest.v2.json'), JSON.stringify(manifest, null, 2) + '\n')
await writeFile(resolve(output, 'review-sources.json'), JSON.stringify(receipts, null, 2) + '\n')
console.log(`Built opt-in eight-leader review pack: ${output}`)
