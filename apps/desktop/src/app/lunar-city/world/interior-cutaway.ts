import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial'
import { Color3 } from '@babylonjs/core/Maths/math.color'
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder'
import { TransformNode } from '@babylonjs/core/Meshes/transformNode'
import type { Scene } from '@babylonjs/core/scene'

export interface InteriorSolid {
  id: string
  kind: string
  center: readonly number[]
  size: readonly number[]
}

export interface InteriorPlan {
  id: string
  title: string
  exteriorTransform: { position: readonly number[]; rotation: readonly number[] }
  solids: readonly InteriorSolid[]
  ramps?: readonly { id: string; start: readonly number[]; end: readonly number[]; width: number; thickness: number }[]
}

export interface CutawayExteriorMesh {
  visibility: number
  material?: { alpha?: number } | null
  isDisposed?(): boolean
}

export interface InteriorCutawayController {
  setBuilding(id: string | undefined): boolean
  refreshExteriorVisibility(): void
  dispose(): void
}

/** Presentation only: no mesh enablement, collision flags, navigation or backend writes. */
export function createInteriorCutaway(
  scene: Scene,
  plans: readonly InteriorPlan[],
  getExteriorMeshes: (id: string) => readonly CutawayExteriorMesh[],
  worldManifestSha256?: string
): InteriorCutawayController {
  const byId = new Map(plans.map(plan => [plan.id, plan]))
  const hidden = new Map<CutawayExteriorMesh, number>()
  let selected: string | undefined
  let root: TransformNode | undefined
  let materials: StandardMaterial[] = []
  let disposed = false

  function clear() {
    for (const [mesh, visibility] of hidden) {
      if (!mesh.isDisposed?.()) {mesh.visibility = visibility}
    }

    hidden.clear()
    root?.dispose()
    root = undefined
    materials.forEach(material => material.dispose())
    materials = []
  }

  function refreshExteriorVisibility() {
    if (disposed || !selected) {return}

    for (const mesh of getExteriorMeshes(selected)) {
      if (mesh.isDisposed?.() || (mesh.material?.alpha ?? 1) < 1) {continue}

      if (!hidden.has(mesh)) {hidden.set(mesh, mesh.visibility)}
      mesh.visibility = 0
    }
  }

  return {
    setBuilding(id) {
      if (disposed) {return false}

      if (id === selected) {refreshExteriorVisibility();

 return id !== undefined}

      clear()
      selected = undefined
      const plan = id === undefined ? undefined : byId.get(id)

      if (!plan) {return false}
      selected = plan.id
      root = new TransformNode(`interior-cutaway:${plan.id}`, scene)
      root.position.set(...plan.exteriorTransform.position as [number, number, number])
      root.rotation.y = plan.exteriorTransform.rotation[1]
      root.metadata = { interiorPrototype: true, buildingId: plan.id, worldManifestSha256 }
      const colors = { floor: '#b5c5bd', wall: '#ebdec5', desk: '#91704c' }

      const palette = Object.fromEntries(Object.entries(colors).map(([kind, color]) => {
        const material = new StandardMaterial(`interior:${plan.id}:${kind}`, scene)
        material.diffuseColor = Color3.FromHexString(color)
        material.specularColor = Color3.Black()
        materials.push(material)

        return [kind, material]
      }))

      for (const solid of plan.solids) {
        if (solid.kind === 'ceiling' || solid.kind === 'header') {continue}
        const [width, originalHeight, depth] = solid.size
        const height = solid.kind === 'wall' ? Math.min(0.9, originalHeight) : originalHeight
        const mesh = MeshBuilder.CreateBox(`interior:${plan.id}:${solid.id}`, { width, height, depth }, scene)
        mesh.parent = root
        mesh.position.set(solid.center[0], solid.center[1] - (originalHeight - height) / 2, solid.center[2])
        mesh.material = palette[solid.kind] ?? palette.floor
        mesh.isPickable = false
        mesh.metadata = { interiorPrototype: true, buildingId: plan.id, solidId: solid.id }
      }

      for (const ramp of plan.ramps ?? []) {
        const dx = ramp.end[0] - ramp.start[0], dy = ramp.end[1] - ramp.start[1], dz = ramp.end[2] - ramp.start[2]
        const horizontal = Math.hypot(dx, dz), length = Math.hypot(horizontal, dy)

        if (!horizontal) {continue}
        const mesh = MeshBuilder.CreateBox(`interior:${plan.id}:${ramp.id}`, { width: ramp.width, height: ramp.thickness, depth: length }, scene)
        mesh.parent = root
        mesh.rotation.set(-Math.atan2(dy, horizontal), Math.atan2(dx, dz), 0)
        // The authored endpoints lie on the walking surface, above the ramp's solid centre.
        mesh.position.set(
          (ramp.start[0] + ramp.end[0]) / 2 + dx * dy / (horizontal * length) * ramp.thickness / 2,
          (ramp.start[1] + ramp.end[1]) / 2 - horizontal / length * ramp.thickness / 2,
          (ramp.start[2] + ramp.end[2]) / 2 + dz * dy / (horizontal * length) * ramp.thickness / 2
        )
        mesh.material = palette.floor
        mesh.isPickable = false
        mesh.metadata = { interiorPrototype: true, buildingId: plan.id, rampId: ramp.id }
      }

      refreshExteriorVisibility()

      return true
    },
    refreshExteriorVisibility,
    dispose() {
      if (disposed) {return}
      clear()
      selected = undefined
      disposed = true
    }
  }
}
