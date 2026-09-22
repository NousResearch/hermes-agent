import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial'
import { Color3 } from '@babylonjs/core/Maths/math.color'
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder'
import { TransformNode } from '@babylonjs/core/Meshes/transformNode'
import type { Scene } from '@babylonjs/core/scene'

const materials = new WeakMap<Scene, StandardMaterial>()

/** A small authored site sign, not a generated building or a construction-progress claim. */
export function createProjectMarker(name: string, scene: Scene): TransformNode {
  let material = materials.get(scene)

  if (!material) {
    material = new StandardMaterial('lunar-city:project-site-sign',scene)
    material.diffuseColor = new Color3(.66,.47,.24)
    material.emissiveColor = new Color3(.1,.055,.01)
    materials.set(scene,material)
  }

  const root = new TransformNode(name,scene)

  for(const [part,width,height,depth,y] of [
    ['base',2.4,.12,1.2,.06], ['post',.12,1.8,.12,.9], ['sign',2.2,.7,.12,1.65]
  ] as const) {
    const mesh=MeshBuilder.CreateBox(`${name}:${part}`,{width,height,depth},scene)
    mesh.parent=root;mesh.position.y=y;mesh.material=material;mesh.isPickable=false
  }

  return root
}
