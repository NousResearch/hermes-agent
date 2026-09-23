import { NullEngine } from '@babylonjs/core/Engines/nullEngine'
import { Vector3 } from '@babylonjs/core/Maths/math.vector'
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder'
import { Scene } from '@babylonjs/core/scene'
import { expect, it } from 'vitest'

import { createInteriorCutaway, type InteriorPlan } from './interior-cutaway'

const plan: InteriorPlan = {
  id: 'library', title: 'Library',
  exteriorTransform: { position: [10, 2, 20], rotation: [0, Math.PI / 2, 0] },
  solids: [
    { id: 'floor', kind: 'floor', center: [0, -0.075, 0], size: [8, 0.15, 8] },
    { id: 'portal-floor', kind: 'floor', center: [0, -0.075, 5], size: [1.5, 0.15, 2] },
    { id: 'wall', kind: 'wall', center: [0, 1.5, -4], size: [8, 3, 0.15] },
    { id: 'desk', kind: 'desk', center: [2, 0.4, 2], size: [1, 0.8, 2] },
    { id: 'roof', kind: 'ceiling', center: [0, 3, 0], size: [8, 0.1, 8] },
    { id: 'header', kind: 'header', center: [0, 2.6, 4], size: [1.2, 0.8, 0.15] }
  ],
  ramps: [{ id: 'ramp', start: [0, 0, 7], end: [0, 0.8, 4], width: 1.5, thickness: 0.12 }]
}

it('renders authored metre solids with yaw only, cut walls and portal collars; restores only its selected exterior on switch/dispose', () => {
  const engine = new NullEngine(), scene = new Scene(engine)
  const a = MeshBuilder.CreateBox('a', {}, scene), b = MeshBuilder.CreateBox('b', {}, scene)
  a.visibility = 0.7
  a.checkCollisions = true
  const controller = createInteriorCutaway(scene, [plan, { ...plan, id: 'other' }], id => id === 'library' ? [a] : [b])
  expect(controller.setBuilding('library')).toBe(true)
  expect(a.visibility).toBe(0)
  expect(a.checkCollisions).toBe(true)
  expect(a.isEnabled()).toBe(true)
  expect(b.visibility).toBe(1)
  const wall = scene.getMeshByName('interior:library:wall')!
  wall.computeWorldMatrix(true)
  expect(wall.getBoundingInfo().boundingBox.extendSizeWorld.y * 2).toBeCloseTo(0.9)
  expect(wall.getBoundingInfo().boundingBox.minimumWorld.y).toBeCloseTo(2)
  const portal = scene.getMeshByName('interior:library:portal-floor')!
  portal.computeWorldMatrix(true)
  expect(portal.absolutePosition.x).toBeCloseTo(15)
  const ramp = scene.getMeshByName('interior:library:ramp')!
  const rampMatrix = ramp.computeWorldMatrix(true)
  const topEnd = Vector3.TransformCoordinates(new Vector3(0, 0.06, Math.hypot(3, 0.8) / 2), rampMatrix)
  const topStart = Vector3.TransformCoordinates(new Vector3(0, 0.06, -Math.hypot(3, 0.8) / 2), rampMatrix)
  expect(topStart.x).toBeCloseTo(17)
  expect(topStart.y).toBeCloseTo(2)
  expect(topEnd.x).toBeCloseTo(14)
  expect(topEnd.y).toBeCloseTo(2.8)
  expect(scene.getMeshByName('interior:library:roof')).toBeNull()
  expect(scene.getMeshByName('interior:library:header')).toBeNull()
  const count = scene.meshes.length
  controller.setBuilding('library')
  expect(scene.meshes.length).toBe(count)
  controller.setBuilding('other')
  expect(a.visibility).toBe(0.7)
  expect(b.visibility).toBe(0)
  expect(scene.getMeshByName('interior:library:wall')).toBeNull()
  controller.dispose()
  controller.dispose()
  expect(b.visibility).toBe(1)
  expect(scene.meshes).toEqual([a, b])
  expect(scene.materials.filter(material => material.name.startsWith('interior:'))).toEqual([])
  expect(controller.setBuilding('library')).toBe(false)
  scene.dispose(); engine.dispose()
})

it('keeps transparent panes visible and captures later LOD meshes without manufacturing visibility on restoration', () => {
  const engine = new NullEngine(), scene = new Scene(engine)
  const opaque = { visibility: 1 }, pane = { visibility: 0.5, material: { alpha: 0.4 } }, hiddenLod = { visibility: 0 }
  const meshes = [opaque, pane]
  const controller = createInteriorCutaway(scene, [plan], () => meshes)
  controller.setBuilding('library')
  meshes.push(hiddenLod)
  controller.refreshExteriorVisibility()
  expect(pane.visibility).toBe(0.5)
  expect(controller.setBuilding('missing')).toBe(false)
  expect(opaque.visibility).toBe(1)
  expect(hiddenLod.visibility).toBe(0)
  expect(scene.meshes).toHaveLength(0)
  controller.dispose(); scene.dispose(); engine.dispose()
})
