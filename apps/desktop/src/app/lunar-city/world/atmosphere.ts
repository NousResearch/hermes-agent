import type { DirectionalLight } from '@babylonjs/core/Lights/directionalLight'
import type { HemisphericLight } from '@babylonjs/core/Lights/hemisphericLight'
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial'
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color'
import { Vector3 } from '@babylonjs/core/Maths/math.vector'
import { Mesh } from '@babylonjs/core/Meshes/mesh'
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder'
import { TransformNode } from '@babylonjs/core/Meshes/transformNode'
import type { Scene } from '@babylonjs/core/scene'

import { createAtmosphereClock } from './atmosphere-cycle'

const phaseControls = new WeakMap<Scene, (phase: number) => void>()

/** Local review control for deterministic day/night screenshots; never reads wall time. */
export function setAtmospherePresentationPhase(scene: Scene, phase: number) {
  const set = phaseControls.get(scene)

  if (!set) {throw new Error('Scene has no Lunar City atmosphere')}
  set(phase)
}

/** Distant scenery has no collision, selection, navigation or operational meaning. */
export function createAtmosphere(scene: Scene) {
  const clock = createAtmosphereClock()
  const root = new TransformNode('lunar-city:atmosphere', scene)

  const material = (name: string, color: Color3) => {
    const result = new StandardMaterial(`lunar-city:atmosphere:${name}`, scene)
    result.disableLighting = true
    result.emissiveColor = color
    result.specularColor = Color3.Black()
    result.fogEnabled = false

    return result
  }

  const skyMaterial = material('sky', new Color3(.2, .39, .53))
  skyMaterial.backFaceCulling = false
  skyMaterial.disableDepthWrite = true
  const sky = MeshBuilder.CreateSphere('lunar-city:sky', { diameter: 900, segments: 12, sideOrientation: Mesh.BACKSIDE }, scene)
  sky.parent = root; sky.material = skyMaterial; sky.isPickable = false; sky.infiniteDistance = true
  const ridgeMaterials = [material('far-ridge', new Color3(.28, .4, .43)), material('near-ridge', new Color3(.18, .31, .33))]

  for (let ring = 0; ring < 2; ring++) {
    for (let index = 0; index < 22; index++) {
      const angle = index * Math.PI * 2 / 22 + ring * .11
      const height = 20 + 15 * (Math.sin(index * 2.37 + ring) * .5 + .5)
      const radius = 245 + ring * 25
      const peak = MeshBuilder.CreateCylinder(`lunar-city:ridge:${ring}:${index}`, { height, diameterBottom: 78, diameterTop: 3, tessellation: 5 }, scene)
      peak.parent = root; peak.material = ridgeMaterials[ring]; peak.isPickable = false
      peak.position.set(Math.cos(angle) * radius, height / 2 - 14, Math.sin(angle) * radius)
      peak.rotation.y = angle
    }
  }

  const sunMaterial = material('sun', new Color3(1, .83, .5))
  const sun = MeshBuilder.CreateSphere('lunar-city:backdrop-sun', { diameter: 13, segments: 12 }, scene)
  sun.parent = root; sun.material = sunMaterial; sun.isPickable = false
  const moonMaterial = material('moon', new Color3(.62, .76, .88))
  const moon = MeshBuilder.CreateSphere('lunar-city:backdrop-moon', { diameter: 20, segments: 12 }, scene)
  moon.parent = root; moon.material = moonMaterial; moon.isPickable = false
  const cloudMaterial = material('cloud', new Color3(.63, .74, .76))
  cloudMaterial.alpha = .3

  const clouds = Array.from({ length: 6 }, (_, index) => {
    const cloud = MeshBuilder.CreateSphere(`lunar-city:cloud:${index}`, { diameter: 1, segments: 8 }, scene)
    cloud.parent = root; cloud.material = cloudMaterial; cloud.isPickable = false
    cloud.scaling.set(36 + index * 3, 3, 12)

    return cloud
  })

  const key = scene.getLightByName('lunar-city:key-light') as DirectionalLight | null
  const fill = scene.getLightByName('lunar-city:fill-light') as HemisphericLight | null
  const daySky = new Color3(.24, .43, .56), nightSky = new Color3(.035, .055, .13)

  const update = (elapsedMs: number, paused: boolean) => {
    const state = clock.advance(elapsedMs, paused)
    const angle = state.phase * Math.PI * 2
    const skyColor = Color3.Lerp(nightSky, daySky, state.daylight)
    skyMaterial.emissiveColor = skyColor
    scene.clearColor = new Color4(skyColor.r, skyColor.g, skyColor.b, 1)
    scene.fogColor = Color3.Lerp(new Color3(.07, .1, .17), new Color3(.33, .45, .46), state.daylight)
    ridgeMaterials[0].emissiveColor = Color3.Lerp(new Color3(.065, .095, .14), new Color3(.25, .38, .4), state.daylight)
    ridgeMaterials[1].emissiveColor = Color3.Lerp(new Color3(.055, .085, .12), new Color3(.18, .31, .32), state.daylight)
    sun.position.set(Math.cos(angle) * 260, state.elevation * 230, -110)
    moon.position.set(-Math.cos(angle) * 250, -state.elevation * 210, 130)

    if (key) {
      key.intensity = state.keyIntensity
      key.diffuse = Color3.Lerp(new Color3(.58, .68, .95), new Color3(1, .9, .74), state.daylight)
      key.direction = new Vector3(-Math.cos(angle) * .6, -Math.max(.3, Math.abs(state.elevation)), .35)
    }

    if (fill) {fill.intensity = state.fillIntensity}
    clouds.forEach((cloud, index) => {
      const drift = angle * .2 + index * Math.PI / 3
      cloud.position.set(Math.cos(drift) * 170, 68 + index * 4, Math.sin(drift) * 170)
    })
  }

  phaseControls.set(scene, phase => { clock.setPhase(phase); update(0, true) })
  update(0, true)

  return { update, dispose: () => { phaseControls.delete(scene); root.dispose(false, true) } }
}
