import { NullEngine } from '@babylonjs/core/Engines/nullEngine'
import { CreateBox } from '@babylonjs/core/Meshes/Builders/boxBuilder'
import type { Scene } from '@babylonjs/core/scene'
import { expect, it } from 'vitest'

import { loadBabylonModules } from './create-world'

it('registers real Babylon ray picking when loading the modular world runtime', async () => {
  const modules = await loadBabylonModules()

  const engine = new NullEngine({
    renderWidth: 64,
    renderHeight: 64,
    textureSize: 64,
    deterministicLockstep: false,
    lockstepMaxSteps: 4
  })

  const sceneLike = new modules.Scene(engine)
  const scene = sceneLike as unknown as Scene

  try {
    new modules.ArcRotateCamera('pick-camera', Math.PI / 2, Math.PI / 2, 5, new modules.Vector3(0, 0, 0), sceneLike)
    const box = CreateBox('physical-asset', {}, scene)
    scene.render()
    expect(scene.pick(32, 32)?.pickedMesh).toBe(box)
  } finally {
    scene.dispose()
    engine.dispose()
  }
})
