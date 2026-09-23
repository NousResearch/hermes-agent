import { describe, expect, it } from 'vitest'

import { createOcclusionController, type OcclusionMaterial } from './occlusion'

function material(alpha = 1): OcclusionMaterial {
  return { alpha }
}

describe('createOcclusionController', () => {
  it('fades only manifest-declared roof or wall groups that obstruct the selected focus anchor', () => {
    const roofMaterial = material()
    const facadeMaterial = material()
    const terrainMaterial = material()

    const controller = createOcclusionController([
      { group: 'library-roof', material: roofMaterial, intersectsFocusRay: () => true },
      { group: 'library-wall', material: facadeMaterial, intersectsFocusRay: () => true },
      { group: 'terrain', material: terrainMaterial, intersectsFocusRay: () => true }
    ])

    controller.update(
      { position: { x: 0, y: 10, z: -20 } },
      { cameraAnchor: { x: 0, y: 1, z: 0 }, occlusionGroup: 'workers' }
    )

    expect(roofMaterial.alpha).toBeLessThan(1)
    expect(facadeMaterial.alpha).toBeLessThan(1)
    expect(terrainMaterial.alpha).toBe(1)
  })

  it('restores original values exactly when focus clears or the obstruction changes', () => {
    const roofMaterial = material(0.72)
    const sharedMaterial = material(0.91)

    const controller = createOcclusionController([
      { group: 'review-roof', material: roofMaterial, intersectsFocusRay: () => true },
      { group: 'review-roof', material: sharedMaterial, intersectsFocusRay: () => false }
    ])

    controller.update(
      { position: { x: 0, y: 10, z: -20 } },
      { cameraAnchor: { x: 0, y: 1, z: 0 }, occlusionGroup: 'workers' }
    )
    controller.clear()

    expect(roofMaterial.alpha).toBe(0.72)
    expect(sharedMaterial.alpha).toBe(0.91)
  })

  it('isolates a shared source material before fading one occluding surface', () => {
    const source = material()
    const clone = material()
    let assigned: OcclusionMaterial = source
    source.clone = () => clone

    const controller = createOcclusionController([
      {
        group: 'council-roof',
        material: source,
        isolateMaterial: true,
        assignMaterial: material => {
          assigned = material
        },
        intersectsFocusRay: () => true
      }
    ])

    controller.update(
      { position: { x: 0, y: 10, z: -20 } },
      { cameraAnchor: { x: 0, y: 1, z: 0 }, occlusionGroup: 'workers' }
    )

    expect(source.alpha).toBe(1)
    expect(clone.alpha).toBeLessThan(1)
    expect(assigned).toBe(clone)
    controller.clear()
    expect(clone.alpha).toBe(1)
    expect(assigned).toBe(source)
  })
})
