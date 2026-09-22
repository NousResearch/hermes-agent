import assert from 'node:assert/strict'
import test from 'node:test'
import { NullEngine, Scene } from './modeling/babylon.mjs'
import { buildTerrain } from './modeling/terrain.mjs'
import { DISTRICT_LAYOUT, HORIZONTAL_SCALE, arrivalPoint } from './settlement-layout.mjs'

test('district perimeter trim stays flat below the walkable deck and clear of the approach', () => {
  const engine = new NullEngine(), scene = new Scene(engine)
  try {
    const root = buildTerrain(scene)
    root.scaling.set(HORIZONTAL_SCALE, 1, HORIZONTAL_SCALE)
    for (const [index, district] of DISTRICT_LAYOUT.entries()) {
      // The waterworks court is open to its channel rather than encircled by trim.
      if (district.id === 'engineering-workshop') continue
      const ring = scene.getMeshByName(`terrain:district-ring:${index}`)
      ring.computeWorldMatrix(true)
      const bounds = ring.getBoundingInfo().boundingBox
      const deckY = arrivalPoint(district)[1]
      assert.ok(bounds.maximumWorld.y <= deckY + 1e-5, `${district.id} trim protrudes into pedestrian space`)
      assert.ok(bounds.minimumWorld.y >= deckY - 0.2, `${district.id} trim is not horizontal at deck level`)
    }
  } finally { scene.dispose(); engine.dispose() }
})
