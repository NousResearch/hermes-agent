import test from 'node:test'
import assert from 'node:assert/strict'
import { RIVER_POINTS, WATERWORKS_PORTS, riverSample } from './modeling/island-shape.mjs'
import { obstacles, ROAD_WIDTH, segmentBlocked } from './settlement-layout.mjs'

test('watercourse passes both architecture ports and stays clear of other occupied footprints', () => {
  for (const port of Object.values(WATERWORKS_PORTS)) assert.ok(riverSample(port[0], port[2]).distance < .001)
  for (const obstacle of obstacles.filter(o => o.id !== 'engineering-workshop')) {
    const bank = { ...obstacle, halfWidth: obstacle.halfWidth - ROAD_WIDTH / 2 - .15 + 2.9,
      halfDepth: obstacle.halfDepth - ROAD_WIDTH / 2 - .15 + 2.9 }
    for (let i = 1; i < RIVER_POINTS.length; i++)
      assert.equal(segmentBlocked(RIVER_POINTS[i - 1], RIVER_POINTS[i], bank), false, `River intersects ${obstacle.id}`)
  }
})
