import assert from 'node:assert/strict'
import test from 'node:test'
import { DISTRICT_LAYOUT, PEDESTRIAN_ROUTES, doorwayPoint, obstacles, segmentBlocked } from './settlement-layout.mjs'

test('every declared district has a connected approach with building clearance and gentle ramps', () => {
  const graph = new Map(DISTRICT_LAYOUT.map(d => [d.id, new Set()]))
  for (const route of PEDESTRIAN_ROUTES) {
    graph.get(route.from).add(route.to)
    graph.get(route.to).add(route.from)
    assert.deepEqual(route.points[0], doorwayPoint(DISTRICT_LAYOUT.find(d => d.id === route.from)))
    assert.deepEqual(route.points.at(-1), doorwayPoint(DISTRICT_LAYOUT.find(d => d.id === route.to)))
    for (let i = 1; i < route.points.length; i++) {
      const a = route.points[i - 1],
        b = route.points[i]
      assert.ok(Math.abs(a[1] - b[1]) / Math.hypot(a[0] - b[0], a[2] - b[2]) <= 1 / 12 + 1e-8)
      for (const obstacle of obstacles)
        assert.equal(segmentBlocked(a, b, obstacle), false, `${route.from} to ${route.to} crosses ${obstacle.id}`)
    }
  }
  const seen = new Set(),
    queue = [DISTRICT_LAYOUT[0].id]
  while (queue.length) {
    const id = queue.shift()
    if (seen.has(id)) continue
    seen.add(id)
    queue.push(...graph.get(id))
  }
  assert.deepEqual([...seen].sort(), DISTRICT_LAYOUT.map(d => d.id).sort())
})

test('rotated building footprints reserve separate physical space', () => {
  const corners = o => {
    const c = Math.cos(o.yaw),
      s = Math.sin(o.yaw)
    return [
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1]
    ].map(([a, b]) => [
      o.x + a * (o.halfWidth - 1.65) * c + b * (o.halfDepth - 1.65) * s,
      o.z - a * (o.halfWidth - 1.65) * s + b * (o.halfDepth - 1.65) * c
    ])
  }
  for (let i = 0; i < obstacles.length; i++)
    for (let j = i + 1; j < obstacles.length; j++) {
      const a = obstacles[i],
        b = obstacles[j],
        ac = corners(a),
        bc = corners(b)
      const overlaps = [a.yaw, b.yaw]
        .flatMap(t => [
          [Math.cos(t), -Math.sin(t)],
          [Math.sin(t), Math.cos(t)]
        ])
        .every(([x, z]) => {
          const aa = ac.map(p => p[0] * x + p[1] * z),
            bb = bc.map(p => p[0] * x + p[1] * z)
          return Math.max(...aa) > Math.min(...bb) && Math.max(...bb) > Math.min(...aa)
        })
      assert.equal(overlaps, false, `${a.id} intersects ${b.id}`)
    }
})
