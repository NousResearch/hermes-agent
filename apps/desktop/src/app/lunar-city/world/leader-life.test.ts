import { expect, it, vi } from 'vitest'

import type { Vec3 } from '../model'

import { createEntityPositionState } from './entity-position-state'
import { createLeaderLife, type LeaderLifeActor, measuredLeaderEnvelope, safeLeaderStrollPoints } from './leader-life'

function actor(): LeaderLifeActor {
  return {
    id: 'owl', position: { x: 0, y: 0, z: 0 }, home: { x: 0, y: 0, z: 0 }, work: { x: 2, y: .8, z: 0 }, strollPoints: [{ x: 1, y: 0, z: 0 }], canWalk: true,
    query: { computePath: (from, to) => [from, to], resolvePosition: point => ({ ...point }), canTraverse: () => true },
    setPosition: vi.fn(), setTraveling: vi.fn()
  }
}

it('holds exact position and intent for pause, dialogue and reduced motion, then completes safe local work travel', () => {
  const life = createLeaderLife(), value = actor()
  expect(life.register(value)).toBe(true)
  life.setMode('owl', 'work')
  life.tick(1)
  life.tick(250)
  const actual = life.get('owl')!.position
  expect(actual.x).toBeGreaterThan(0)
  expect(actual.y).toBeGreaterThan(0)

  for (const setHold of [(hold: boolean) => life.setPaused(hold), (hold: boolean) => life.setReducedMotion(hold), (hold: boolean) => life.hold('owl', hold)]) {
    setHold(true)
    life.tick(10000)
    expect(life.get('owl')?.position).toEqual(actual)
    expect(life.nextWakeDelayMs()).toBeUndefined()
    setHold(false)
    life.tick(1)
  }

  for (let step = 0; step < 20; step++) {life.tick(250)}
  expect(life.get('owl')).toMatchObject({ mode: 'work', status: 'at-work-anchor', position: value.work, provenance: 'local-authored' })
  expect(value.position.x).toBe(0)
  expect(life.nextWakeDelayMs()).toBeUndefined()
  life.dispose()
})

it('never slides an unrigged leader or crosses rejected geometry and preserves identity during rebinding', () => {
  const envelope = measuredLeaderEnvelope([{minimum:{x:-.9,y:2,z:-.5},maximum:{x:.9,y:4.3,z:.5}}], 2)!
  expect(envelope.radius).toBeCloseTo(Math.hypot(1.8, 1) / 2 + .05)
  expect(measuredLeaderEnvelope([{minimum:{x:-.9,y:2,z:-.5},maximum:{x:.9,y:4.3,z:.5}}],2,1.2)?.radius).toBeCloseTo(1.25)
  expect(envelope.height).toBeCloseTo(2.3)
  expect(measuredLeaderEnvelope([], 2)).toBeUndefined()
  const life = createLeaderLife(), value = actor()
  value.canWalk = false
  life.register(value)
  life.setMode('owl', 'work')
  life.tick(100000)
  expect(life.get('owl')).toMatchObject({ status: 'locomotion-unavailable', position: value.position })
  expect(life.nextWakeDelayMs()).toBeUndefined()
  const replacement = { ...value, canWalk: true, position: { x: 50, y: 0, z: 0 } }
  life.register(replacement)
  expect(life.get('owl')?.position).toEqual(value.position)
  let blocked = false
  replacement.query.canTraverse = (_from: Vec3, _to: Vec3) => !blocked
  life.tick(1)
  life.tick(250)
  const safe = life.get('owl')!.position
  blocked = true
  life.tick(250)
  expect(life.get('owl')).toMatchObject({ status: 'route-unavailable', position: safe })
  expect(value.setTraveling).toHaveBeenLastCalledWith(false)
  life.dispose()
})

it('restores only safe scoped positions across remount and chooses complete traversable town routes', () => {
  const value = actor()
  value.positionState = createEntityPositionState({scope:'leader-life-remount-test',maxEntries:8})
  const first = createLeaderLife()
  first.register(value)
  first.setMode('owl','work')
  first.tick(1)
  first.tick(250)
  const actual = first.get('owl')!.position
  first.dispose()
  const remount = createLeaderLife()
  remount.register(value)
  expect(remount.get('owl')?.position).toEqual(actual)
  expect(safeLeaderStrollPoints(value.query,value.home,[value.work,{x:3,y:0,z:0},{x:4,y:0,z:0},{x:5,y:0,z:0}])).toHaveLength(3)
  expect(safeLeaderStrollPoints({...value.query,computePath:(from)=>[from]},value.home,[value.work])).toEqual([])
  remount.dispose()
  const blocked = createLeaderLife()
  blocked.register({...value,query:{...value.query,resolvePosition:point=>point.x>0 ? undefined : point}})
  expect(blocked.get('owl')?.position).toEqual(value.position)
  blocked.remove('owl')
})
