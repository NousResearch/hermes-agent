import { describe, expect, it, vi } from 'vitest'

import type { EntityKey, Vec3 } from '../model'

import {
  createNavigationController,
  createRecastNavigationQuery,
  type NavigationEntity,
  type NavigationQuery
} from './navigation'

const workerKey = 'session:worker' as EntityKey

function worker(position: Vec3 = { x: 1, y: 0, z: 2 }): NavigationEntity {
  return { animation: 'idle', key: workerKey, position: { ...position } }
}

function query(
  path: readonly Vec3[] = [
    { x: 1, y: 0, z: 2 },
    { x: 9, y: 0, z: 2 }
  ]
): NavigationQuery {
  return { computePath: vi.fn(() => path), dispose: vi.fn() }
}

const destinations = { review: { x: 9, y: 0, z: 2 } }

describe('NavigationController', () => {
  it('computes from the entity’s actual position, uses the declared walk clip, and advances only on ticks', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      workerClips: new Set(['idle', 'walk'])
    })

    const entity = worker()

    expect(controller.move(entity, 'review')).toBe(true)
    expect(navQuery.computePath).toHaveBeenCalledWith({ x: 1, y: 0, z: 2 }, { x: 9, y: 0, z: 2 })
    expect(entity.animation).toBe('walk')
    expect(entity.position).toEqual({ x: 1, y: 0, z: 2 })

    controller.tick(500)

    expect(entity.position.x).toBeGreaterThan(1)
    expect(entity.position.x).toBeLessThan(9)
  })

  it('does not recompute an unchanged origin, destination, and walkability revision', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      workerClips: new Set(['idle', 'walk'])
    })

    const entity = worker()

    controller.move(entity, 'review')
    controller.move(entity, 'review')
    controller.setWalkabilityRevision(1)
    controller.move(entity, 'review')

    expect(navQuery.computePath).toHaveBeenCalledTimes(2)
  })

  it.each([
    ['missing destination', {} as Record<string, Vec3>, query()],
    ['empty path', destinations, query([])],
    [
      'query failure',
      destinations,
      {
        computePath: vi.fn(() => {
          throw new Error('no navmesh')
        })
      }
    ]
  ])('fails closed for %s and never falls back to a straight line', (_name, targetDestinations, navQuery) => {
    const diagnostic = vi.fn()

    const controller = createNavigationController({
      destinations: targetDestinations,
      query: navQuery,
      workerClips: new Set(['idle', 'walk']),
      diagnostic
    })

    const entity = worker()

    expect(controller.move(entity, 'review')).toBe(false)
    controller.tick(1_000)

    expect(entity.position).toEqual({ x: 1, y: 0, z: 2 })
    expect(entity.animation).toBe('unavailable')
    expect(diagnostic).toHaveBeenCalledTimes(1)
  })

  it('uses the manifest static pose and holds actual position under reduced motion', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      reducedMotion: true,
      staticPose: 'rest',
      workerClips: new Set(['idle', 'walk', 'rest'])
    })

    const entity = worker()

    expect(controller.move(entity, 'review')).toBe(true)
    expect(entity.position).toEqual({ x: 1, y: 0, z: 2 })
    expect(entity.animation).toBe('rest')
    expect(navQuery.computePath).not.toHaveBeenCalled()
  })

  it('holds every active route in place when reduced motion becomes active', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      staticPose: 'idle',
      workerClips: new Set(['idle', 'walk'])
    })

    const entity = worker()

    controller.move(entity, 'review')
    controller.tick(250)
    expect(entity.position.x).toBeGreaterThan(1)
    expect(entity.position.x).toBeLessThan(9)

    const pausedPosition = { ...entity.position }
    controller.setReducedMotion(true)
    expect(entity.position).toEqual(pausedPosition)
    expect(entity.animation).toBe('idle')
    expect(controller.isMoving(workerKey)).toBe(false)
  })

  it('disposes its navigation query once and ignores later work', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      workerClips: new Set(['idle', 'walk'])
    })

    controller.dispose()
    controller.dispose()

    expect(controller.move(worker(), 'review')).toBe(false)
    expect(navQuery.dispose).toHaveBeenCalledOnce()
  })

  it('recomputes an active route after an authoritative origin correction and preserves the walk clip until arrival', () => {
    const navQuery = { computePath: vi.fn((from: Vec3, to: Vec3) => [{ ...from }, { ...to }]) }

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      speedUnitsPerSecond: 100,
      workerClips: new Set(['idle', 'walk', 'work', 'review'])
    })

    const entity = worker()

    expect(controller.move(entity, 'review', 'work')).toBe(true)
    entity.position = { x: 4, y: 0, z: 2 }
    expect(controller.move(entity, 'review', 'review')).toBe(true)
    expect(entity.animation).toBe('walk')
    expect(navQuery.computePath).toHaveBeenCalledTimes(2)

    controller.tick(1_000)

    expect(entity.position).toEqual(destinations.review)
    expect(entity.animation).toBe('review')
    expect(controller.isMoving(workerKey)).toBe(false)
  })

  it('cancels a removed entity path without changing its presentation after reconciliation releases it', () => {
    const navQuery = query()

    const controller = createNavigationController({
      destinations,
      query: navQuery,
      workerClips: new Set(['idle', 'walk'])
    })

    const entity = worker()

    controller.move(entity, 'review')
    controller.cancel(entity.key)
    controller.tick(1_000)

    expect(entity.position).toEqual({ x: 1, y: 0, z: 2 })
    expect(controller.isMoving(entity.key)).toBe(false)
  })
})

describe('Recast NavigationQuery adapter', () => {
  it('reads an actual Recast path and disposes the underlying navmesh once', () => {
    const destroy = vi.fn()

    const navMesh = {
      computePath: vi.fn(() => ({
        getPoint: (index: number) =>
          [
            { x: 1, y: 0, z: 2 },
            { x: 9, y: 0, z: 2 }
          ][index],
        getPointCount: () => 2
      })),
      destroy
    }

    const query = createRecastNavigationQuery(navMesh, (x, y, z) => ({ x, y, z }))

    expect(query.computePath({ x: 1, y: 0, z: 2 }, { x: 9, y: 0, z: 2 })).toEqual([
      { x: 1, y: 0, z: 2 },
      { x: 9, y: 0, z: 2 }
    ])
    query.dispose?.()
    query.dispose?.()
    expect(destroy).toHaveBeenCalledOnce()
  })

  it('releases every temporary Recast wrapper even when a malformed path fails closed', () => {
    const start = { destroy: vi.fn(), x: 1, y: 0, z: 2 }
    const end = { destroy: vi.fn(), x: 9, y: 0, z: 2 }
    const validPoint = { destroy: vi.fn(), x: 1, y: 0, z: 2 }
    const invalidPoint = { destroy: vi.fn(), x: Number.NaN, y: 0, z: 2 }

    const path = {
      destroy: vi.fn(),
      getPoint: (index: number) => [validPoint, invalidPoint][index],
      getPointCount: () => 2
    }

    const query = createRecastNavigationQuery({ computePath: vi.fn(() => path) }, (x, y, z) => (x === 1 ? start : end))

    expect(query.computePath({ x: 1, y: 0, z: 2 }, { x: 9, y: 0, z: 2 })).toBeUndefined()
    expect(start.destroy).toHaveBeenCalledOnce()
    expect(end.destroy).toHaveBeenCalledOnce()
    // NavPath owns getPoint() vectors; only the path may release them.
    expect(validPoint.destroy).not.toHaveBeenCalled()
    expect(invalidPoint.destroy).not.toHaveBeenCalled()
    expect(path.destroy).toHaveBeenCalledOnce()
  })

  it('releases the first owned vector when construction of the second vector fails', () => {
    const start = { destroy: vi.fn(), x: 1, y: 0, z: 2 }

    const query = createRecastNavigationQuery({ computePath: vi.fn() }, x => {
      if (x === 9) {
        throw new Error('out of WASM memory')
      }

      return start
    })

    expect(query.computePath({ x: 1, y: 0, z: 2 }, { x: 9, y: 0, z: 2 })).toBeUndefined()
    expect(start.destroy).toHaveBeenCalledOnce()
  })
})

it('rejects snapped routes and never uses a long frame to tunnel through a new obstacle', () => {
  const entity = worker()
  const snapped = createNavigationController({destinations,query:query([{x:4,y:0,z:2},destinations.review]),workerClips:new Set(['walk'])})
  expect(snapped.move(entity,'review')).toBe(false)
  expect(entity.position.x).toBe(1)
  let blocked = false
  const nav = createNavigationController({destinations,query:{computePath:(from,to)=>[from,to],canTraverse:()=>!blocked,resolvePosition:point=>point},workerClips:new Set(['walk','idle'])})
  expect(nav.move(entity,'review')).toBe(true)
  nav.tick(60000)
  expect(entity.position.x).toBeLessThanOrEqual(1.3)
  const safe = {...entity.position}
  blocked = true
  nav.tick(100)
  expect(entity.position).toEqual(safe)
  expect(entity.animation).toBe('unavailable')
  expect(nav.isMoving(entity.key)).toBe(false)
})
