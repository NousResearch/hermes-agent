import type { EntityKey, LeaderId, Vec3 } from '../model'

import type { EntityPositionState } from './entity-position-state'
import { createNavigationController, type NavigationEntity, type NavigationQuery } from './navigation'

export type LeaderLifeMode = 'home' | 'work' | 'idle' | 'unavailable'
export type LeaderLifeStatus = 'at-home' | 'at-work-anchor' | 'idle' | 'moving' | 'held' | 'locomotion-unavailable' | 'route-unavailable'

export interface LeaderLifePresentation {
  manualOverride?: boolean
  id: LeaderId
  mode: LeaderLifeMode
  status: LeaderLifeStatus
  position: Vec3
  canWalk: boolean
  provenance: 'local-authored'
}

export interface LeaderLifeActor {
  id: LeaderId
  position: Vec3
  home: Vec3
  work: Vec3
  strollPoints: readonly Vec3[]
  /** True only when this actual imported asset exposes a usable walk clip. */
  canWalk: boolean
  positionState?: EntityPositionState
  /** Must include ground resolution and traversal using the leader's physical envelope. */
  query: NavigationQuery
  setPosition(position: Vec3): void
  setFacing?(yaw: number): void
  setTraveling?(moving: boolean): void
}

/** One-time imported rest-pose bounds; no per-frame vertex or rig evaluation. */
export function measuredLeaderEnvelope(bounds: readonly { minimum: Vec3; maximum: Vec3 }[], declaredHeight: number, radialExtent?: number): { radius: number; height: number } | undefined {
  if (!bounds.length) {return undefined}
  const low = {x: Infinity, y: Infinity, z: Infinity}, high = {x: -Infinity, y: -Infinity, z: -Infinity}

  for (const bound of bounds) {
    for (const axis of ['x', 'y', 'z'] as const) {
      if (!Number.isFinite(bound.minimum[axis]) || !Number.isFinite(bound.maximum[axis])) {return undefined}
      low[axis] = Math.min(low[axis], bound.minimum[axis])
      high[axis] = Math.max(high[axis], bound.maximum[axis])
    }
  }

  return {radius: (radialExtent ?? Math.hypot(high.x - low.x, high.z - low.z) / 2) + .05, height: Math.max(declaredHeight, high.y - low.y)}
}

interface LeaderRecord {
  actor: LeaderLifeActor
  entity: NavigationEntity
  navigation: ReturnType<typeof createNavigationController>
  mode: LeaderLifeMode
  status: LeaderLifeStatus
  held: boolean
  pending: boolean
  target: Vec3
  resumeTarget?: Vec3
  waitMs: number
  strollIndex: number
  returningHome: boolean
  traveling: boolean
}

const LEADERS = new Set<LeaderId>(['owl', 'fox', 'elephant', 'cat', 'capybara', 'lion', 'beaver', 'monkey'])
const distance = (a: Vec3, b: Vec3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
const cadence = (id: LeaderId) => 24_000 + [...id].reduce((sum, char) => sum * 31 + char.charCodeAt(0), 0) % 16_000

/** Choose only complete, physically traversable town routes for local idle presentation. */
export function safeLeaderStrollPoints(query: NavigationQuery, origin: Vec3, candidates: readonly Vec3[]): Vec3[] {
  return [...candidates].sort((a,b) => distance(origin,a)-distance(origin,b)).filter(target => {
    if (distance(origin,target)<.25 || !query.resolvePosition?.(target)) {return false}
    const path = query.computePath(origin,target)

    if (!path?.length || distance(path[path.length-1]!,target)>.05) {return false}
    const route = [origin,...path,target]

    return route.slice(1).every((point,index)=>query.canTraverse?.(route[index]!,point)===true)
  }).slice(0,3).map(point=>({...point}))
}

/** Authored local staging only: no session activity, source status or missing animation is invented. */
export function createLeaderLife() {
  const records = new Map<LeaderId, LeaderRecord>()
  let paused = false
  let reduced = false
  let disposed = false

  const travel = (record: LeaderRecord, moving: boolean) => {
    if (record.traveling !== moving) {
      record.traveling = moving
      record.actor.setTraveling?.(moving)
    }
  }

  const hold = (record: LeaderRecord) => {
    if (record.navigation.isMoving(record.entity.key)) {record.resumeTarget = { ...record.target }}
    record.navigation.cancel(record.entity.key)
    record.pending = true
    record.status = 'held'
    travel(record, false)
  }

  const move = (record: LeaderRecord) => {
    record.pending = false

    if (record.mode === 'unavailable') { record.status = 'held';

 return }

    const target = record.resumeTarget ?? (record.mode === 'work' ? record.actor.work : record.mode === 'home' || record.returningHome
      ? record.actor.home : record.actor.strollPoints[record.strollIndex++ % Math.max(1, record.actor.strollPoints.length)] ?? record.actor.home)

    record.resumeTarget = undefined
    record.target = { ...target }

    if (distance(record.entity.position, target) < .01) {
      record.status = record.mode === 'work' ? 'at-work-anchor' : record.mode === 'home' ? 'at-home' : 'idle'
      record.waitMs = cadence(record.actor.id)

      return
    }

    if (!record.actor.canWalk) { record.status = 'locomotion-unavailable';

 return }

    if (!record.navigation.move(record.entity, 'project', 'idle')) {
      record.status = 'route-unavailable'
      record.waitMs = cadence(record.actor.id)

      return
    }

    record.status = 'moving'
    travel(record, true)
  }

  return {
    register(actor: LeaderLifeActor): boolean {
      if (disposed || !LEADERS.has(actor.id) || !actor.query.resolvePosition || !actor.query.canTraverse) {return false}
      const previous = records.get(actor.id)

      if (previous) {
        if (!actor.query.resolvePosition(previous.entity.position)) {
          travel(previous, false)
          previous.navigation.dispose()
          records.delete(actor.id)

          return false
        }

        // Identity remains fixed; rebinding a view never resets the actor's actual position.
        previous.actor.setTraveling?.(false)
        previous.actor = actor
        hold(previous)
        actor.setPosition({ ...previous.entity.position })

        return true
      }

      const key = `lunar-city:leader:${actor.id}` as EntityKey
      const restored = actor.positionState?.read(key)?.position
      const position = (restored && actor.query.resolvePosition(restored)) ?? actor.query.resolvePosition(actor.position)

      if (!position) {return false}
      const entity: NavigationEntity = { key, position: { ...position }, animation: 'idle' }
      let record: LeaderRecord

      const navigation = createNavigationController({
        destinations: { project: actor.home },
        targetFor: () => record.target,
        query: {
          computePath: (from, to) => record.actor.query.computePath(from, to),
          resolvePosition: point => record.actor.query.resolvePosition?.(point),
          canTraverse: (from, to) => record.actor.query.canTraverse?.(from, to) === true
        },
        workerClips: new Set(['idle', 'walk']),
        speedUnitsPerSecond: .8
      })

      record = { actor, entity, navigation, mode: 'idle', status: 'idle', held: false, pending: false, target: { ...actor.home }, waitMs: cadence(actor.id), strollIndex: 0, returningHome: false, traveling: false }
      records.set(actor.id, record)
      actor.positionState?.save(key, {position}, true)
      actor.setPosition({ ...position })

      return true
    },
    setMode(id: LeaderId, mode: LeaderLifeMode): void {
      const record = records.get(id)

      if (!record || record.mode === mode) {return}
      hold(record)
      record.mode = mode
      record.resumeTarget = undefined
      record.returningHome = false
      record.waitMs = mode === 'idle' ? cadence(id) : 0
      record.pending = mode !== 'idle'
    },
    hold(id: LeaderId, held: boolean): void {
      const record = records.get(id)

      if (!record || record.held === held) {return}
      record.held = held
      hold(record)
    },
    setPaused(value: boolean): void {
      if (paused === value) {return}
      paused = value

      for (const record of records.values()) {hold(record)}
    },
    setReducedMotion(value: boolean): void {
      if (reduced === value) {return}
      reduced = value

      for (const record of records.values()) {hold(record)}
    },
    tick(elapsedMs: number): boolean {
      if (disposed || paused || reduced || !Number.isFinite(elapsedMs) || elapsedMs <= 0) {return false}
      const delta = elapsedMs

      for (const record of records.values()) {
        if (record.held || record.mode === 'unavailable') {continue}

        if (record.navigation.isMoving(record.entity.key)) {
          const before = { ...record.entity.position }
          record.navigation.tick(delta)

          if (distance(before, record.entity.position) > 1e-8) {
            record.actor.setPosition({ ...record.entity.position })
            record.actor.positionState?.save(record.entity.key, {position: record.entity.position}, true)
            record.actor.setFacing?.(Math.atan2(record.entity.position.x - before.x, record.entity.position.z - before.z))
          }

          if (!record.navigation.isMoving(record.entity.key)) {
            travel(record, false)
            record.status = record.navigation.didArrive(record.entity.key) ? record.mode === 'work' ? 'at-work-anchor' : record.mode === 'home' ? 'at-home' : 'idle' : 'route-unavailable'
            record.returningHome = !record.returningHome
            record.waitMs = cadence(record.actor.id)
          }
        } else {
          record.waitMs -= delta

          if (record.pending || (record.mode === 'idle' && record.waitMs <= 0)) {move(record)}
        }
      }

      return [...records.values()].some(record => !record.held && (record.pending || record.traveling))
    },
    nextWakeDelayMs(): number | undefined {
      if (disposed || paused || reduced) {return undefined}

      const delays = [...records.values()].filter(record => !record.held && record.mode !== 'unavailable' && record.actor.canWalk)
        .flatMap(record => record.pending || record.traveling ? [0] : record.mode === 'idle' ? [Math.max(0, record.waitMs)] : [])

      return delays.length ? Math.min(...delays) : undefined
    },
    get(id: LeaderId): LeaderLifePresentation | undefined {
      const record = records.get(id)

      return record ? { id, mode: record.mode, status: record.status, position: { ...record.entity.position }, canWalk: record.actor.canWalk, provenance: 'local-authored' as const } : undefined
    },
    remove(id: LeaderId): void {
      const record = records.get(id)

      if (record) { travel(record, false); record.actor.positionState?.forget(record.entity.key); record.navigation.dispose(); records.delete(id) }
    },
    dispose(): void {
      disposed = true

      for (const record of records.values()) { travel(record, false); record.actor.positionState?.save(record.entity.key, {position: record.entity.position}, true); record.navigation.dispose() }
      records.clear()
    }
  }
}
