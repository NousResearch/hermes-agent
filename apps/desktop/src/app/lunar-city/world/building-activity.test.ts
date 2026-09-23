import { expect, it, vi } from 'vitest'

import { entityKey } from '../identity'
import type { DestinationId, LunarCitySnapshot } from '../model'

import { createBuildingActivity } from './building-activity'

function clip(name: string) {
  return { name, start: vi.fn(), stop: vi.fn(), reset: vi.fn() }
}

function snapshot(
  rows: readonly [DestinationId, string | undefined][],
  authority: 'authoritative' | 'stale' = 'authoritative'
): LunarCitySnapshot {
  return {
    revision: 1,
    observedAt: 1,
    sources: [],
    entities: new Map(
      rows.map(([destination, sourceState], i) => {
        const identity = { kind: 'session', connectionId: 'local', profile: 'worker', sessionId: String(i) } as const
        const key = entityKey(identity)

        return [key, { key, identity, authority, destination, sourceState, animation: 'work', observedAt: 1 }]
      })
    )
  }
}

it('stops autoplay and only runs imported matching clips for authoritative source activity', () => {
  const controller = createBuildingActivity()

  const portal = clip('portal-idle'),
    unrelated = clip('door-open')

  controller.register('review-office', [portal, unrelated])
  expect(portal.stop).toHaveBeenCalledOnce()
  expect(unrelated.stop).toHaveBeenCalledOnce()
  controller.update(
    snapshot([
      ['review', undefined],
      ['project', 'review']
    ])
  )
  expect(portal.start).not.toHaveBeenCalled()
  const current = snapshot([['review', 'review']])
  controller.update(current)
  controller.update(current)
  expect(portal.start).toHaveBeenCalledExactlyOnceWith(true)
  expect(unrelated.start).not.toHaveBeenCalled()
  controller.update(snapshot([['review', 'review']], 'stale'))
  expect(controller.isActive()).toBe(false)
  expect(portal.stop).toHaveBeenCalledTimes(2)
  controller.update(snapshot([['review', 'idle']]))
  expect(controller.isActive()).toBe(false)
  controller.register('review-office', [])
  controller.update(current)
  expect(controller.isActive()).toBe(false)
})
it('bounds playback and parks props for reduced motion, efficient quality and disposal', () => {
  const controller = createBuildingActivity()
  const models = ['review-office', 'triage', 'council', 'depot', 'research-lab', 'library']

  const names = [
    'portal-idle',
    'triage-station-idle',
    'lights-idle',
    'workbench-cycle',
    'telescope-scan',
    'lights-idle'
  ]

  const clips = names.map(clip)
  models.forEach((model, i) => controller.register(model, [clips[i]]))

  const garden = clip('garden-idle'),
    bus = clip('idle')

  controller.register('garden', [garden])
  controller.register('bus', [bus])

  const current = snapshot([
    ['garden', 'idle'],
    ['bus', 'queued'],
    ['review', 'review'],
    ['triage', 'triage'],
    ['council', 'orchestration'],
    ['depot', 'working'],
    ['lab', 'running'],
    ['library', 'working']
  ])

  controller.update(current)
  expect(controller.activeCount()).toBe(4)
  expect(garden.start).not.toHaveBeenCalled()
  expect(bus.start).not.toHaveBeenCalled()
  expect(clips[0].start).toHaveBeenCalled()
  expect(clips[1].start).toHaveBeenCalled()

  for (const disable of [() => controller.setReducedMotion(true), () => controller.setEnabled(false)]) {
    disable()
    expect(controller.isActive()).toBe(false)
    controller.update(current)
    expect(controller.isActive()).toBe(false)
    controller.setReducedMotion(false)
    controller.setEnabled(true)
    expect(controller.activeCount()).toBe(4)
  }

  controller.dispose()
  controller.update(current)
  controller.setEnabled(true)
  expect(controller.isActive()).toBe(false)
})

it('plays an observed arrival once, preempts ambient, and restores it only after imported playback ends', () => {
  const controller = createBuildingActivity()
  const ambient = clip('triage-station-idle')
  const door = { ...clip('door-open'), isPlaying: false }
  door.start.mockImplementation(() => { door.isPlaying = true })
  door.stop.mockImplementation(() => { door.isPlaying = false })
  controller.register('triage', [ambient, door])
  const current = snapshot([['triage', 'blocked']])
  const key = [...current.entities.keys()][0]
  controller.update(current)
  expect(controller.arrival(key, 'triage')).toBe(true)
  expect(door.start).toHaveBeenCalledExactlyOnceWith(false)
  expect(ambient.stop).toHaveBeenCalledTimes(2)
  controller.update(current); controller.tick()
  expect(controller.arrival(key, 'triage')).toBe(false)
  expect(ambient.start).toHaveBeenCalledTimes(1)
  door.isPlaying = false
  controller.tick()
  expect(ambient.start).toHaveBeenCalledTimes(2)
  expect(controller.arrival(key, 'triage')).toBe(false)
  controller.update({ ...current, entities: new Map([...current.entities].map(([id, entity]) => [id, { ...entity, observedAt: 2 }])) })
  expect(controller.arrival(key, 'triage')).toBe(true)
  controller.setReducedMotion(true)
  expect(door.isPlaying).toBe(false)
  expect(controller.isActive()).toBe(false)
  controller.setReducedMotion(false)
  expect(controller.arrival(key, 'triage')).toBe(false)
})

it('requires exact authoritative arrival, a real finite clip, and enabled motion', () => {
  const controller = createBuildingActivity()
  const current = snapshot([['triage', 'blocked']])
  const key = [...current.entities.keys()][0]
  controller.update(current)
  expect(controller.arrival(key, 'triage')).toBe(false)
  const door = { ...clip('door-open'), isPlaying: false }
  controller.register('triage', [door])
  expect(controller.arrival(key, 'bus')).toBe(false)
  controller.update(snapshot([['triage', 'blocked']], 'stale'))
  expect(controller.arrival(key, 'triage')).toBe(false)
  controller.update(current)
  controller.setEnabled(false)
  expect(controller.arrival(key, 'triage')).toBe(false)
  expect(door.start).not.toHaveBeenCalled()
  controller.setEnabled(true)
  controller.register('triage', [{ ...door, hasMotion: false }])
  expect(controller.arrival(key, 'triage')).toBe(false)
})
