import { describe, expect, it } from 'vitest'

import type { WorldEvent } from './world-events'
import { resolveNpcPersonality, resolveWorldPresentation, stableEventSeed } from './world-presentation'

function event(kind: string, overrides: Partial<WorldEvent> = {}): WorldEvent {
  return {
    actionKinds: ['inspect'],
    facts: {},
    id: 'event-7',
    kind,
    occurredAt: 100,
    receivedAt: 100,
    scope: 'task',
    severity: 'warning',
    source: 'kanban',
    title: 'Fix the blocker',
    transition: true,
    ...overrides
  }
}

describe('world presentation', () => {
  it('turns a blocked card into a repairable local crisis', () => {
    const presentation = resolveWorldPresentation(
      event('task.blocked', {
        detail: 'dependency failed',
        sourceRef: { board: 'main', taskId: 'task-7' }
      }),
      []
    )

    expect(presentation).toMatchObject({ sceneTag: 'crisis.fire.local', scope: 'task' })
    expect(presentation.animationTags).toEqual(expect.arrayContaining(['task.blocked', 'repair', 'extinguish']))
    expect(presentation.actionKinds).toContain('inspect')
    expect(presentation.npcActivities.map(activity => activity.state)).toEqual(['panicking', 'repairing'])
  })

  it('escalates a critical blocker to a district fire scene', () => {
    const presentation = resolveWorldPresentation(
      event('task.blocked', { scope: 'district', severity: 'critical' }),
      []
    )

    expect(presentation.sceneTag).toBe('crisis.fire.district')
    expect(presentation.cosmetic.intensity).toBe(3)
  })

  it('renders stable merges as a city celebration with a cosmetic flourish', () => {
    const presentation = resolveWorldPresentation(
      event('pr.merged_stable', {
        scope: 'city',
        severity: 'success',
        source: 'pull_request',
        sourceRef: { prId: 'pr-9' }
      }),
      []
    )

    expect(presentation.sceneTag).toBe('celebration.citywide')
    expect(presentation.animationTags).toEqual(expect.arrayContaining(['celebration.citywide', 'fireworks']))
    expect(presentation.npcActivities.some(activity => activity.animationTags.includes('dance'))).toBe(true)
  })

  it('keeps unknown events visible through a generic fallback', () => {
    const presentation = resolveWorldPresentation(event('system.future_event'), [], new Set(['alert']))

    expect(presentation.sceneTag).toBe('alert.unclassified')
    expect(presentation.animationTags).toContain('fallback.generic')
  })

  it('is deterministic for the same event and does not invent dialogue without detail', () => {
    const first = resolveWorldPresentation(event('pr.merge_conflict'), [])
    const second = resolveWorldPresentation(event('pr.merge_conflict'), [])

    expect(stableEventSeed(event('pr.merge_conflict'))).toBe(stableEventSeed(event('pr.merge_conflict')))
    expect(first).toEqual(second)
    expect(first.npcActivities[0].groundedDialogue).toBeUndefined()
  })

  it('gives worker classes stable presentation personalities', () => {
    expect(resolveNpcPersonality('research lead', { agentId: 'researcher' }, event('task.running'))).toBe('curious')
    expect(resolveNpcPersonality('reviewer', { agentId: 'reviewer' }, event('task.in_review'))).toBe('methodical')
    expect(resolveNpcPersonality('unknown', { agentId: 'unknown' }, event('task.running'))).toBe(
      resolveNpcPersonality('unknown', { agentId: 'unknown' }, event('task.running'))
    )
  })
})
