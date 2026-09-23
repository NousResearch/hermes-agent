import { describe, expect, it } from 'vitest'

import { mapObservedState } from './state-map'

describe('mapObservedState', () => {
  it.each([
    ['ready', 'bus', 'queue'],
    ['queued', 'bus', 'queue'],
    ['working', 'project', 'work'],
    ['running', 'project', 'work'],
    ['waiting_for_resource', 'depot', 'wait'],
    ['resource wait', 'depot', 'wait'],
    ['review', 'review', 'review'],
    ['failed', 'triage', 'failed'],
    ['blocked', 'triage', 'blocked'],
    ['heartbeat', 'garden', 'heartbeat'],
    ['idle', 'garden', 'rest'],
    ['recovery', 'garden', 'rest'],
    ['pause', 'garden', 'rest'],
    ['paused', 'garden', 'rest'],
    ['triage', 'triage', 'triage'],
    ['orchestration', 'council', 'handoff'],
    ['dependency', 'council', 'handoff'],
    ['completed', 'project', 'done'],
    ['done', 'project', 'done']
  ])('maps authoritative %s truthfully', (status, destination, animation) => {
    expect(mapObservedState({ source: 'kanban', status, fresh: true })).toEqual({
      animation,
      authority: 'authoritative',
      destination
    })
  })

  it('fails closed for unknown, stale, partial, and disconnected state', () => {
    expect(mapObservedState({ source: 'kanban', status: 'mystery', fresh: true })).toEqual({
      animation: 'unavailable',
      authority: 'unknown',
      destination: 'unknown'
    })
    expect(mapObservedState({ source: 'session', status: 'working', fresh: false })).toEqual({
      animation: 'unavailable',
      authority: 'stale',
      destination: 'unknown'
    })
    expect(mapObservedState({ source: 'subagent', status: 'running', fresh: true, authority: 'partial' })).toEqual({
      animation: 'unavailable',
      authority: 'partial',
      destination: 'unknown'
    })
    expect(mapObservedState({ source: 'session', status: 'working', fresh: true, connected: false })).toEqual({
      animation: 'unavailable',
      authority: 'unknown',
      destination: 'unknown'
    })
  })
})
