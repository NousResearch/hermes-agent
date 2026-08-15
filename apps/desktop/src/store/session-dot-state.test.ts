import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { $sessions } from './session'
import { $delegatingSessionIds, hasLiveTurn, showsRunningArc } from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { $subagentsBySession, type SubagentProgress } from './subagents'

describe('showsRunningArc', () => {
  it('keeps the arc when an authoritative turn goes quiet', () => {
    expect(showsRunningArc('working')).toBe(true)
    expect(showsRunningArc('stalled')).toBe(true)
  })

  it('yields to the needs-input treatment rather than running both', () => {
    expect(showsRunningArc('needs-input')).toBe(false)
  })

  it('leaves a session that is not running unmarked', () => {
    expect(showsRunningArc('background')).toBe(false)
    expect(showsRunningArc('idle')).toBe(false)
    expect(showsRunningArc('unread')).toBe(false)
  })
})

describe('hasLiveTurn', () => {
  it('counts a turn waiting on an answer as still live', () => {
    expect(hasLiveTurn('needs-input')).toBe(true)
  })

  it('covers everything the arc covers', () => {
    for (const state of ['background', 'idle', 'needs-input', 'stalled', 'unread', 'working'] as const) {
      expect(hasLiveTurn(state) || !showsRunningArc(state)).toBe(true)
    }
  })

  it('excludes work that outlived the turn', () => {
    expect(hasLiveTurn('background')).toBe(false)
    expect(hasLiveTurn('unread')).toBe(false)
  })
})

describe('$delegatingSessionIds', () => {
  const subagent = (status: SubagentProgress['status']): SubagentProgress => ({
    id: 'sub-1',
    parentId: null,
    goal: 'do a thing',
    status,
    taskCount: 1,
    taskIndex: 0,
    startedAt: 0,
    updatedAt: 0,
    filesRead: [],
    filesWritten: [],
    stream: []
  })

  afterEach(() => {
    clearAllSessionStates()
    $subagentsBySession.set({})
    $sessions.set([])
  })

  it('claims the stored id while a subagent is running after the parent turn ended', () => {
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: false })
    $subagentsBySession.set({ 'runtime-1': [subagent('running')] })

    expect($delegatingSessionIds.get()).toContain('stored-1')
  })

  it('drops the session once every subagent reaches a terminal status', () => {
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: false })
    $subagentsBySession.set({ 'runtime-1': [subagent('running')] })
    $subagentsBySession.set({ 'runtime-1': [subagent('completed')] })

    expect($delegatingSessionIds.get()).not.toContain('stored-1')
  })

  it('falls back to the runtime id for a not-yet-persisted conversation', () => {
    $subagentsBySession.set({ 'runtime-fresh': [subagent('queued')] })

    expect($delegatingSessionIds.get()).toContain('runtime-fresh')
  })
})
