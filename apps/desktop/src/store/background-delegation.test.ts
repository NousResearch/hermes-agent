import { beforeEach, describe, expect, it } from 'vitest'

import { $backgroundResume } from './background-delegation'
import { $activeSessionId, $busy } from './session'
import { $subagentsBySession, type SubagentProgress, type SubagentStreamEntry } from './subagents'

const sub = (over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  id: over.id ?? 'deleg:1',
  parentId: null,
  goal: 'do the thing',
  status: 'running',
  taskCount: 1,
  taskIndex: 0,
  startedAt: 0,
  updatedAt: 0,
  filesRead: [],
  filesWritten: [],
  stream: [],
  ...over
})

const stream = (text: string): SubagentStreamEntry => ({ at: 0, kind: 'progress', text })

describe('$backgroundResume', () => {
  beforeEach(() => {
    $busy.set(false)
    $activeSessionId.set('s1')
    $subagentsBySession.set({})
  })

  it('counts running/queued children for the active session while idle', () => {
    $subagentsBySession.set({ s1: [sub({ id: 'a' }), sub({ id: 'b', status: 'queued' })] })
    expect($backgroundResume.get()?.count).toBe(2)
  })

  it('surfaces the primary child latest stream line as live activity', () => {
    $subagentsBySession.set({ s1: [sub({ id: 'a', stream: [stream('Searching the web…')] })] })
    expect($backgroundResume.get()?.activity).toBe('Searching the web…')
  })

  it('activity is null when no stream line has arrived (UI uses generic copy)', () => {
    $subagentsBySession.set({ s1: [sub({ id: 'a' })] })
    expect($backgroundResume.get()?.activity).toBeNull()
  })

  it('is null while a turn is busy (the turn owns the main loader)', () => {
    $subagentsBySession.set({ s1: [sub({ id: 'a' })] })
    $busy.set(true)
    expect($backgroundResume.get()).toBeNull()
  })

  it('is null when only terminal children or other sessions have work', () => {
    $subagentsBySession.set({
      s1: [sub({ id: 'a', status: 'completed' }), sub({ id: 'b', status: 'failed' })],
      s2: [sub({ id: 'c' })]
    })
    expect($backgroundResume.get()).toBeNull()
  })

  it('is null when there is no active session', () => {
    $subagentsBySession.set({ s1: [sub({ id: 'a' })] })
    $activeSessionId.set(null)
    expect($backgroundResume.get()).toBeNull()
  })
})
