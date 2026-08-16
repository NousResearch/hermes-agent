import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { $sessions, $unreadFinishedSessionIds, setSessions } from './session'
import { $delegatingSessionIds, $sessionDotStateById, hasLiveTurn, showsRunningArc } from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { $unreadWriteGuard } from './session-unread-remote'
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

const storedRow = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

describe('persisted unread (backend watermark)', () => {
  beforeEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $unreadFinishedSessionIds.set([])
    $unreadWriteGuard.set(new Map())
  })

  afterEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $unreadFinishedSessionIds.set([])
    $unreadWriteGuard.set(new Map())
  })

  it('claims unread for a session whose row carries unread: true', () => {
    setSessions([storedRow('s1', { unread: true })])

    expect($sessionDotStateById.get()['s1']).toBe('unread')
  })

  it('keeps draft weaker than persisted unread', () => {
    // A blank tile (no busy, no messages, message_count 0) is a draft; the
    // persisted unread claim must speak over it.
    setSessions([storedRow('s1', { message_count: 0, unread: true })])
    publishSessionState('rt1', createClientSessionState('s1'))

    expect($sessionDotStateById.get()['s1']).toBe('unread')
  })

  it('lets working outrank persisted unread', () => {
    setSessions([storedRow('s1', { unread: true })])
    publishSessionState('rt1', { ...createClientSessionState('s1'), busy: true })

    expect($sessionDotStateById.get()['s1']).toBe('working')
  })

  it('fences a stale page with the write guard', () => {
    const guard = new Map<string, { at: number; value: boolean }>()
    guard.set('s1', { at: Date.now(), value: true })
    $unreadWriteGuard.set(guard)

    // A page issued before our PATCH still says read — keep OUR value.
    setSessions([storedRow('s1', { unread: false })])
    expect($sessionDotStateById.get()['s1']).toBe('unread')

    // The guard expires: the page wins and the dot drops.
    const expired = new Map<string, { at: number; value: boolean }>()
    expired.set('s1', { at: Date.now() - 60_000, value: true })
    $unreadWriteGuard.set(expired)
    expect($sessionDotStateById.get()['s1']).not.toBe('unread')
  })

  it('leaves a row alone when the backend omits the flag (older runtime)', () => {
    setSessions([storedRow('s1')])

    expect($sessionDotStateById.get()['s1'] ?? 'idle').not.toBe('unread')
  })
})
