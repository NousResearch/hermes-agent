import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { $backgroundStatusBySession } from './composer-status'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unreadFinishedSessionIds,
  markAllSessionsRead,
  setCronSessions,
  setMessagingSessions,
  setSessions
} from './session'
import {
  $activeSubagentCountBySessionId,
  $delegatingSessionIds,
  $sessionDotStateById,
  $sessionLiveTurnStateById,
  $unreadSessionCount,
  hasLiveTurn,
  showsRunningArc,
  unreadSessionCount
} from './session-dot-state'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { $unreadWriteGuard } from './session-unread-remote'
import { $subagentsBySession, type SubagentProgress, upsertSubagent } from './subagents'

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
  const subagent = (status: SubagentProgress['status'], id = 'sub-1'): SubagentProgress => ({
    id,
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
    $backgroundStatusBySession.set({})
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

  it('keeps a violet subagent claim while any queued or running child remains', () => {
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: false })
    upsertSubagent('runtime-1', { goal: 'first', status: 'running', subagent_id: 'a', task_index: 0 })
    upsertSubagent('runtime-1', { goal: 'second', status: 'queued', subagent_id: 'b', task_index: 1 })

    expect($activeSubagentCountBySessionId.get()['stored-1']).toBe(2)
    expect($sessionDotStateById.get()['stored-1']).toBe('subagents')

    upsertSubagent('runtime-1', { status: 'completed', subagent_id: 'a', task_index: 0 }, false, 'subagent.complete')
    expect($activeSubagentCountBySessionId.get()['stored-1']).toBe(1)
    expect($sessionDotStateById.get()['stored-1']).toBe('subagents')

    upsertSubagent('runtime-1', { status: 'error', subagent_id: 'b', task_index: 1 }, false, 'subagent.complete')
    expect($activeSubagentCountBySessionId.get()['stored-1']).toBeUndefined()
    expect($sessionDotStateById.get()['stored-1']).not.toBe('subagents')
  })

  it('publishes the active count under every stored lineage alias', () => {
    setSessions([
      storedRow('tip', {
        _lineage_ids: ['root', 'middle', 'tip'],
        _lineage_root_id: 'root'
      })
    ])
    publishSessionState('runtime-1', { ...createClientSessionState('tip'), busy: false })
    $subagentsBySession.set({ 'runtime-1': [subagent('running', 'child')] })

    expect($activeSubagentCountBySessionId.get()).toMatchObject({ root: 1, middle: 1, tip: 1 })
    expect($sessionDotStateById.get()).toMatchObject({ root: 'subagents', middle: 'subagents', tip: 'subagents' })
  })

  it('lets needs-input outrank subagents without hiding a concurrent parent turn', () => {
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: true })
    $subagentsBySession.set({ 'runtime-1': [subagent('running')] })

    expect($sessionDotStateById.get()['stored-1']).toBe('subagents')
    expect($sessionLiveTurnStateById.get()['stored-1']).toBe('working')
    expect(showsRunningArc($sessionLiveTurnStateById.get()['stored-1'])).toBe(true)

    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: true, needsInput: true })

    expect($sessionDotStateById.get()['stored-1']).toBe('needs-input')
    expect($sessionLiveTurnStateById.get()['stored-1']).toBe('needs-input')
    expect(showsRunningArc($sessionLiveTurnStateById.get()['stored-1'])).toBe(false)
  })

  it('restores the background terminal state after the last child finishes', () => {
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: false })
    $backgroundStatusBySession.set({
      'runtime-1': [{ id: 'job-1', state: 'running', title: 'sleep', type: 'background' }]
    })

    expect($sessionDotStateById.get()['stored-1']).toBe('background')

    upsertSubagent('runtime-1', { goal: 'child', status: 'running', subagent_id: 'a', task_index: 0 })
    expect($sessionDotStateById.get()['stored-1']).toBe('subagents')

    upsertSubagent('runtime-1', { status: 'completed', subagent_id: 'a', task_index: 0 }, false, 'subagent.complete')
    expect($sessionDotStateById.get()['stored-1']).toBe('background')
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

describe('unreadSessionCount', () => {
  it('counts listed unread rows and skips archived', () => {
    expect(
      unreadSessionCount({ a: 'unread', b: 'working', c: 'unread' }, [
        { id: 'a' },
        { id: 'b' },
        { archived: true, id: 'c' },
        { id: 'missing' }
      ])
    ).toBe(1)
  })

  it('does not count alias keys that are not listed rows', () => {
    expect(unreadSessionCount({ tip: 'unread', root: 'unread' }, [{ id: 'tip' }])).toBe(1)
  })
})

describe('$unreadSessionCount (titlebar badge)', () => {
  beforeEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $unreadFinishedSessionIds.set([])
    $unreadWriteGuard.set(new Map())
  })

  afterEach(() => {
    clearAllSessionStates()
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $unreadFinishedSessionIds.set([])
    $unreadWriteGuard.set(new Map())
  })

  it('does not count an unread cron session — cron runs finish unwatched by design (#93552)', () => {
    setCronSessions([storedRow('cron-1', { source: 'cron' })])
    $unreadFinishedSessionIds.set(['cron-1'])

    // The cron row itself still paints unread in the sidebar cron section...
    expect($sessionDotStateById.get()['cron-1']).toBe('unread')
    // ...but the titlebar badge stays quiet.
    expect($unreadSessionCount.get()).toBe(0)
  })

  it('counts an unread regular session', () => {
    setSessions([storedRow('reg-1', { unread: true })])

    expect($unreadSessionCount.get()).toBe(1)
  })

  it('counts regular + messaging but never cron', () => {
    setSessions([storedRow('reg-1', { unread: true })])
    setMessagingSessions([storedRow('msg-1')])
    setCronSessions([storedRow('cron-1', { source: 'cron' })])
    $unreadFinishedSessionIds.set(['msg-1', 'cron-1'])

    expect($unreadSessionCount.get()).toBe(2)
  })

  it('mark-all-read clears regular and cron unread alike', () => {
    setSessions([storedRow('reg-1')])
    setCronSessions([storedRow('cron-1', { source: 'cron' })])
    $unreadFinishedSessionIds.set(['reg-1', 'cron-1'])

    expect($unreadSessionCount.get()).toBe(1)
    expect($sessionDotStateById.get()['cron-1']).toBe('unread')

    markAllSessionsRead()

    expect($unreadSessionCount.get()).toBe(0)
    expect($sessionDotStateById.get()['cron-1']).not.toBe('unread')
  })
})
