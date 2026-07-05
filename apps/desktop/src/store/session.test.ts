import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  $activeSessionId,
  $attentionProfileCounts,
  $attentionSessionIds,
  $connection,
  $cronSessions,
  $currentCwd,
  $messagingSessions,
  $replyReadySessionIds,
  $replyReadySessionProfiles,
  $sessions,
  $workingSessionIds,
  applyConfiguredDefaultProjectDir,
  clearSessionNotifications,
  clearSessionReplyReady,
  getRecentlySettledSessionIds,
  mergeSessionPage,
  sessionPinId,
  setCurrentCwd,
  setSessionAttention,
  setSessionReplyReady,
  setSessionWorking,
  workspaceCwdForNewSession
} from './session'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

describe('setSessionAttention', () => {
  afterEach(() => {
    $attentionSessionIds.set([])
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
  })

  it('adds and removes a session id without duplicating it', () => {
    $attentionSessionIds.set([])

    setSessionAttention('s1', true)
    setSessionAttention('s1', true)
    expect($attentionSessionIds.get()).toEqual(['s1'])

    setSessionAttention('s2', true)
    expect($attentionSessionIds.get()).toEqual(['s1', 's2'])

    setSessionAttention('s1', false)
    expect($attentionSessionIds.get()).toEqual(['s2'])

    $attentionSessionIds.set([])
  })

  it('ignores empty ids and no-op clears', () => {
    $attentionSessionIds.set([])

    setSessionAttention(null, true)
    setSessionAttention(undefined, true)
    setSessionAttention('', true)
    setSessionAttention('missing', false)
    expect($attentionSessionIds.get()).toEqual([])
  })

  it('derives stable waiting counts by profile without exposing full session arrays to the rail', () => {
    $sessions.set([
      session({ id: 's1', profile: 'film-maker', title: 'Waiting one' }),
      session({ id: 's2', profile: 'film-maker', title: 'Waiting two' }),
      session({ id: 's3', profile: 'default', title: 'Idle default' })
    ])
    $cronSessions.set([])
    $messagingSessions.set([])
    $attentionSessionIds.set(['s1'])
    $replyReadySessionIds.set(['s2'])
    $replyReadySessionProfiles.set({})

    const first = $attentionProfileCounts.get()

    expect(first).toEqual({ 'film-maker': 2 })

    $sessions.set([
      session({ id: 's1', profile: 'film-maker', title: 'Waiting one renamed' }),
      session({ id: 's2', profile: 'film-maker', title: 'Waiting two renamed' }),
      session({ id: 's3', profile: 'default', title: 'Idle default renamed' })
    ])

    expect($attentionProfileCounts.get()).toBe(first)
  })

  it('ignores persisted reply-ready profile hints until the session is visible', () => {
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $attentionSessionIds.set([])
    $replyReadySessionIds.set(['fresh-session'])
    $replyReadySessionProfiles.set({ 'fresh-session': 'film-maker' })

    expect($attentionProfileCounts.get()).toEqual({})
  })

  it('counts loaded reply-ready rows under their visible profile', () => {
    $sessions.set([session({ id: 'fresh-session', profile: 'film-maker' })])
    $cronSessions.set([])
    $messagingSessions.set([])
    $attentionSessionIds.set([])
    $replyReadySessionIds.set(['fresh-session'])
    $replyReadySessionProfiles.set({})

    expect($attentionProfileCounts.get()).toEqual({ 'film-maker': 1 })
  })

  it('resolves attention and reply-ready markers through session aliases', () => {
    $sessions.set([session({ id: 'tip-2', _lineage_root_id: 'root-1', profile: 'film-maker' })])
    $cronSessions.set([])
    $messagingSessions.set([])
    $attentionSessionIds.set(['root-1'])
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})

    expect($attentionProfileCounts.get()).toEqual({ 'film-maker': 1 })
  })
})

describe('setSessionReplyReady', () => {
  afterEach(() => {
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $workingSessionIds.set([])
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})
    localStorage.removeItem('hermes.desktop.replyReadySessionIds')
    localStorage.removeItem('hermes.desktop.replyReadySessionProfiles')

    for (const id of getRecentlySettledSessionIds(Number.MAX_SAFE_INTEGER)) {
      void id
    }
  })

  it('tracks finished sessions waiting for the next user reply', () => {
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})

    setSessionReplyReady('s1', true, 'film-maker')
    setSessionReplyReady('s1', true, 'film-maker')
    expect($replyReadySessionIds.get()).toEqual(['s1'])
    expect($replyReadySessionProfiles.get()).toEqual({ s1: 'film-maker' })

    setSessionReplyReady('s2', true, 'default')
    expect($replyReadySessionIds.get()).toEqual(['s1', 's2'])

    setSessionReplyReady('s1', false)
    expect($replyReadySessionIds.get()).toEqual(['s2'])
    expect($replyReadySessionProfiles.get()).toEqual({ s2: 'default' })

    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})
  })

  it('does not persist reply-ready markers across app restarts', () => {
    setSessionReplyReady('s1', true, 'film-maker')

    expect(localStorage.getItem('hermes.desktop.replyReadySessionIds')).toBeNull()
    expect(localStorage.getItem('hermes.desktop.replyReadySessionProfiles')).toBeNull()
  })

  it('does not show a rail badge for a finished turn until its row is visible', () => {
    $sessions.set([])
    $cronSessions.set([])
    $messagingSessions.set([])
    $workingSessionIds.set([])
    $replyReadySessionIds.set([])
    $replyReadySessionProfiles.set({})

    setSessionWorking('bg-session', true, 'film-maker')
    setSessionWorking('bg-session', false)

    expect($replyReadySessionIds.get()).toEqual(['bg-session'])
    expect($attentionProfileCounts.get()).toEqual({})
  })

  it('clears reply-ready markers by live id, stored id, or lineage root', () => {
    $sessions.set([session({ id: 'tip-2', _lineage_root_id: 'root-1', profile: 'film-maker' })])
    $cronSessions.set([])
    $messagingSessions.set([])
    $replyReadySessionIds.set(['root-1'])
    $replyReadySessionProfiles.set({ 'root-1': 'film-maker' })

    clearSessionReplyReady('tip-2')

    expect($replyReadySessionIds.get()).toEqual([])
    expect($replyReadySessionProfiles.get()).toEqual({})
    expect($attentionProfileCounts.get()).toEqual({})
  })

  it('clears all visible notification aliases for a session', () => {
    const current = session({ id: 'tip-2', _lineage_root_id: 'root-1', profile: 'film-maker' })

    $sessions.set([current])
    $cronSessions.set([])
    $messagingSessions.set([])
    $attentionSessionIds.set(['tip-2', 'root-1'])
    $replyReadySessionIds.set(['root-1'])
    $replyReadySessionProfiles.set({ 'root-1': 'film-maker' })

    clearSessionNotifications(current)

    expect($attentionSessionIds.get()).toEqual([])
    expect($replyReadySessionIds.get()).toEqual([])
    expect($replyReadySessionProfiles.get()).toEqual({})
    expect($attentionProfileCounts.get()).toEqual({})
  })
})

describe('sessionPinId', () => {
  it('uses the live id when there is no compression lineage', () => {
    expect(sessionPinId(session({ id: 'abc' }))).toBe('abc')
  })

  it('uses the lineage root so a pin survives compression', () => {
    // After auto-compression the entry surfaces under a fresh tip id but keeps
    // the original root — pinning on the root keeps the pin stable.
    expect(sessionPinId(session({ id: 'tip', _lineage_root_id: 'root' }))).toBe('root')
  })
})

describe('mergeSessionPage', () => {
  it('returns the server page untouched when there is nothing to keep', () => {
    const previous = [session({ id: 'a' }), session({ id: 'b' })]
    const incoming = [session({ id: 'a' })]

    // Content, not identity: the title-carry map rebuilds the array even when
    // nothing is carried, and `incoming` is a fresh server page every fetch.
    expect(mergeSessionPage(previous, incoming, [])).toEqual(incoming)
  })

  it('keeps a still-working session the server omitted', () => {
    // Repro of the disappearing-sessions bug: A finished and is returned by the
    // server, but B and C are mid-first-response (message_count 0 in the DB) so
    // listSessions(min_messages=1) skips them. They must survive the refresh.
    const previous = [session({ id: 'c' }), session({ id: 'b' }), session({ id: 'a' })]
    const incoming = [session({ id: 'a', message_count: 2 })]

    const merged = mergeSessionPage(previous, incoming, ['b', 'c'])

    expect(merged.map(s => s.id)).toEqual(['c', 'b', 'a'])
    // The finished session comes from the fresh server payload, not the stale
    // optimistic copy.
    expect(merged.find(s => s.id === 'a')?.message_count).toBe(2)
  })

  it('does not duplicate a working session the server already returned', () => {
    const previous = [session({ id: 'b' }), session({ id: 'a' })]
    const incoming = [session({ id: 'b', message_count: 4 }), session({ id: 'a' })]

    const merged = mergeSessionPage(previous, incoming, ['b'])

    expect(merged.map(s => s.id)).toEqual(['b', 'a'])
    expect(merged.find(s => s.id === 'b')?.message_count).toBe(4)
  })

  it('never resurrects a session the server dropped that is not in the keep set', () => {
    // A deleted/archived session is removed from `previous` optimistically and
    // is not in the keep set, so it must stay gone after a refresh.
    const previous = [session({ id: 'b' }), session({ id: 'gone' })]
    const incoming = [session({ id: 'b' })]

    expect(mergeSessionPage(previous, incoming, ['b']).map(s => s.id)).toEqual(['b'])
  })

  it('keeps a pinned session that has aged off the recent page', () => {
    // Repro of "loses pins until you refresh": a pinned chat falls off the
    // most-recent page, so the server stops returning it. A hard replace would
    // evict it and the Pinned section would go empty. The keep set (which
    // carries pinned ids) must hold it in memory.
    const previous = [session({ id: 'recent' }), session({ id: 'pinned' })]
    const incoming = [session({ id: 'recent' })]

    const merged = mergeSessionPage(previous, incoming, ['pinned'])

    expect(merged.map(s => s.id)).toEqual(['pinned', 'recent'])
  })

  it('keeps a pinned session matched by its lineage root after compression', () => {
    // The pin is stored on the lineage-root id, but the loaded row surfaces
    // under its live compression tip. Matching on _lineage_root_id keeps it.
    const previous = [session({ id: 'tip', _lineage_root_id: 'root' })] as SessionInfo[]
    const incoming = [session({ id: 'other' })] as SessionInfo[]

    const merged = mergeSessionPage(previous, incoming, ['root'])

    expect(merged.map(s => s.id)).toEqual(['tip', 'other'])
  })

  it('evicts an old compression tip when the incoming page has the new tip from the same lineage', () => {
    // Repro of #43483: after auto-compression rotates the tip (#4 → #5),
    // the sidebar showed both the old tip and the new tip as separate rows.
    // The old tip must be evicted because its lineage key matches the incoming
    // new tip's lineage key.
    const previous = [session({ id: 'tip-4', _lineage_root_id: 'root' }), session({ id: 'other' })] as SessionInfo[]

    const incoming = [session({ id: 'tip-5', _lineage_root_id: 'root' })] as SessionInfo[]

    // 'tip-4' is in the keep set (e.g. it was the active/working session),
    // but should still be evicted because the incoming page carries the same
    // lineage under a new tip id.
    const merged = mergeSessionPage(previous, incoming, ['tip-4'])

    expect(merged.map(s => s.id)).toEqual(['tip-5'])
    // The new tip comes from the server payload.
    expect(merged.find(s => s.id === 'tip-5')?._lineage_root_id).toBe('root')
  })

  it('preserves an unrelated pinned session even when lineage dedup is active', () => {
    // Regression guard: lineage dedup must not accidentally evict sessions
    // from a different lineage that happen to be in the keep set.
    const previous = [
      session({ id: 'a-old', _lineage_root_id: 'lineage-a' }),
      session({ id: 'b', _lineage_root_id: 'lineage-b' })
    ] as SessionInfo[]

    const incoming = [session({ id: 'a-new', _lineage_root_id: 'lineage-a' })] as SessionInfo[]

    const merged = mergeSessionPage(previous, incoming, ['b'])

    expect(merged.map(s => s.id)).toEqual(['b', 'a-new'])
  })
})

describe('workspaceCwdForNewSession', () => {
  afterEach(() => {
    applyConfiguredDefaultProjectDir(null)
    $connection.set(null)
    $currentCwd.set('')
    $activeSessionId.set(null)
    window.localStorage.removeItem('hermes.desktop.workspace-cwd')
    window.localStorage.removeItem('hermes.desktop.workspace-cwd.remote.http%3A%2F%2Fbackend-a.default')
    window.localStorage.removeItem('hermes.desktop.workspace-cwd.remote.http%3A%2F%2Fbackend-b.default')
  })

  it('prefers the configured default over the sticky remembered workspace', () => {
    window.localStorage.setItem('hermes.desktop.workspace-cwd', '/home/user/sticky')
    applyConfiguredDefaultProjectDir('/home/user/configured')

    expect(workspaceCwdForNewSession()).toBe('/home/user/configured')
  })

  it('starts detached (no inherited cwd) when no default project dir is configured', () => {
    // A bare new chat must NOT inherit the sticky/remembered or live workspace —
    // that's the "why is my new session already on a branch" bug. Only an
    // explicit configured default pre-attaches.
    window.localStorage.setItem('hermes.desktop.workspace-cwd', '/home/user/sticky')
    $currentCwd.set('/home/user/live')

    expect(workspaceCwdForNewSession()).toBe('')
  })

  it('does not rewrite the live cwd while a session is active', () => {
    $activeSessionId.set('sess-1')
    $currentCwd.set('/live/session/path')
    applyConfiguredDefaultProjectDir('/home/user/configured')

    expect($currentCwd.get()).toBe('/live/session/path')
    expect(workspaceCwdForNewSession()).toBe('/home/user/configured')
  })

  it('keeps remote workspace memory separate from local and other remotes', () => {
    window.localStorage.setItem('hermes.desktop.workspace-cwd', '/local/project')
    $currentCwd.set('/live/session/path')
    $connection.set({ baseUrl: 'http://backend-a', mode: 'remote' } as never)

    expect(workspaceCwdForNewSession()).toBe('')

    setCurrentCwd('/backend/project-a')
    expect(workspaceCwdForNewSession()).toBe('/backend/project-a')

    $connection.set({ baseUrl: 'http://backend-b', mode: 'remote' } as never)
    expect(workspaceCwdForNewSession()).toBe('')

    setCurrentCwd('/backend/project-b')
    expect(workspaceCwdForNewSession()).toBe('/backend/project-b')

    // Back on local with no configured default: a bare new chat is detached and
    // never reads the remote keys (nor inherits the sticky local workspace).
    $connection.set(null)
    expect(workspaceCwdForNewSession()).toBe('')
  })
})

describe('getRecentlySettledSessionIds', () => {
  afterEach(() => {
    vi.useRealTimers()
    $workingSessionIds.set([])

    // Drain anything left in the grace map so tests stay isolated.
    for (const id of getRecentlySettledSessionIds(Number.MAX_SAFE_INTEGER)) {
      void id
    }
  })

  it('keeps a session for the grace window after its turn settles, then drops it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessionIds.set([])

    // A turn starts then ends: the working→idle transition grants grace.
    setSessionWorking('s1', true)
    setSessionWorking('s1', false)
    expect(getRecentlySettledSessionIds()).toEqual(['s1'])

    // Still inside the window.
    vi.setSystemTime(29_000)
    expect(getRecentlySettledSessionIds()).toEqual(['s1'])

    // Past the window: the entry is pruned on read.
    vi.setSystemTime(31_000)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('does not grant grace when the session was never working (idle re-asserts)', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessionIds.set([])

    // updateSessionState re-asserts `false` for idle sessions on every tick;
    // these must not pin an idle chat into the keep-set indefinitely.
    setSessionWorking('idle', false)
    setSessionWorking('idle', false)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('clears the grace timer when the session goes busy again', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessionIds.set([])

    setSessionWorking('s2', true)
    setSessionWorking('s2', false)
    expect(getRecentlySettledSessionIds()).toEqual(['s2'])

    // A new turn for the same session is "working" again — drop it from the
    // settled set so it's tracked as working, not recently-finished.
    setSessionWorking('s2', true)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })
})
