import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $selectedStoredSessionId,
  $unreadFinishedSessionIds,
  applyConfiguredDefaultProjectDir,
  mergeSessionPage,
  resolveComposerSessionKey,
  sessionPinId,
  setCurrentCwd,
  setSelectedStoredSessionId,
  workspaceCwdForNewSession
} from './session'
import {
  $attentionSessionIds,
  clearAllSessionStates,
  getRecentlySettledSessionIds,
  publishSessionState
} from './session-states'

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

describe('computed $attentionSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects sessions with needsInput=true and a storedSessionId', () => {
    publishSessionState('rt1', { ...createClientSessionState('s1'), needsInput: true })
    publishSessionState('rt2', { ...createClientSessionState('s2'), needsInput: false })

    expect($attentionSessionIds.get()).toEqual(['s1'])
  })

  it('updates when needsInput changes', () => {
    publishSessionState('rt1', { ...createClientSessionState('s1'), needsInput: true })
    expect($attentionSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', { ...createClientSessionState('s1'), needsInput: false })
    expect($attentionSessionIds.get()).toEqual([])
  })

  it('ignores sessions without a storedSessionId', () => {
    publishSessionState('rt1', { ...createClientSessionState(null), needsInput: true })
    expect($attentionSessionIds.get()).toEqual([])
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

describe('resolveComposerSessionKey', () => {
  it('keeps the lineage root across compression tip rotation', () => {
    const tipBefore = '20260720_062637_ad96b3'
    const tipAfter = '20260720_071049_a28905'
    const sessions = [session({ id: tipAfter, _lineage_root_id: tipBefore })]

    expect(resolveComposerSessionKey(tipBefore, [session({ id: tipBefore })])).toBe(tipBefore)
    expect(resolveComposerSessionKey(tipAfter, sessions)).toBe(tipBefore)
  })

  it('falls back to the live id when the tip row is not loaded yet', () => {
    expect(resolveComposerSessionKey('tip-new', [])).toBe('tip-new')
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

function makeState(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState('s1'), ...over }
}

describe('getRecentlySettledSessionIds', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    // clearAllSessionStates also drops settle-grace entries + watchdog timers,
    // so nothing leaks in from a previous test.
    clearAllSessionStates()
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
  })

  afterEach(() => {
    vi.useRealTimers()
    clearAllSessionStates()
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
  })

  it('keeps a session for the grace window after its turn settles, then drops it', () => {
    // A turn starts then ends: the working→idle transition grants grace.
    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect(getRecentlySettledSessionIds()).toEqual(['s1'])

    // Still inside the window.
    vi.setSystemTime(29_000)
    expect(getRecentlySettledSessionIds()).toEqual(['s1'])

    // Past the window: the entry is pruned on read.
    vi.setSystemTime(31_000)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('does not grant grace when the session was never working (idle re-asserts)', () => {
    const idle = makeState({ busy: false, storedSessionId: 'idle' })
    publishSessionState('rt1', idle)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('clears the grace timer when the session goes busy again', () => {
    const working = makeState({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect(getRecentlySettledSessionIds()).toEqual(['s2'])

    // A new turn for the same session is "working" again — drop it from the
    // settled set so it's tracked as working, not recently-finished.
    const workingAgain = { ...idle, busy: true }
    publishSessionState('rt1', workingAgain)

    expect(getRecentlySettledSessionIds()).toEqual([])
  })
})

describe('unread finished sessions', () => {
  beforeEach(() => {
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
  })

  afterEach(() => {
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
  })

  it('marks a session unread when its turn finishes in the background', () => {
    $selectedStoredSessionId.set('other-session')

    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])
  })

  it('does NOT mark unread when the finishing session is the active one', () => {
    $selectedStoredSessionId.set('s1')

    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('does NOT mark unread on idle→idle re-asserts (no prior working state)', () => {
    $selectedStoredSessionId.set('other-session')

    const idle = makeState({ busy: false, storedSessionId: 's1' })
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('clears unread when the user opens the session', () => {
    $selectedStoredSessionId.set('other')

    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])

    setSelectedStoredSessionId('s1')
    expect($unreadFinishedSessionIds.get()).toEqual([])
  })
})
