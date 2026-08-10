import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  _resetLegacyDiscardForTests,
  applyConfiguredDefaultProjectDir,
  commitWorkspaceCwdForSelectedSession,
  getRememberedRoute,
  getRememberedSessionId,
  mergeSessionPage,
  rememberedSessionProfile,
  resolveComposerSessionKey,
  sessionBelongsToProfile,
  sessionPinId,
  setCurrentCwd,
  setCurrentCwdTransient,
  setRememberedRoute,
  setRememberedSessionId,
  setSelectedStoredSessionId,
  setSessions,
  shouldMigrateComposerScope,
  touchSessionActivity,
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

  // A chat that hasn't been persisted yet has no stored id, and until it gets
  // one the surfaces key on its runtime id — so publishing under that is what
  // lets a clarify prompt on the very first turn reach the row.
  it('falls back to the runtime id for a session with no storedSessionId', () => {
    publishSessionState('rt1', { ...createClientSessionState(null), needsInput: true })
    expect($attentionSessionIds.get()).toEqual(['rt1'])
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

describe('shouldMigrateComposerScope', () => {
  it('allows tip → lineage-root rekey within the same conversation', () => {
    const sessions = [session({ id: 'tip-a', _lineage_root_id: 'root-a' })]

    expect(shouldMigrateComposerScope('tip-a', 'root-a', sessions)).toBe(true)
  })

  it('blocks cross-session migrate when route flipped but store selection lags', () => {
    // ChatView mid-switch: selectedStoredSessionId still A, route-driven
    // queueSessionKey already B. Migrating would re-home A's queue onto B.
    const sessions = [
      session({ id: 'tip-a', _lineage_root_id: 'root-a' }),
      session({ id: 'tip-b', _lineage_root_id: 'root-b' })
    ]

    expect(shouldMigrateComposerScope('tip-a', 'root-b', sessions)).toBe(false)
    expect(shouldMigrateComposerScope('root-a', 'root-b', sessions)).toBe(false)
    expect(shouldMigrateComposerScope('tip-a', 'tip-b', sessions)).toBe(false)
  })

  it('is a no-op for identical or missing keys', () => {
    const sessions = [session({ id: 'tip-a', _lineage_root_id: 'root-a' })]

    expect(shouldMigrateComposerScope('root-a', 'root-a', sessions)).toBe(false)
    expect(shouldMigrateComposerScope(null, 'root-a', sessions)).toBe(false)
    expect(shouldMigrateComposerScope('root-a', null, sessions)).toBe(false)
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

  it('never regresses last_active behind an optimistic user-send bump', () => {
    const previous = [session({ id: 'old', last_active: 9_000 })]
    const incoming = [session({ id: 'old', last_active: 100, message_count: 4 })]

    const merged = mergeSessionPage(previous, incoming, [])

    expect(merged[0]?.last_active).toBe(9_000)
    expect(merged[0]?.message_count).toBe(4)
  })

  it('carries an optimistic last_active across a compression tip rotation', () => {
    const previous = [session({ id: 'tip-4', _lineage_root_id: 'root', last_active: 9_000 })] as SessionInfo[]
    const incoming = [session({ id: 'tip-5', _lineage_root_id: 'root', last_active: 50 })] as SessionInfo[]

    const merged = mergeSessionPage(previous, incoming, ['tip-4'])

    expect(merged.map(s => s.id)).toEqual(['tip-5'])
    expect(merged[0]?.last_active).toBe(9_000)
  })

  it('sorts survivors by last_active so they interleave with incoming instead of forming a stale block', () => {
    // Repro of #47203: two survivors (B and C) have different last_active
    // timestamps. B settled more recently than C. Without sorting, survivors
    // are prepended in their old order from `previous`, which may be stale.
    // With sorting, B (more recent) should appear before C.
    const previous = [
      session({ id: 'c', last_active: 100 }),
      session({ id: 'b', last_active: 200 }),
      session({ id: 'a', last_active: 300 })
    ]

    // Server returns A (fresh page, order=recent), omits B and C (min_messages=1)
    const incoming = [session({ id: 'a', last_active: 300, message_count: 2 })]

    const merged = mergeSessionPage(previous, incoming, ['b', 'c'])

    // B (last_active 200) should come before C (last_active 100)
    expect(merged.map(s => s.id)).toEqual(['a', 'b', 'c'])
  })

  it('places a very recent survivor in correct position among incoming sessions', () => {
    // A survivor with last_active between two incoming sessions should be
    // interleaved, not prepended as a block.
    const previous = [session({ id: 'survivor', last_active: 150 }), session({ id: 'old', last_active: 50 })]

    const incoming = [session({ id: 'newest', last_active: 200 }), session({ id: 'older', last_active: 100 })]

    const merged = mergeSessionPage(previous, incoming, ['survivor'])

    // survivor (150) should be between newest (200) and older (100)
    expect(merged.map(s => s.id)).toEqual(['newest', 'survivor', 'older'])
  })

  it('keeps a survivor whose optimistic last_active outranks the whole page on top', () => {
    // touchSessionActivity stamps last_active on user-send before the server
    // sees the message; that bump must place the survivor by its FRESH time.
    const previous = [session({ id: 'typing', last_active: 900 }), session({ id: 'settled', last_active: 100 })]

    const incoming = [session({ id: 'settled', last_active: 100, message_count: 3 })]

    const merged = mergeSessionPage(previous, incoming, ['typing'])

    expect(merged.map(s => s.id)).toEqual(['typing', 'settled'])
  })

  it('falls back to started_at for survivors that have no last_active yet', () => {
    // A brand-new session (no persisted message) carries last_active 0; the
    // backend's effective-recency key falls back to started_at, so we must
    // too, or a fresh draft sinks to the very bottom of the sidebar.
    const previous = [
      session({ id: 'draft', last_active: 0, started_at: 500 }),
      session({ id: 'other', last_active: 400 })
    ]

    const incoming = [session({ id: 'other', last_active: 400, message_count: 2 })]

    const merged = mergeSessionPage(previous, incoming, ['draft'])

    expect(merged.map(s => s.id)).toEqual(['draft', 'other'])
  })

  it('interleaves against the title-preserving merged rows, not the raw incoming page', () => {
    // The optimistic last_active carried onto an incoming row must count for
    // its position in the interleave: previous knows 'bumped' was touched at
    // 300 even though the server page still reports 100.
    const previous = [session({ id: 'survivor', last_active: 200 }), session({ id: 'bumped', last_active: 300 })]

    const incoming = [session({ id: 'bumped', last_active: 100, message_count: 2 })]

    const merged = mergeSessionPage(previous, incoming, ['survivor'])

    expect(merged.map(s => s.id)).toEqual(['bumped', 'survivor'])
    expect(merged[0]?.last_active).toBe(300)
  })
})

describe('touchSessionActivity', () => {
  afterEach(() => {
    setSessions([])
  })

  it('bumps last_active for a live id and a lineage-root pin target', () => {
    setSessions([
      session({ id: 'tip', _lineage_root_id: 'root', last_active: 10, preview: 'old' }),
      session({ id: 'other', last_active: 20 })
    ] as SessionInfo[])

    touchSessionActivity('root', { at: 99, preview: 'just sent' })

    const rows = $sessions.get()
    const tip = rows.find(s => s.id === 'tip')
    const other = rows.find(s => s.id === 'other')

    expect(tip?.last_active).toBe(99)
    expect(tip?.preview).toBe('just sent')
    expect(other?.last_active).toBe(20)
  })

  it('is monotonic — a stale stamp does not pull the row down', () => {
    setSessions([session({ id: 'a', last_active: 50 })])

    touchSessionActivity('a', { at: 10 })

    expect($sessions.get()[0]?.last_active).toBe(50)
  })

  it('preserves array identity when nothing matched', () => {
    const prev = [session({ id: 'a', last_active: 1 })]
    setSessions(prev)

    touchSessionActivity('missing', { at: 99 })

    expect($sessions.get()).toBe(prev)
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

  it('keeps the configured default separate from a selected workspace', () => {
    setCurrentCwd('/home/user/repo/.worktrees/feature')

    applyConfiguredDefaultProjectDir('/home/user/configured')

    expect(workspaceCwdForNewSession()).toBe('/home/user/configured')
    expect($currentCwd.get()).toBe('/home/user/repo/.worktrees/feature')
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

  it('remembers only the workspace the user picked, not the one they looked at', () => {
    // The reported bug (#77496 / #80213): on a remote backend a new chat starts
    // in the remembered workspace, and every session resume used to write that
    // key — so opening a project chat silently made it the destination for the
    // next "New session". Following a conversation must leave the memory alone.
    $connection.set({ baseUrl: 'http://backend-a', mode: 'remote' } as never)
    setCurrentCwd('/backend/picked')

    setCurrentCwdTransient('/backend/some-other-project')

    expect($currentCwd.get()).toBe('/backend/some-other-project')
    expect(workspaceCwdForNewSession()).toBe('/backend/picked')
  })

  it('settling a resumed session does not move where the next new chat starts', () => {
    // The reporter's exact sequence: work in a project, open a chat from it,
    // then ask for a new session. Resume settling publishes the conversation's
    // cwd through commitWorkspaceCwdForSelectedSession — which must not claim
    // that folder as the user's chosen workspace.
    $connection.set({ baseUrl: 'http://backend-a', mode: 'remote' } as never)
    setCurrentCwd('/backend/picked')

    setSelectedStoredSessionId('sess-in-project')
    commitWorkspaceCwdForSelectedSession('/backend/last-project')

    expect(workspaceCwdForNewSession()).toBe('/backend/picked')
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

  it('clears the whole conversation family when any row is opened', () => {
    $sessions.set([
      session({ id: 'parent', _lineage_root_id: null }),
      session({ id: 'child', _lineage_root_id: 'parent' }),
      session({ id: 'root', _lineage_root_id: null })
    ])
    $selectedStoredSessionId.set('other')

    // Parent and child both finish in the background.
    for (const storedId of ['parent', 'child']) {
      const working = makeState({ busy: true, storedSessionId: storedId })
      publishSessionState(`rt-${storedId}`, working)
      publishSessionState(`rt-${storedId}`, { ...working, busy: false })
    }

    expect($unreadFinishedSessionIds.get().sort()).toEqual(['child', 'parent'])

    // Opening the CHILD clears the PARENT's dot too (same family).
    setSelectedStoredSessionId('child')
    expect($unreadFinishedSessionIds.get()).toEqual([])

    $sessions.set([])
  })

  it('does NOT re-light a completion that settled before the user read it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000_000)
    $selectedStoredSessionId.set('other')

    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    // User reads the session at t=2s, then the same completion re-asserts at
    // t=3s — the re-assert is the same settled state, so it must not re-light.
    vi.setSystemTime(2_000_000)
    setSelectedStoredSessionId('s1')
    expect($unreadFinishedSessionIds.get()).toEqual([])

    vi.setSystemTime(3_000_000)
    publishSessionState('rt1', { ...working, busy: false })
    expect($unreadFinishedSessionIds.get()).toEqual([])

    vi.useRealTimers()
  })

  it('re-lights when a NEW turn settles after the read baseline', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000_000)
    $selectedStoredSessionId.set('other')

    const working = makeState({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    // User reads s1 at t=2s, then moves on to another session.
    vi.setSystemTime(2_000_000)
    setSelectedStoredSessionId('s1')
    expect($unreadFinishedSessionIds.get()).toEqual([])
    setSelectedStoredSessionId('other')

    // A NEW turn starts (busy again) and finishes at t=4s — genuinely new
    // completion after the read baseline, so it re-lights.
    vi.setSystemTime(3_000_000)
    publishSessionState('rt1', { ...working, busy: true })

    vi.setSystemTime(4_000_000)
    publishSessionState('rt1', { ...working, busy: false })
    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])

    vi.useRealTimers()
  })

  it('openSession marks read before any focus short-circuit', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000_000)
    $selectedStoredSessionId.set('s1')
    $unreadFinishedSessionIds.set(['s1'])

    // A no-op navigate — openSession with 'in-place' against the already
    // selected session hits focusOpenSession and returns without loading.
    const { openSession } = await import('@/app/open-session')
    openSession('s1', () => {}, 'in-place')

    expect($unreadFinishedSessionIds.get()).toEqual([])

    vi.useRealTimers()
  })
})

describe('remembered session id (per profile)', () => {
  beforeEach(() => {
    localStorage.clear()
    _resetLegacyDiscardForTests()
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('scopes the remembered session by profile so one profile cannot read another', () => {
    setRememberedSessionId('work-session', 'ai-engineer')
    setRememberedSessionId('personal-session', 'default')

    expect(getRememberedSessionId('ai-engineer')).toBe('work-session')
    expect(getRememberedSessionId('default')).toBe('personal-session')
    // A profile with nothing remembered does not inherit another's session.
    expect(getRememberedSessionId('research')).toBeNull()
  })

  it('discards legacy unsuffixed keys on first read (zero-migration, refuse-to-guess)', () => {
    // An existing install remembered its session under the pre-per-profile key.
    localStorage.setItem('hermes.desktop.lastSessionId', 'legacy-session')

    // Reading from any profile discards the legacy key — ownership is unknowable.
    expect(getRememberedSessionId('default')).toBeNull()
    expect(getRememberedSessionId('coder')).toBeNull()

    // The legacy key must be cleared.
    expect(localStorage.getItem('hermes.desktop.lastSessionId')).toBeNull()
  })

  it('uses encodeURIComponent so profile names with reserved chars are isolated', () => {
    setRememberedSessionId('ops-session', 'research/ops')

    expect(getRememberedSessionId('research/ops')).toBe('ops-session')
    // Verify the storage key uses encoded form.
    expect(localStorage.getItem('hermes.desktop.lastSessionId.profile.research%2Fops')).toBe('ops-session')
    // Another profile with a different encoding cannot read it.
    expect(getRememberedSessionId('research')).toBeNull()
  })

  it('clearing one profile leaves the others intact', () => {
    setRememberedSessionId('work-session', 'ai-engineer')
    setRememberedSessionId('personal-session', 'default')

    setRememberedSessionId(null, 'ai-engineer')

    expect(getRememberedSessionId('ai-engineer')).toBeNull()
    expect(getRememberedSessionId('default')).toBe('personal-session')
  })
})

describe('remembered route (per profile)', () => {
  beforeEach(() => {
    localStorage.clear()
    _resetLegacyDiscardForTests()
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('scopes the remembered route by profile so one profile cannot restore another', () => {
    // A session route embeds a session id. Remembered globally, a cold start
    // under 'default' would navigate straight into ai-engineer's conversation.
    setRememberedRoute('/session/work-session', 'ai-engineer')
    setRememberedRoute('/session/personal-session', 'default')

    expect(getRememberedRoute('ai-engineer')).toBe('/session/work-session')
    expect(getRememberedRoute('default')).toBe('/session/personal-session')
    expect(getRememberedRoute('research')).toBeNull()
  })

  it('discards legacy unsuffixed keys on first read (zero-migration, refuse-to-guess)', () => {
    localStorage.setItem('hermes.desktop.lastRoute', '/skills')

    // Reading from any profile discards the legacy key.
    expect(getRememberedRoute('default')).toBeNull()
    expect(getRememberedRoute('coder')).toBeNull()

    expect(localStorage.getItem('hermes.desktop.lastRoute')).toBeNull()
  })

  it('uses encodeURIComponent so profile names with reserved chars are isolated', () => {
    setRememberedRoute('/cron', 'research/ops')

    expect(getRememberedRoute('research/ops')).toBe('/cron')
    expect(localStorage.getItem('hermes.desktop.lastRoute.profile.research%2Fops')).toBe('/cron')
    expect(getRememberedRoute('research')).toBeNull()
  })

  it('clearing one profile leaves the others intact', () => {
    setRememberedRoute('/session/work-session', 'ai-engineer')
    setRememberedRoute('/session/personal-session', 'default')

    setRememberedRoute(null, 'ai-engineer')

    expect(getRememberedRoute('ai-engineer')).toBeNull()
    expect(getRememberedRoute('default')).toBe('/session/personal-session')
  })

  it('route and session id agree on the owner, so restore cannot cross profiles', () => {
    // The cold-start restore prefers the route over the id, so the two keys
    // must be written under the same owner or the id scoping is bypassed.
    const owner = rememberedSessionProfile([session({ id: 'stored-1', profile: 'ai-engineer' })], 'stored-1', 'default')

    setRememberedSessionId('stored-1', owner)
    setRememberedRoute('/session/stored-1', owner)

    expect(getRememberedRoute('default')).toBeNull()
    expect(getRememberedSessionId('default')).toBeNull()
    expect(getRememberedRoute('ai-engineer')).toBe('/session/stored-1')
  })
})

describe('sessionBelongsToProfile', () => {
  it('validates that a session row matches a stored id and target profile', () => {
    const sessions = [
      session({ id: 's1', profile: 'ai-engineer' }),
      session({ id: 's2', profile: 'default' }),
      session({ id: 's3', profile: 'ai-engineer' })
    ]

    expect(sessionBelongsToProfile(sessions, 's1', 'ai-engineer')).toBe(true)
    expect(sessionBelongsToProfile(sessions, 's3', 'ai-engineer')).toBe(true)
    expect(sessionBelongsToProfile(sessions, 's2', 'default')).toBe(true)
    // Wrong profile.
    expect(sessionBelongsToProfile(sessions, 's1', 'default')).toBe(false)
    // Missing session.
    expect(sessionBelongsToProfile(sessions, 's-missing', 'ai-engineer')).toBe(false)
  })

  it('matches on lineage root so compressed tips validate their owner', () => {
    const sessions = [session({ id: 'tip-2', _lineage_root_id: 'root-1', profile: 'work' })]

    expect(sessionBelongsToProfile(sessions, 'root-1', 'work')).toBe(true)
    expect(sessionBelongsToProfile(sessions, 'tip-2', 'work')).toBe(true)
    // Wrong profile even when lineage matches.
    expect(sessionBelongsToProfile(sessions, 'root-1', 'personal')).toBe(false)
  })

  it('normalizes blank/empty profiles to default', () => {
    const sessions = [session({ id: 's1', profile: '' }), session({ id: 's2', profile: null as unknown as string })]

    expect(sessionBelongsToProfile(sessions, 's1', 'default')).toBe(true)
    expect(sessionBelongsToProfile(sessions, 's1', '')).toBe(true)
    expect(sessionBelongsToProfile(sessions, 's2', 'default')).toBe(true)
  })

  it('returns false for an empty session list', () => {
    expect(sessionBelongsToProfile([], 'any-id', 'default')).toBe(false)
  })
})

describe('rememberedSessionProfile', () => {
  it('keys by the session row owning profile, not the active one', () => {
    const sessions = [session({ id: 'stored-1', profile: 'ai-engineer' })]

    expect(rememberedSessionProfile(sessions, 'stored-1', 'default')).toBe('ai-engineer')
  })

  it('matches on the lineage root so a compressed tip resolves its owner', () => {
    const sessions = [session({ _lineage_root_id: 'root-1', id: 'tip-2', profile: 'work' })]

    expect(rememberedSessionProfile(sessions, 'root-1', 'default')).toBe('work')
  })

  it('falls back to the active profile for a session not yet in the list', () => {
    expect(rememberedSessionProfile([], 'uncached', 'research')).toBe('research')
  })

  it('normalizes a blank active profile to default', () => {
    expect(rememberedSessionProfile([], null, '')).toBe('default')
    expect(rememberedSessionProfile([], null, null)).toBe('default')
  })
})
