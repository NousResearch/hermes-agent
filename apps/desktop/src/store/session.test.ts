import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  $activeSessionId,
  $attentionSessions,
  $connection,
  $currentCwd,
  $workingSessions,
  applyConfiguredDefaultProjectDir,
  getRecentlySettledSessions,
  mergeSessionPage,
  sessionPinId,
  setCurrentCwd,
  setSessionAttention,
  setSessionWorking,
  workspaceCwdForNewSession
} from './session'
import { makeSessionIdentity } from './session-identity'

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

const identity = (sessionId: string, profile = 'default') => makeSessionIdentity(profile, sessionId)

describe('setSessionAttention', () => {
  it('tracks the same stored id independently across profiles', () => {
    $attentionSessions.set([])

    setSessionAttention('default', 'same-id', true)
    setSessionAttention('default', 'same-id', true)
    setSessionAttention('work', 'same-id', true)
    expect($attentionSessions.get()).toEqual([
      { profile: 'default', sessionId: 'same-id' },
      { profile: 'work', sessionId: 'same-id' }
    ])

    setSessionAttention('default', 'same-id', false)
    expect($attentionSessions.get()).toEqual([{ profile: 'work', sessionId: 'same-id' }])

    $attentionSessions.set([])
  })

  it('ignores empty ids and no-op clears', () => {
    $attentionSessions.set([])

    setSessionAttention('default', null, true)
    setSessionAttention('default', undefined, true)
    setSessionAttention('default', '', true)
    setSessionAttention('default', 'missing', false)
    expect($attentionSessions.get()).toEqual([])
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

    const merged = mergeSessionPage(previous, incoming, [identity('b'), identity('c')])

    expect(merged.map(s => s.id)).toEqual(['c', 'b', 'a'])
    // The finished session comes from the fresh server payload, not the stale
    // optimistic copy.
    expect(merged.find(s => s.id === 'a')?.message_count).toBe(2)
  })

  it('keeps only the matching profile when two rows share a stored id', () => {
    const previous = [session({ id: 'same', profile: 'default' }), session({ id: 'same', profile: 'work' })]

    const merged = mergeSessionPage(previous, [], [{ profile: 'work', sessionId: 'same' }])

    expect(merged).toEqual([session({ id: 'same', profile: 'work' })])
  })

  it('does not duplicate a working session the server already returned', () => {
    const previous = [session({ id: 'b' }), session({ id: 'a' })]
    const incoming = [session({ id: 'b', message_count: 4 }), session({ id: 'a' })]

    const merged = mergeSessionPage(previous, incoming, [identity('b')])

    expect(merged.map(s => s.id)).toEqual(['b', 'a'])
    expect(merged.find(s => s.id === 'b')?.message_count).toBe(4)
  })

  it('never resurrects a session the server dropped that is not in the keep set', () => {
    // A deleted/archived session is removed from `previous` optimistically and
    // is not in the keep set, so it must stay gone after a refresh.
    const previous = [session({ id: 'b' }), session({ id: 'gone' })]
    const incoming = [session({ id: 'b' })]

    expect(mergeSessionPage(previous, incoming, [identity('b')]).map(s => s.id)).toEqual(['b'])
  })

  it('keeps a pinned session that has aged off the recent page', () => {
    // Repro of "loses pins until you refresh": a pinned chat falls off the
    // most-recent page, so the server stops returning it. A hard replace would
    // evict it and the Pinned section would go empty. The keep set (which
    // carries pinned ids) must hold it in memory.
    const previous = [session({ id: 'recent' }), session({ id: 'pinned' })]
    const incoming = [session({ id: 'recent' })]

    const merged = mergeSessionPage(previous, incoming, [identity('pinned')])

    expect(merged.map(s => s.id)).toEqual(['pinned', 'recent'])
  })

  it('keeps a pinned session matched by its lineage root after compression', () => {
    // The pin is stored on the lineage-root id, but the loaded row surfaces
    // under its live compression tip. Matching on _lineage_root_id keeps it.
    const previous = [session({ id: 'tip', _lineage_root_id: 'root' })] as SessionInfo[]
    const incoming = [session({ id: 'other' })] as SessionInfo[]

    const merged = mergeSessionPage(previous, incoming, [identity('root')])

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
    const merged = mergeSessionPage(previous, incoming, [identity('tip-4')])

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

    const merged = mergeSessionPage(previous, incoming, [identity('b')])

    expect(merged.map(s => s.id)).toEqual(['b', 'a-new'])
  })
})

describe('workspaceCwdForNewSession', () => {
  let localStorageDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    const values = new Map<string, string>()
    localStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')

    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        clear: () => values.clear(),
        getItem: (key: string) => values.get(key) ?? null,
        key: (index: number) => [...values.keys()][index] ?? null,
        get length() {
          return values.size
        },
        removeItem: (key: string) => values.delete(key),
        setItem: (key: string, value: string) => values.set(key, value)
      } satisfies Storage
    })
  })

  afterEach(() => {
    applyConfiguredDefaultProjectDir(null)
    $connection.set(null)
    $currentCwd.set('')
    $activeSessionId.set(null)
    window.localStorage.removeItem('hermes.desktop.workspace-cwd')
    window.localStorage.removeItem('hermes.desktop.workspace-cwd.remote.http%3A%2F%2Fbackend-a.default')
    window.localStorage.removeItem('hermes.desktop.workspace-cwd.remote.http%3A%2F%2Fbackend-b.default')

    if (localStorageDescriptor) {
      Object.defineProperty(window, 'localStorage', localStorageDescriptor)
    } else {
      Reflect.deleteProperty(window, 'localStorage')
    }
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

describe('getRecentlySettledSessions', () => {
  afterEach(() => {
    vi.useRealTimers()
    $workingSessions.set([])

    // Drain anything left in the grace map so tests stay isolated.
    for (const id of getRecentlySettledSessions(Number.MAX_SAFE_INTEGER)) {
      void id
    }
  })

  it('keeps a session for the grace window after its turn settles, then drops it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessions.set([])

    // A turn starts then ends: the working→idle transition grants grace.
    setSessionWorking('default', 's1', true)
    setSessionWorking('default', 's1', false)
    expect(getRecentlySettledSessions()).toEqual([{ profile: 'default', sessionId: 's1' }])

    // Still inside the window.
    vi.setSystemTime(29_000)
    expect(getRecentlySettledSessions()).toEqual([{ profile: 'default', sessionId: 's1' }])

    // Past the window: the entry is pruned on read.
    vi.setSystemTime(31_000)
    expect(getRecentlySettledSessions()).toEqual([])
  })

  it('does not grant grace when the session was never working (idle re-asserts)', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessions.set([])

    // updateSessionState re-asserts `false` for idle sessions on every tick;
    // these must not pin an idle chat into the keep-set indefinitely.
    setSessionWorking('default', 'idle', false)
    setSessionWorking('default', 'idle', false)
    expect(getRecentlySettledSessions()).toEqual([])
  })

  it('clears the grace timer when the session goes busy again', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessions.set([])

    setSessionWorking('default', 's2', true)
    setSessionWorking('default', 's2', false)
    expect(getRecentlySettledSessions()).toEqual([{ profile: 'default', sessionId: 's2' }])

    // A new turn for the same session is "working" again — drop it from the
    // settled set so it's tracked as working, not recently-finished.
    setSessionWorking('default', 's2', true)
    expect(getRecentlySettledSessions()).toEqual([])
  })

  it('keeps same-id grace entries separate across profiles', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessions.set([])

    setSessionWorking('default', 'same-id', true)
    setSessionWorking('work', 'same-id', true)
    setSessionWorking('default', 'same-id', false)
    setSessionWorking('work', 'same-id', false)

    expect(getRecentlySettledSessions()).toEqual([
      { profile: 'default', sessionId: 'same-id' },
      { profile: 'work', sessionId: 'same-id' }
    ])
  })
})

describe('setSessionWorking', () => {
  afterEach(() => $workingSessions.set([]))

  it('tracks the same stored id independently across profiles', () => {
    setSessionWorking('default', 'same-id', true)
    setSessionWorking('work', 'same-id', true)

    expect($workingSessions.get()).toEqual([
      { profile: 'default', sessionId: 'same-id' },
      { profile: 'work', sessionId: 'same-id' }
    ])

    setSessionWorking('default', 'same-id', false)
    expect($workingSessions.get()).toEqual([{ profile: 'work', sessionId: 'same-id' }])
  })
})
