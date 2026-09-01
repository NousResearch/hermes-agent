import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

const setUnreadRemote = vi.fn<(id: string, unread: boolean, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  // Opening a session now PATCHes its persisted unread flag (clearUnreadOnOpen
  // -> markSessionUnread); keep the REST mutation minimal for the suite.
  setApiRequestProfile: () => {},
  setSessionUnreadRemote: (id: string, unread: boolean, profile?: null | string) => setUnreadRemote(id, unread, profile)
}))

import { deferred } from '../test/deferred'
import { makeSessionInfo } from '../test/session-info'

import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $currentModel,
  $currentProvider,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  _resetLegacyDiscardForTests,
  _resetSessionOwnerHintsForTests,
  applyConfiguredDefaultProjectDir,
  commitWorkspaceCwdForSelectedSession,
  ensureDefaultWorkspaceCwd,
  forgetSessionOwnerHintsForConnection,
  forgetSessionOwnerHintsForSession,
  getConfiguredDefaultProjectDir,
  getRememberedRoute,
  getRememberedSessionId,
  getSessionOwnerHint,
  getSessionOwnerHints,
  hydrateSessionOwnerHints,
  knownSessionOwner,
  knownSessionProfile,
  mergeSessionPage,
  rememberedSessionProfile,
  resolveComposerSessionKey,
  sessionBelongsToProfile,
  sessionOwnerRouteFromRow,
  sessionPinId,
  setComposerSelectionOwner,
  setConnection,
  setCurrentCwd,
  setCurrentCwdTransient,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setRememberedRoute,
  setRememberedSessionId,
  setSelectedStoredSessionId,
  setSessionOwnerHint,
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

const session = (over: Partial<SessionInfo>): SessionInfo => makeSessionInfo({ id: 'live', ...over })

describe('composer model persistence scope', () => {
  const local = { baseUrl: '', connectionId: 'local', mode: 'local' } as never

  beforeEach(() => {
    window.localStorage.clear()
    setConnection(local)
    setCurrentModel('')
    setCurrentProvider('')
    setCurrentModelSource('')
  })

  afterEach(() => {
    setConnection(local)
    window.localStorage.clear()
  })

  it('keeps manual model selections isolated by remote connection and profile', () => {
    const remote = (profile: string) =>
      ({ baseUrl: 'https://aibox.example', connectionId: 'aibox', mode: 'remote', profile }) as never

    setConnection(remote('fred'))
    setCurrentModel('grok-4')
    setCurrentProvider('xai-oauth')
    setCurrentModelSource('manual')

    setConnection(remote('fred-work'))
    expect($currentModel.get()).toBe('')
    expect($currentProvider.get()).toBe('')

    setCurrentModel('local/model')
    setCurrentProvider('custom:local')
    setConnection(remote('fred'))

    expect($currentModel.get()).toBe('grok-4')
    expect($currentProvider.get()).toBe('xai-oauth')
  })

  it('keeps inferred local-primary connections on the historical bare keys', () => {
    setComposerSelectionOwner('remote', 'default')
    window.localStorage.setItem('hermes.desktop.composer.model', 'legacy-model')
    window.localStorage.setItem('hermes.desktop.composer.provider', 'legacy-provider')

    setConnection({ baseUrl: '', connectionId: 'local', mode: 'local', profile: 'default' } as never)

    expect($currentModel.get()).toBe('legacy-model')
    expect($currentProvider.get()).toBe('legacy-provider')
    setCurrentModel('next-model')
    expect(window.localStorage.getItem('hermes.desktop.composer.model')).toBe('next-model')
    expect(window.localStorage.getItem('hermes.desktop.composer.model.registry.local.default')).toBeNull()
  })

  it('uses the live registry owner when the connection descriptor is stale', () => {
    const remote = (profile: string) =>
      ({ baseUrl: 'https://aibox.example', connectionId: 'aibox', mode: 'remote', profile }) as never

    setConnection(remote('fred'))
    setCurrentModel('grok-4')
    setCurrentProvider('xai-oauth')
    setCurrentModelSource('manual')

    // ensureGatewayAgent publishes this coordinate even if getConnectionFor
    // fails and $connection therefore still describes fred.
    setComposerSelectionOwner('aibox', 'fred-work')
    setCurrentModel('local/model')
    setCurrentProvider('custom:local')
    setCurrentModelSource('default')

    setComposerSelectionOwner('aibox', 'fred')
    expect($currentModel.get()).toBe('grok-4')
    expect($currentProvider.get()).toBe('xai-oauth')

    setComposerSelectionOwner('aibox', 'fred-work')
    expect($currentModel.get()).toBe('local/model')
    expect($currentProvider.get()).toBe('custom:local')
  })
})

describe('session owner hints', () => {
  afterEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  it('preserves the registry owner recorded on a discovered session row', () => {
    expect(
      knownSessionOwner(
        [session({ connection_id: 'test-amnezia', id: 'registry-session', profile: 'default' })],
        'registry-session'
      )
    ).toEqual({ connectionId: 'test-amnezia', profile: 'default' })
  })

  it('preserves the exact registry owner for session-scoped RPC routing', () => {
    const route = {
      connectionId: 'test-amnezia',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'default'
    }

    setSessionOwnerHint('remote-session', route)

    expect(knownSessionOwner([session({ id: 'remote-session', profile: 'default' })], 'remote-session')).toEqual(route)
  })

  it('keeps identical session ids separate across connection and profile owners', () => {
    const sourceA = { connectionId: 'source-a', mode: 'remote' as const, profile: 'worker', targetProfile: 'backend-a' }
    const sourceB = { connectionId: 'source-b', mode: 'remote' as const, profile: 'worker', targetProfile: 'backend-b' }

    setSessionOwnerHint('same-session', sourceA)
    setSessionOwnerHint('same-session', sourceB)

    expect(getSessionOwnerHint('same-session', sourceA)).toEqual(sourceA)
    expect(getSessionOwnerHint('same-session', sourceB)).toEqual(sourceB)
    expect(getSessionOwnerHint('same-session')).toBeUndefined()
  })

  it('bounds owner hints and evicts the oldest scoped identity', () => {
    for (let index = 0; index < 257; index += 1) {
      setSessionOwnerHint(`bounded-${index}`, {
        connectionId: 'bounded-source',
        mode: 'remote',
        profile: 'worker',
        targetProfile: 'backend-worker'
      })
    }

    const scope = { connectionId: 'bounded-source', profile: 'worker' }
    expect(getSessionOwnerHint('bounded-0', scope)).toBeUndefined()
    expect(getSessionOwnerHint('bounded-256', scope)).toMatchObject({ connectionId: 'bounded-source' })
  })

  it('survives a relaunch: hints are persisted and rehydrated in LRU order', () => {
    const omar = { connectionId: 'local', mode: 'local' as const, profile: 'omar' }
    const remote = { connectionId: 'homelab', mode: 'remote' as const, profile: 'worker', targetProfile: 'w' }

    setSessionOwnerHint('stored-omar', omar)
    setSessionOwnerHint('stored-remote', remote)

    // "Relaunch": the in-memory map is gone, storage is not.
    _resetSessionOwnerHintsForTests()
    expect(getSessionOwnerHint('stored-omar')).toBeUndefined()

    hydrateSessionOwnerHints()

    expect(getSessionOwnerHint('stored-omar')).toEqual(omar)
    expect(getSessionOwnerHint('stored-remote')).toEqual(remote)

    // LRU order survives: the oldest persisted entry is the first evicted.
    for (let index = 0; index < 255; index += 1) {
      setSessionOwnerHint(`filler-${index}`, { connectionId: 'filler', profile: 'p' })
    }

    expect(getSessionOwnerHint('stored-omar')).toBeUndefined()
    expect(getSessionOwnerHint('stored-remote')).toEqual(remote)
  })

  it('ignores malformed persisted entries and never throws on hydrate', () => {
    window.localStorage.setItem(
      'hermes.desktop.sessionOwnerHints.v1',
      JSON.stringify([
        'junk',
        ['no-route', null],
        ['bad-shape', { connectionId: 7, profile: 'x' }],
        ['good', { connectionId: 'local', profile: 'omar', mode: 'sideways' }]
      ])
    )

    _resetSessionOwnerHintsForTests()
    expect(() => hydrateSessionOwnerHints()).not.toThrow()
    expect(getSessionOwnerHint('good')).toEqual({ connectionId: 'local', profile: 'omar' })
    expect(getSessionOwnerHint('no-route')).toBeUndefined()
    expect(getSessionOwnerHint('bad-shape')).toBeUndefined()
  })

  it('forgets every hint naming a removed connection, in memory and on disk', () => {
    setSessionOwnerHint('stored-a', { connectionId: 'gone', profile: 'omar' })
    setSessionOwnerHint('stored-b', { connectionId: 'gone', profile: 'default' })
    setSessionOwnerHint('stored-c', { connectionId: 'local', profile: 'omar' })

    forgetSessionOwnerHintsForConnection('gone')

    expect(getSessionOwnerHint('stored-a')).toBeUndefined()
    expect(getSessionOwnerHint('stored-b')).toBeUndefined()
    expect(getSessionOwnerHint('stored-c')).toEqual({ connectionId: 'local', profile: 'omar' })

    _resetSessionOwnerHintsForTests()
    hydrateSessionOwnerHints()
    expect(getSessionOwnerHint('stored-a')).toBeUndefined()
    expect(getSessionOwnerHint('stored-c')).toEqual({ connectionId: 'local', profile: 'omar' })
  })

  it('forgets every route for one session without disturbing other sessions', () => {
    setSessionOwnerHint('poisoned', { connectionId: 'local', mode: 'local', profile: 'default' })
    setSessionOwnerHint('poisoned', { connectionId: 'remote-a', mode: 'remote', profile: 'default' })
    setSessionOwnerHint('healthy', { connectionId: 'remote-a', mode: 'remote', profile: 'default' })

    forgetSessionOwnerHintsForSession('poisoned')

    expect(getSessionOwnerHints('poisoned')).toEqual([])
    expect(getSessionOwnerHint('healthy')).toMatchObject({ connectionId: 'remote-a' })

    _resetSessionOwnerHintsForTests()
    hydrateSessionOwnerHints()
    expect(getSessionOwnerHints('poisoned')).toEqual([])
    expect(getSessionOwnerHint('healthy')).toMatchObject({ connectionId: 'remote-a' })
  })

  it('pins only connection-tagged rows and leaves primary SSH rows ambient', () => {
    expect(sessionOwnerRouteFromRow(session({ connection_id: 'source-a', profile: 'worker' }))).toEqual({
      connectionId: 'source-a',
      profile: 'worker',
      targetProfile: 'worker'
    })
    expect(sessionOwnerRouteFromRow(session({ profile: 'default' }))).toBeUndefined()
    expect(sessionOwnerRouteFromRow(session({ connection_id: '  ', profile: 'default' }))).toBeUndefined()
  })
})

describe('knownSessionOwner', () => {
  afterEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  it('returns the EXACT route for a connection-tagged row, the bare profile otherwise', () => {
    const rows = [
      session({ connection_id: 'local', id: 'tagged', profile: 'omar' }),
      session({ id: 'untagged', profile: 'coder' }),
      session({ connection_id: '  ', id: 'blank-tag', profile: 'coder' })
    ]

    expect(knownSessionOwner(rows, 'tagged')).toEqual({ connectionId: 'local', profile: 'omar' })
    expect(knownSessionOwner(rows, 'untagged')).toBe('coder')
    expect(knownSessionOwner(rows, 'blank-tag')).toBe('coder')
    expect(knownSessionOwner(rows, null)).toBeUndefined()
  })

  it('falls through to the EXACT hint route for an unlisted session', () => {
    const route = { connectionId: 'homelab', profile: 'worker', targetProfile: 'w' }

    setSessionOwnerHint('hidden', route)

    expect(knownSessionOwner([], 'hidden')).toEqual(route)
  })
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
  it('carries the owning connection onto a row that comes back untagged (local registry source)', () => {
    // A `local` registry source's rows are served by the primary aggregate as
    // plain local rows: the unified-list splice tags only non-local sources.
    // The refresh must not strip the exact owner the routed create stamped.
    const previous = [session({ connection_id: 'local', id: 'omar-1', last_active: 5, profile: 'omar' })]
    const incoming = [session({ id: 'omar-1', last_active: 5, profile: 'omar' })]

    expect(mergeSessionPage(previous, incoming, [])[0]).toMatchObject({ connection_id: 'local', profile: 'omar' })
  })

  it('drops the carried tag when the refreshed row names a different profile, and never overrides an incoming tag', () => {
    const previous = [
      session({ connection_id: 'local', id: 'moved', profile: 'omar' }),
      session({ connection_id: 'local', id: 'foreign', profile: 'omar' })
    ]

    const incoming = [
      session({ id: 'moved', profile: 'default' }),
      session({ connection_id: 'homelab', id: 'foreign', profile: 'omar' })
    ]

    const merged = mergeSessionPage(previous, incoming, [])
    expect(merged.find(s => s.id === 'moved')?.connection_id).toBeUndefined()
    expect(merged.find(s => s.id === 'foreign')?.connection_id).toBe('homelab')
  })

  it('keeps reference identity when the carried tag is already present', () => {
    const previous = [session({ connection_id: 'local', id: 'same', last_active: 1, profile: 'omar' })]
    const incoming = [session({ connection_id: 'local', id: 'same', last_active: 1, profile: 'omar' })]

    expect(mergeSessionPage(previous, incoming, [])[0]).toBe(incoming[0])
  })

  it('never stitches one profile twin onto another: same id in two profiles stays two sessions (#92454)', () => {
    // Two profiles can hold sessions with the SAME stored id (restored
    // backups, copied state.dbs). Keyed by bare id these collapsed into one
    // row whose title/activity carry crossed profiles — the visible seed of
    // the cross-profile transcript/route mixups.
    const previous = [session({ id: 'twin', last_active: 50, profile: 'testbot', title: 'Testbot work' })]

    const incoming = [
      session({ id: 'twin', last_active: 10, profile: 'quietbot' }),
      session({ id: 'twin', last_active: 50, profile: 'testbot', title: 'Testbot work' })
    ]

    const merged = mergeSessionPage(previous, incoming, [])

    // Both twins survive as distinct rows…
    expect(merged.map(s => `${s.profile}:${s.id}`).sort()).toEqual(['quietbot:twin', 'testbot:twin'])
    // …and testbot's title/activity is NOT stitched onto quietbot's row.
    const quiet = merged.find(s => s.profile === 'quietbot')
    expect(quiet?.title ?? null).toBeNull()
    expect(quiet?.last_active).toBe(10)
  })

  it('a kept twin in another profile survives the incoming page dedupe (#92454)', () => {
    // The incoming page carries only ONE profile's copy; the other profile's
    // twin was in the previous list and pinned via keep. Bare-id dedupe
    // treated the ids as equal and dropped the kept twin.
    const previous = [session({ id: 'twin', profile: 'quietbot' })]
    const incoming = [session({ id: 'twin', message_count: 2, profile: 'testbot' })]

    const merged = mergeSessionPage(previous, incoming, ['twin'])

    expect(merged.map(s => `${s.profile}:${s.id}`).sort()).toEqual(['quietbot:twin', 'testbot:twin'])
  })

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
    delete (window as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('does not publish a delayed configured default after ownership is lost', async () => {
    const settingsResult = deferred<{
      defaultLabel: string
      dir: string
      resolvedCwd: string
    }>()

    const sanitizeWorkspaceCwd = vi.fn(async (cwd: string) => ({ cwd }))

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      sanitizeWorkspaceCwd,
      settings: { getDefaultProjectDir: vi.fn(() => settingsResult.promise) }
    }
    applyConfiguredDefaultProjectDir('/newer/default')
    let ownsSwitch = true

    const seeding = ensureDefaultWorkspaceCwd(() => ownsSwitch)
    ownsSwitch = false
    settingsResult.resolve({ defaultLabel: '/stale', dir: '/stale/default', resolvedCwd: '/stale/default' })
    await seeding

    expect(getConfiguredDefaultProjectDir()).toBe('/newer/default')
    expect($currentCwd.get()).toBe('')
    expect(sanitizeWorkspaceCwd).not.toHaveBeenCalled()
  })

  it('does not publish a delayed sanitized cwd after ownership is lost', async () => {
    const sanitized = deferred<{ cwd: string }>()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = {
      sanitizeWorkspaceCwd: vi.fn(() => sanitized.promise),
      settings: {
        getDefaultProjectDir: vi.fn(async () => ({
          defaultLabel: '/configured',
          dir: '/configured',
          resolvedCwd: '/configured'
        }))
      }
    }
    let ownsSwitch = true

    const seeding = ensureDefaultWorkspaceCwd(() => ownsSwitch)
    await vi.waitFor(() => expect(getConfiguredDefaultProjectDir()).toBe('/configured'))
    ownsSwitch = false
    sanitized.resolve({ cwd: '/stale/sanitized' })
    await seeding

    expect($currentCwd.get()).toBe('')
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
    $sessions.set([])
    setUnreadRemote.mockClear()
  })

  afterEach(() => {
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $sessions.set([])
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

  it('clears a persisted unread row when the session is opened', async () => {
    $sessions.set([session({ id: 's1', unread: true })])

    setSelectedStoredSessionId('s1')

    // The optimistic flip is synchronous; the PATCH is fire-and-forget.
    expect($sessions.get().find(s => s.id === 's1')?.unread).toBe(false)

    await Promise.resolve()
    expect(setUnreadRemote).toHaveBeenCalledWith('s1', false, undefined)
  })

  it('does not PATCH a read row when it is opened', async () => {
    $sessions.set([session({ id: 's1', unread: false })])

    setSelectedStoredSessionId('s1')

    await Promise.resolve()
    expect(setUnreadRemote).not.toHaveBeenCalled()
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

  it('consults the owner hint for a hidden session absent from the list (Bot Chat 4001 class)', () => {
    // Bot Mode's canonical chats are born hidden: no sidebar row ever exists,
    // so without the hint the router would fall back to the ACTIVE profile and
    // land prompt.submit on a backend that never owned the session.
    setSessionOwnerHint('hidden-bot-chat', { connectionId: 'local', mode: 'local', profile: 'developer' })

    expect(rememberedSessionProfile([], 'hidden-bot-chat', 'default')).toBe('developer')
  })

  it('prefers the routed targetProfile over the route profile in the hint fallback', () => {
    setSessionOwnerHint('hidden-remote-chat', {
      connectionId: 'ssh-1',
      profile: 'default',
      targetProfile: 'clippy'
    })

    expect(rememberedSessionProfile([], 'hidden-remote-chat', 'default')).toBe('clippy')
  })

  it('still prefers a real session row over the hint', () => {
    setSessionOwnerHint('rowed-session', { connectionId: 'local', profile: 'wrong-hint' })
    const sessions = [session({ id: 'rowed-session', profile: 'right-owner' })]

    expect(rememberedSessionProfile(sessions, 'rowed-session', 'default')).toBe('right-owner')
  })

  it('normalizes a blank active profile to default', () => {
    expect(rememberedSessionProfile([], null, '')).toBe('default')
    expect(rememberedSessionProfile([], null, null)).toBe('default')
  })
})

describe('knownSessionProfile', () => {
  it('returns the row owner when the session is listed', () => {
    const sessions = [session({ id: 'stored-1', profile: 'ai-engineer' })]

    expect(knownSessionProfile(sessions, 'stored-1')).toBe('ai-engineer')
  })

  it('returns the owner hint for a hidden session absent from the list', () => {
    setSessionOwnerHint('hidden-bot-chat', { connectionId: 'local', mode: 'local', profile: 'developer' })

    expect(knownSessionProfile([], 'hidden-bot-chat')).toBe('developer')
  })

  it('returns undefined for an unknown session — NEVER falls back to active', () => {
    // This is the whole point of the no-active-fallback architecture: an
    // unresolved owner must be undefined so the caller does a cross-profile
    // probe, not silently route the RPC to whatever profile is on screen.
    expect(knownSessionProfile([], 'totally-unknown')).toBeUndefined()
    expect(knownSessionProfile([], null)).toBeUndefined()
  })
})

describe('knownSessionOwner', () => {
  it('preserves a registry connection on a same-named session row', () => {
    expect(
      knownSessionOwner(
        [session({ connection_id: 'source-b', id: 'shared-session', profile: 'default' })],
        'shared-session'
      )
    ).toEqual({ connectionId: 'source-b', profile: 'default' })
  })

  it('preserves a composite owner hint when the row is not listed', () => {
    const owner = { connectionId: 'source-a', profile: 'default', targetProfile: 'backend-default' }
    setSessionOwnerHint('hidden-session', owner)

    expect(knownSessionOwner([], 'hidden-session')).toEqual(owner)
  })
})
