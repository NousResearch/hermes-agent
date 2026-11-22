import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import type * as HermesModule from '@/hermes'
import type { ProfileInfo, SessionInfo } from '@/types/hermes'

const getSessionMock =
  vi.fn<(id: string, profile?: null | string | { connectionId: string; profile: string }) => Promise<SessionInfo>>()

const getProfilesMock = vi.fn<() => Promise<{ profiles: ProfileInfo[] }>>()

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof HermesModule>()

  return {
    ...actual,
    getProfiles: () => getProfilesMock(),
    getSession: (id: string, profile?: null | string | { connectionId: string; profile: string }) =>
      getSessionMock(id, profile)
  }
})

import { clearSessionDraft, reloadPersistedDrafts, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  simulateComposerQueueReloadForTests
} from '@/store/composer-queue'
import { wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { $pinnedSessionIds } from '@/store/layout'
import { $activeGatewayProfile, $profiles, refreshProfiles } from '@/store/profile'
import { $sessions, _resetSessionOwnerHintsForTests, setConnection, setSessionOwnerHint } from '@/store/session'
import { $sessionTiles, openSessionTile } from '@/store/session-states'

import { __runDeadSessionPrunePass, resetDeadSessionPrune, watchDeadSessionPrune } from './dead-session-prune'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

const profile = (name: string): ProfileInfo =>
  ({
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: name,
    provider: null,
    skill_count: 0
  }) as ProfileInfo

const notFound = () => new Error('404: {"detail":"Session not found"}')

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
})

beforeEach(async () => {
  setApiRequestConnection(null)
  window.localStorage.clear()
  reloadPersistedDrafts()
  setConnection({ connectionId: 'local', mode: 'local', baseUrl: 'http://localhost' } as never)
  $sessions.set([])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
  $profiles.set([profile('default'), profile('work')])
  $activeGatewayProfile.set('default')
  getProfilesMock.mockReset()
  getProfilesMock.mockResolvedValue({ profiles: [profile('default'), profile('work')] })
  await refreshProfiles()
  getSessionMock.mockReset()
  _resetSessionOwnerHintsForTests({ storage: true })
  $sessionTiles.set([])
  resetDeadSessionPrune()
})

afterEach(() => {
  clearSessionDraft('dead-draft-session')
  clearSessionDraft('__new__')
  $sessions.set([])
  $pinnedSessionIds.set([])
  $queuedPromptsBySession.set({})
})

describe('__runDeadSessionPrunePass', () => {
  it('unpins a pin whose session 404s on every profile', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['dead-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
    // Active-scope probe first, then each other profile.
    expect(getSessionMock).toHaveBeenCalledWith('dead-pin', { connectionId: 'local', profile: 'default' })
    expect(getSessionMock).toHaveBeenCalledWith('dead-pin', { connectionId: 'local', profile: 'work' })
  })

  it('keeps a pin whose session lives on another profile, and caches it as alive', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['other-pin'])
    getSessionMock.mockImplementation(async (id, profileName) => {
      if (typeof profileName === 'object' && profileName?.profile === 'work') {
        return row(id, { profile: 'work' })
      }

      throw notFound()
    })

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['other-pin'])

    // Alive verdict is cached — a second pass must not re-probe.
    getSessionMock.mockClear()
    await __runDeadSessionPrunePass()
    expect(getSessionMock).not.toHaveBeenCalled()
  })

  it('defers on non-404 probe failures instead of dropping, and retries later', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['flaky-pin'])
    getSessionMock.mockRejectedValue(new Error('gateway restarting'))

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['flaky-pin'])

    // Not cached — the next pass probes again, and a definitive 404 prunes.
    getSessionMock.mockRejectedValue(notFound())
    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
  })

  it.each([403, 500])('does not treat HTTP %i with a Session not found detail as a 404', async status => {
    $pinnedSessionIds.set(['error-pin'])
    getSessionMock.mockRejectedValue(new Error(`${status}: {"detail":"Session not found"}`))

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual(['error-pin'])
    expect(getSessionMock).toHaveBeenCalledTimes(1)
  })

  it.each(['404: {"detail":"Not Found"}', '404: <html>route missing</html>', '404: Session not found'])(
    'keeps a pin on an inconclusive HTTP 404 (%s)',
    async message => {
      $pinnedSessionIds.set(['uncertain-pin'])
      getSessionMock.mockRejectedValue(new Error(message))

      await __runDeadSessionPrunePass()

      expect($pinnedSessionIds.get()).toEqual(['uncertain-pin'])
      expect(getSessionMock).toHaveBeenCalledTimes(1)
    }
  )

  it('waits for a fresh roster on return to a previously cached but incomplete connection', async () => {
    setApiRequestConnection('source-a')
    setConnection({ connectionId: 'source-a', mode: 'remote', baseUrl: 'https://a.example' } as never)
    getProfilesMock.mockResolvedValueOnce({ profiles: [profile('default')] })
    await refreshProfiles()
    setApiRequestConnection('source-b')
    setConnection({ connectionId: 'source-b', mode: 'remote', baseUrl: 'https://b.example' } as never)
    setApiRequestConnection('source-a')
    setConnection({ connectionId: 'source-a', mode: 'remote', baseUrl: 'https://a.example' } as never)
    expect($profiles.get().map(p => p.name)).toEqual(['default']) // cached paint, not authority
    $pinnedSessionIds.set(['remote-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()
    expect($pinnedSessionIds.get()).toEqual(['remote-pin'])
    expect(getSessionMock).not.toHaveBeenCalled()

    getProfilesMock.mockResolvedValueOnce({ profiles: [profile('default'), profile('work')] })
    await refreshProfiles()
    getSessionMock.mockImplementation(async (id, route) => {
      if (typeof route === 'object' && route?.profile === 'work') {
        return row(id)
      }

      throw notFound()
    })
    await __runDeadSessionPrunePass()
    expect(getSessionMock).toHaveBeenCalledWith('remote-pin', { connectionId: 'source-a', profile: 'work' })
    expect($pinnedSessionIds.get()).toEqual(['remote-pin'])

    getSessionMock.mockRejectedValue(notFound())
    $pinnedSessionIds.set(['truly-dead'])
    await __runDeadSessionPrunePass()
    expect($pinnedSessionIds.get()).toEqual([])
    expect(getSessionMock).toHaveBeenCalledWith('truly-dead', { connectionId: 'source-a', profile: 'default' })
    expect(getSessionMock).toHaveBeenCalledWith('truly-dead', { connectionId: 'source-a', profile: 'work' })
  })

  it('preserves typed drafts even after a same-backend 404, including new-chat text', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['dead-draft-session'])
    setSessionOwnerHint('dead-draft-session', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('dead-draft-session', 'stale text', [])
    stashSessionDraft('__new__', 'brand new', [])
    reloadPersistedDrafts()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).toHaveBeenCalledWith('dead-draft-session', { connectionId: 'local', profile: 'default' })
    expect($pinnedSessionIds.get()).toEqual([])
    expect(takeSessionDraft('dead-draft-session').text).toBe('stale text')
    expect(takeSessionDraft('__new__').text).toBe('brand new')
    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')['dead-draft-session']).toBe(
      'stale text'
    )
  })

  it('keeps a remote tile draft when a stale local hint contradicts its authoritative route', async () => {
    setSessionOwnerHint('tile-draft', { connectionId: 'local', profile: 'default' })
    openSessionTile('tile-draft', 'right', undefined, undefined, {
      workspaceMode: 'sessions',
      ownerRoute: { connectionId: 'remote-a', profile: 'default' }
    })
    stashSessionDraft('tile-draft', 'remote unsent text', [])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).not.toHaveBeenCalled()
    expect(takeSessionDraft('tile-draft').text).toBe('remote unsent text')
    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')['tile-draft']).toBe(
      'remote unsent text'
    )
  })

  it('preserves a same-backend queued prompt and draft on 404 while pruning its pin', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['dead-queue-session'])
    setSessionOwnerHint('dead-queue-session', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('dead-queue-session', { text: 'go', attachments: [] })
    stashSessionDraft('dead-queue-session', 'unsent', [])
    const queueBefore = window.localStorage.getItem('hermes.desktop.composerQueue.v1')
    const draftBefore = window.localStorage.getItem('hermes:composer-drafts:v3')
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
    expect(getQueuedPrompts('dead-queue-session').map(entry => entry.text)).toEqual(['go'])
    expect(window.localStorage.getItem('hermes.desktop.composerQueue.v1')).toBe(queueBefore)
    expect(window.localStorage.getItem('hermes:composer-drafts:v3')).toBe(draftBefore)
  })

  it('does not probe or prune a persisted queue without a pin', async () => {
    setSessionOwnerHint('persisted-dead', { connectionId: 'local', profile: 'work' })
    enqueueQueuedPrompt('persisted-dead', { text: 'stale pending', attachments: [] })
    simulateComposerQueueReloadForTests()
    const persisted = window.localStorage.getItem('hermes.desktop.composerQueue.v1')
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getQueuedPrompts('persisted-dead').map(entry => entry.text)).toEqual(['stale pending'])
    expect(window.localStorage.getItem('hermes.desktop.composerQueue.v1')).toBe(persisted)
    expect(getSessionMock).not.toHaveBeenCalled()
  })

  it('preserves a legacy persisted queue even if a later same-id owner hint appears', async () => {
    window.localStorage.setItem(
      'hermes.desktop.composerQueue.v1',
      JSON.stringify({
        'legacy-session': [{ id: 'legacy-entry', text: 'do not lose me', attachments: [], queuedAt: 1 }]
      })
    )
    simulateComposerQueueReloadForTests()
    setSessionOwnerHint('legacy-session', { connectionId: 'local', profile: 'default' })
    $pinnedSessionIds.set(['legacy-session'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getQueuedPrompts('legacy-session').map(entry => entry.text)).toEqual(['do not lose me'])
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('keeps foreign-gateway queue entries under a same-id pin on the selected connection', async () => {
    setSessionOwnerHint('collision', { connectionId: 'remote-b', profile: 'work' })
    enqueueQueuedPrompt('collision', { text: 'remote work', attachments: [] })
    $pinnedSessionIds.set(['collision'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).toHaveBeenCalledWith('collision', { connectionId: 'local', profile: 'default' })
    expect($pinnedSessionIds.get()).toEqual([])
    expect(getQueuedPrompts('collision').map(entry => entry.text)).toEqual(['remote work'])
  })

  it('preserves a legacy draft when a later hint cannot establish its original owner', async () => {
    // Saved before any owner hint existed, through the production write path (the
    // v4 stash is the authority; a raw v3 list is only adopted at module load).
    stashSessionDraft('legacy-draft', 'keep unsent', [])
    reloadPersistedDrafts()
    setSessionOwnerHint('legacy-draft', { connectionId: 'local', profile: 'default' })
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).not.toHaveBeenCalled()
    expect(takeSessionDraft('legacy-draft').text).toBe('keep unsent')
  })

  it('keeps a replacement draft written by another window during its dead probe', async () => {
    setSessionOwnerHint('race-draft', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('race-draft', 'original', [])
    $pinnedSessionIds.set(['race-draft'])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ 'race-draft': 'replacement' }))
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect(JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')['race-draft']).toBe(
      'replacement'
    )
  })

  it('preserves both a dead draft and an unrelated cross-window write before its storage event', async () => {
    setSessionOwnerHint('dead-a', { connectionId: 'local', profile: 'default' })
    setSessionOwnerHint('other-b', { connectionId: 'local', profile: 'default' })
    stashSessionDraft('dead-a', 'delete me', [])
    stashSessionDraft('other-b', 'old B', [])
    $pinnedSessionIds.set(['dead-a'])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )
    getSessionMock.mockRejectedValue(notFound())

    const pass = __runDeadSessionPrunePass()
    const texts = JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ ...texts, 'other-b': 'NEW B' }))
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    const persisted = JSON.parse(window.localStorage.getItem('hermes:composer-drafts:v3') ?? '{}')
    expect(getSessionMock).toHaveBeenCalledWith('dead-a', { connectionId: 'local', profile: 'default' })
    expect(persisted['dead-a']).toBe('delete me')
    expect(persisted['other-b']).toBe('NEW B')
  })

  it('skips ids already covered by loaded sidebar rows', async () => {
    $sessions.set([row('covered', { _lineage_root_id: 'root-covered' })])
    $pinnedSessionIds.set(['covered', 'root-covered', 'rowless-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect(getSessionMock).not.toHaveBeenCalledWith('covered', expect.anything())
    expect(getSessionMock).not.toHaveBeenCalledWith('root-covered', expect.anything())
    expect(getSessionMock).toHaveBeenCalledWith('rowless-pin', { connectionId: 'local', profile: 'default' })
    // Covered pins are alive (their rows exist) — only the rowless pin dies.
    expect($pinnedSessionIds.get()).toEqual(['covered', 'root-covered'])
  })

  it('defers when the profile list is not loaded yet', async () => {
    $sessions.set([row('unrelated')])
    $profiles.set([])
    $pinnedSessionIds.set(['mystery-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    // Can't rule out another profile we don't know about — keep the pin.
    expect($pinnedSessionIds.get()).toEqual(['mystery-pin'])
  })

  it('prunes on a genuinely empty backend (all sessions purged)', async () => {
    // An empty list is a real answer here — the wipe path is handled by the
    // gateway-switch reset cancelling the wipe-scheduled sweep, not by the
    // sweep refusing to run. A backend with zero sessions has no rows to
    // cover stored ids, so probes decide.
    $sessions.set([])
    $pinnedSessionIds.set(['orphan-pin'])
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
    expect(getSessionMock).toHaveBeenCalledWith('orphan-pin', { connectionId: 'local', profile: 'default' })
  })

  it('re-probes an alive id once its verdict TTL expires', async () => {
    vi.useFakeTimers()

    try {
      $sessions.set([row('unrelated')])
      $pinnedSessionIds.set(['ttl-pin'])
      getSessionMock.mockImplementation(async (id, profileName) => {
        if (typeof profileName === 'object' && profileName?.profile === 'work') {
          return row(id, { profile: 'work' })
        }

        throw notFound()
      })

      // Alive on another profile — cached, not pruned, not re-probed.
      await __runDeadSessionPrunePass()
      expect($pinnedSessionIds.get()).toEqual(['ttl-pin'])
      getSessionMock.mockClear()
      await __runDeadSessionPrunePass()
      expect(getSessionMock).not.toHaveBeenCalled()

      // Session deleted mid-session; after the TTL the next pass re-probes
      // and a definitive all-profile 404 prunes.
      vi.setSystemTime(Date.now() + 10 * 60_000 + 1_000)
      getSessionMock.mockRejectedValue(notFound())
      await __runDeadSessionPrunePass()

      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it('discards in-flight verdicts when a gateway reset happens mid-sweep', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['race-pin'])
    let resolveProbe!: (value: SessionInfo) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveProbe = resolve
        })
    )

    const pass = __runDeadSessionPrunePass()
    // Gateway switch while the probe is in flight.
    resetDeadSessionPrune()
    resolveProbe(row('race-pin', { profile: 'default' }))
    await pass

    // The pre-reset verdict must not be applied: the pin survives, and the
    // discarded 'alive' verdict must not poison the cache (next pass re-probes).
    expect($pinnedSessionIds.get()).toEqual(['race-pin'])
    getSessionMock.mockClear()
    getSessionMock.mockRejectedValue(notFound())
    $sessions.set([row('unrelated')])

    await __runDeadSessionPrunePass()

    expect(getSessionMock).toHaveBeenCalledWith('race-pin', { connectionId: 'local', profile: 'default' })
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('keeps cross-window queue writes during a pin probe without rewriting storage', async () => {
    setSessionOwnerHint('race-queue', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('race-queue', { text: 'original', attachments: [] })
    $pinnedSessionIds.set(['race-queue'])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()

    const fromOtherWindow = JSON.stringify({
      'race-queue': [
        {
          id: 'new-from-other-window',
          text: 'keep me',
          attachments: [],
          queuedAt: 2,
          ownerRoute: { connectionId: 'local', profile: 'default' }
        }
      ],
      'other-session': [{ id: 'other-entry', text: 'also keep', attachments: [], queuedAt: 2 }]
    })

    window.localStorage.setItem('hermes.desktop.composerQueue.v1', fromOtherWindow)
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect($pinnedSessionIds.get()).toEqual([])
    expect(window.localStorage.getItem('hermes.desktop.composerQueue.v1')).toBe(fromOtherWindow)
    simulateComposerQueueReloadForTests()
    expect(getQueuedPrompts('race-queue').map(entry => entry.text)).toEqual(['keep me'])
  })

  it('discards a stale 404 when a connection descriptor changes without a prune reset', async () => {
    setSessionOwnerHint('route-race', { connectionId: 'local', profile: 'default' })
    enqueueQueuedPrompt('route-race', { text: 'pending', attachments: [] })
    $pinnedSessionIds.set(['route-race'])
    let rejectProbe!: (reason: Error) => void
    getSessionMock.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const pass = __runDeadSessionPrunePass()
    setConnection({ connectionId: 'remote-race', mode: 'remote', baseUrl: 'https://race.example' } as never)
    rejectProbe(notFound())
    getSessionMock.mockRejectedValue(notFound())
    await pass

    expect(getSessionMock).toHaveBeenCalledWith('route-race', { connectionId: 'local', profile: 'default' })
    expect(getQueuedPrompts('route-race').map(entry => entry.text)).toEqual(['pending'])
  })

  it('resetDeadSessionPrune drops the alive cache so a switched gateway is re-probed', async () => {
    $sessions.set([row('unrelated')])
    $pinnedSessionIds.set(['switched-pin'])
    getSessionMock.mockImplementation(async id => row(id, { profile: 'default' }))

    await __runDeadSessionPrunePass()
    expect($pinnedSessionIds.get()).toEqual(['switched-pin'])

    // The gateway switched; the new backend no longer has the session. Without
    // the reset the alive verdict would keep the dead pin forever.
    resetDeadSessionPrune()
    getSessionMock.mockRejectedValue(notFound())

    await __runDeadSessionPrunePass()

    expect($pinnedSessionIds.get()).toEqual([])
  })
})

describe('watchDeadSessionPrune', () => {
  it('cancels a scheduled prune when disposed, without touching pins, queues or drafts', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['disposed-pin'])
      enqueueQueuedPrompt('disposed-pin', { text: 'pending', attachments: [] })
      stashSessionDraft('disposed-pin', 'unsent', [])
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('unrelated')])
      expect(vi.getTimerCount()).toBe(1)

      unsubscribe()
      expect(vi.getTimerCount()).toBe(0)
      await vi.advanceTimersByTimeAsync(2_000)

      expect(getSessionMock).not.toHaveBeenCalled()
      expect($pinnedSessionIds.get()).toEqual(['disposed-pin'])
      expect(getQueuedPrompts('disposed-pin').map(entry => entry.text)).toEqual(['pending'])
      expect(takeSessionDraft('disposed-pin').text).toBe('unsent')
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('discards an in-flight dead verdict when disposed, and a new watcher can start fresh', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['in-flight-pin'])
      let rejectProbe!: (reason: Error) => void
      getSessionMock.mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectProbe = reject
          })
      )
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('unrelated')])
      await vi.advanceTimersByTimeAsync(2_000)
      expect(getSessionMock).toHaveBeenCalledTimes(1)

      unsubscribe()
      rejectProbe(notFound())
      await vi.waitFor(() => expect(getSessionMock).toHaveBeenCalledTimes(2))
      expect($pinnedSessionIds.get()).toEqual(['in-flight-pin'])

      const nextUnsubscribe = watchDeadSessionPrune()

      try {
        $sessions.set([row('fresh-payload')])
        expect(vi.getTimerCount()).toBe(1)
        unsubscribe() // A repeated disposal must not cancel the new watcher's pass.
        await vi.advanceTimersByTimeAsync(2_000)
        expect($pinnedSessionIds.get()).toEqual([])
      } finally {
        nextUnsubscribe()
      }
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('retries the pin pass when a fresh roster arrives after the first sweep without a new sidebar list', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      setApiRequestConnection('source-a')
      setConnection({ connectionId: 'source-a', mode: 'remote', baseUrl: 'https://a.example' } as never)
      const stableRows = [row('unrelated')]
      $sessions.set(stableRows)
      $pinnedSessionIds.set(['late-pin'])
      getSessionMock.mockRejectedValue(notFound())

      await vi.advanceTimersByTimeAsync(2_000)
      expect(getSessionMock).not.toHaveBeenCalled()
      expect($pinnedSessionIds.get()).toEqual(['late-pin'])

      await refreshProfiles()
      expect($sessions.get()).toBe(stableRows)
      expect(vi.getTimerCount()).toBe(1)
      await vi.advanceTimersByTimeAsync(5_000)
      expect($pinnedSessionIds.get()).toEqual([])
      expect(getSessionMock).toHaveBeenCalledWith('late-pin', { connectionId: 'source-a', profile: 'work' })
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('does not retry from cached paint or an HTTP roster fetched through a foreign route', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      setApiRequestConnection('source-b')
      setConnection({ connectionId: 'source-a', mode: 'remote', baseUrl: 'https://a.example' } as never)
      $sessions.set([row('unrelated')])
      $pinnedSessionIds.set(['foreign-roster-pin'])
      getSessionMock.mockRejectedValue(notFound())
      await vi.advanceTimersByTimeAsync(2_000)

      // A successful HTTP response reached the picker, but its routed source
      // was different from the selected connection. It is not prune authority.
      await refreshProfiles()
      expect(vi.getTimerCount()).toBe(0)
      await vi.advanceTimersByTimeAsync(5_000)
      expect(getSessionMock).not.toHaveBeenCalled()
      expect($pinnedSessionIds.get()).toEqual(['foreign-roster-pin'])
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('does not schedule a pin pass from a roster stranded by a profile switch during refresh', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      setApiRequestConnection('source-a')
      setConnection({ connectionId: 'source-a', mode: 'remote', baseUrl: 'https://a.example' } as never)
      $sessions.set([row('unrelated')])
      $pinnedSessionIds.set(['switched-profile-pin'])
      getSessionMock.mockRejectedValue(notFound())
      await vi.advanceTimersByTimeAsync(2_000)

      let finishRefresh!: (value: { profiles: ProfileInfo[] }) => void
      getProfilesMock.mockImplementationOnce(
        () =>
          new Promise(resolve => {
            finishRefresh = resolve
          })
      )
      const oldRefresh = refreshProfiles()
      $activeGatewayProfile.set('work')
      finishRefresh({ profiles: [profile('default'), profile('work')] })
      await oldRefresh
      expect(vi.getTimerCount()).toBe(0)
      await vi.advanceTimersByTimeAsync(5_000)
      expect(getSessionMock).not.toHaveBeenCalled()
      expect($pinnedSessionIds.get()).toEqual(['switched-profile-pin'])
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('keeps the original gateway queue and draft after a switch and real scheduled prune', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $sessions.set([row('local-session')])
      setSessionOwnerHint('local-session', { connectionId: 'local', profile: 'work' })
      enqueueQueuedPrompt('local-session', { text: 'unsent follow-up', attachments: [] })
      stashSessionDraft('local-session', 'unsent draft', [])
      $pinnedSessionIds.set(['local-session'])

      wipeSessionListsForGatewaySwitch()
      setConnection({ connectionId: 'remote-a', mode: 'remote', baseUrl: 'https://remote.example' } as never)
      $profiles.set([profile('default'), profile('work')])
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('remote-session')])

      await vi.advanceTimersByTimeAsync(2_000)

      expect(getQueuedPrompts('local-session').map(entry => entry.text)).toEqual(['unsent follow-up'])
      expect(takeSessionDraft('local-session').text).toBe('unsent draft')
      expect(getSessionMock).not.toHaveBeenCalledWith('local-session', expect.anything())
      setConnection({ connectionId: 'local', mode: 'local', baseUrl: 'http://localhost' } as never)
      expect($pinnedSessionIds.get()).toEqual(['local-session'])
      expect(getQueuedPrompts('local-session').map(entry => entry.text)).toEqual(['unsent follow-up'])
    } finally {
      unsubscribe()
      resetDeadSessionPrune()
      vi.useRealTimers()
    }
  })

  it('sweeps once after the first list payload', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['timed-pin'])
      getSessionMock.mockRejectedValue(notFound())

      $sessions.set([row('boot-row')])

      vi.advanceTimersByTime(2_000)
      await vi.runOnlyPendingTimersAsync()

      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })

  it('clamps the debounce so list churn cannot starve the sweep', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      $pinnedSessionIds.set(['churn-pin'])
      getSessionMock.mockRejectedValue(notFound())

      // First payload schedules the pass at +2s; a change 1s later would
      // naively re-schedule to +6s. The clamp keeps the earlier deadline —
      // and must not stack a second timer for it.
      $sessions.set([row('a')])
      vi.advanceTimersByTime(1_000)
      $sessions.set([row('a'), row('b')])
      expect(vi.getTimerCount()).toBe(1)

      vi.advanceTimersByTime(1_000)
      await vi.runOnlyPendingTimersAsync()

      expect($pinnedSessionIds.get()).toEqual([])
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })

  it('re-opens a fresh debounce window after a sweep fires', async () => {
    vi.useFakeTimers()
    const unsubscribe = watchDeadSessionPrune()

    try {
      // First sweep fires at +2s and prunes the boot dead pin.
      $pinnedSessionIds.set(['first-pin'])
      getSessionMock.mockRejectedValue(notFound())
      $sessions.set([row('a')])
      vi.advanceTimersByTime(2_000)
      await vi.runOnlyPendingTimersAsync()
      expect($pinnedSessionIds.get()).toEqual([])

      // A change right after the sweep must NOT fire an immediate pass — the
      // next pass waits out a fresh RERUN_DELAY window.
      $pinnedSessionIds.set(['second-pin'])
      $sessions.set([row('a'), row('c')])

      vi.advanceTimersByTime(1_000)
      expect($pinnedSessionIds.get()).toEqual(['second-pin'])

      vi.advanceTimersByTime(4_000)
      await vi.runOnlyPendingTimersAsync()
      expect($pinnedSessionIds.get()).toEqual([])
    } finally {
      unsubscribe()
      vi.useRealTimers()
    }
  })
})
