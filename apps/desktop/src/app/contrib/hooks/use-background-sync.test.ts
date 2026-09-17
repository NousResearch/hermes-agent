import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { sessionMessagesSignature } from '@/lib/session-signatures'
import { $changeEventsAvailable, notifySessionsChanged, resetLiveSync } from '@/store/live-sync'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  _resetSessionOwnerHintsForTests,
  setBusy,
  setCronSessions,
  setMessagingSessions,
  setSessionOwnerHint,
  setSessions
} from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import {
  $attentionSessionIds,
  $openStoredSessionIds,
  $sessionStates,
  $sessionTiles,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  closeSessionTile,
  discardSessionTile,
  openSessionTile,
  patchSessionTile,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS
} from '@/store/session-states'
import {
  $todoProgressBySession,
  $todosBySession,
  _todoHydrationAuthorityStatsForTests,
  applyTodoContinuationSnapshot,
  clearAllSessionTodos,
  clearAllTodoContinuations,
  setSessionTodos
} from '@/store/todos'
import { deferred } from '@/test/deferred'
import { LEGACY_TODOS_277757, legacyEvidenceMessage277757 } from '@/test/evidence-row-277757'

import {
  type ActiveTranscriptRefreshDeps,
  isTypingBurstActive,
  noteRendererKeyboardActivity,
  profileScopeForTranscriptSession,
  reconcileActiveTranscript,
  reconcileTileTranscripts as reconcileTileTranscriptsForTest,
  rehydrateLiveSessionStatuses,
  resetLiveRuntimeTracking,
  resetTypingActivityTracking,
  resolveActiveTranscriptSession,
  useBackgroundSync,
  windowIsActivelyViewed
} from './use-background-sync'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn(),
  getSessionMessages: vi.fn()
}))

vi.mock('@/store/projects', async importOriginal => ({
  ...(await importOriginal()),
  refreshProjectTree: vi.fn(async () => undefined)
}))

const { getLatestSessionMessages, getSessionMessages } = await import('@/hermes')
const { refreshProjectTree } = await import('@/store/projects')

const ACTIVE_RUNTIME_ID = 'runtime-active'
const ACTIVE_STORED_ID = 'stored-active'
const carrierTodos = [
  { content: 'parent', id: 'plan', status: 'completed' as const },
  { content: 'child', id: 'child', parent: 'plan', status: 'in_progress' as const },
  { content: 'next', id: 'next', status: 'pending' as const }
]

function carrierTranscript(sessionId = ACTIVE_STORED_ID) {
  return {
    messages: [
      {
        content: 'synthetic carrier text',
        display_kind: 'hidden',
        display_metadata: { todo_snapshot: { todos: carrierTodos } },
        role: 'user',
        timestamp: 3
      }
    ],
    session_id: sessionId
  }
}

function legacyEvidenceTranscript(sessionId = ACTIVE_STORED_ID) {
  return {
    messages: [{ ...legacyEvidenceMessage277757, session_id: sessionId, timestamp: 3 }],
    session_id: sessionId
  }
}

function transcript(
  answer: string,
  sessionId = ACTIVE_STORED_ID
): Awaited<ReturnType<typeof getLatestSessionMessages>> {
  return {
    messages: [
      { content: 'question', role: 'user', timestamp: 1 },
      { content: answer, role: 'assistant', timestamp: 2 }
    ],
    session_id: sessionId
  }
}

type TileStateUpdater = Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState']

function bindRealTile(storedSessionId: string, runtimeId: string, ownerRoute?: SessionProfileRoute): void {
  openSessionTile(storedSessionId)
  patchSessionTile(storedSessionId, { ownerRoute, runtimeId })
}

function makePublishingTileUpdater(): ReturnType<typeof vi.fn<TileStateUpdater>> {
  return vi.fn<TileStateUpdater>((sessionId, updater, storedSessionId) => {
    const previous = $sessionStates.get()[sessionId] ?? createClientSessionState(storedSessionId ?? sessionId)
    const next = updater(previous)

    publishSessionState(sessionId, next)

    return next
  })
}

function reconcileRealTiles({
  requestSequenceRef = { current: 0 },
  signatureRef = { current: new Map<string, string>() },
  updateSessionState = makePublishingTileUpdater()
}: {
  requestSequenceRef?: { current: number }
  signatureRef?: { current: Map<string, string> }
  updateSessionState?: TileStateUpdater
} = {}) {
  const request = reconcileTileTranscriptsForTest({
    requestSequenceRef,
    signatureRef,
    updateSessionState
  })

  return { request, requestSequenceRef, signatureRef, updateSessionState }
}

function makeRefresh(resolveSession: ActiveTranscriptRefreshDeps['resolveSession'] = () => ({ profile: 'default' })) {
  const activeSessionIdRef = { current: ACTIVE_RUNTIME_ID as string | null }
  const selectedStoredSessionIdRef = { current: ACTIVE_STORED_ID as string | null }
  const busyRef = { current: false }
  const requestSequenceRef = { current: 0 }
  const signatureRef = { current: new Map<string, string>() }
  const state = createClientSessionState(ACTIVE_STORED_ID)
  const states = new Map([[ACTIVE_RUNTIME_ID, state]])

  const updateSessionStateRef = {
    updateSessionState: vi.fn((sessionId: string, updater: (value: typeof state) => typeof state) => {
      const next = updater(states.get(sessionId) ?? createClientSessionState(ACTIVE_STORED_ID))
      states.set(sessionId, next)

      return next
    })
  }

  const { updateSessionState } = updateSessionStateRef

  const refresh = () =>
    reconcileActiveTranscript({
      activeSessionIdRef,
      busyRef,
      requestSequenceRef,
      resolveSession,
      selectedStoredSessionIdRef,
      signatureRef,
      updateSessionState
    })

  return {
    activeSessionIdRef,
    busyRef,
    refresh,
    selectedStoredSessionIdRef,
    signatureRef,
    state,
    states,
    updateSessionState
  }
}

function useSyncHarness({
  activeIsMessaging = false,
  activeSessionId,
  activeStoredSessionId,
  gatewayState = 'open',
  refreshActiveTranscript
}: {
  activeIsMessaging?: boolean
  activeSessionId: string | null
  activeStoredSessionId: string | null
  gatewayState?: string
  refreshActiveTranscript: () => Promise<void>
}) {
  const updateSessionState: Parameters<typeof useBackgroundSync>[0]['updateSessionState'] = vi.fn(
    (sessionId, updater) => {
      const current = {} as Parameters<typeof updater>[0]

      return updater(current)
    }
  )

  useBackgroundSync({
    activeConnectionId: 'local',
    activeGatewayProfile: 'default',
    activeIsMessaging,
    activeSessionId,
    activeStoredSessionId,
    freshDraftReady: false,
    gatewayState,
    refreshActiveTranscript,
    refreshCronJobs: vi.fn(),
    refreshCurrentModel: vi.fn(),
    refreshHermesConfig: vi.fn(),
    refreshMessagingSessions: vi.fn(),
    refreshSessions: vi.fn(),
    updateSessionState,
    requestGateway: vi.fn(async () => ({ sessions: [] })) as never
  })
}

type SyncOptions = {
  activeIsMessaging?: boolean
  activeSessionId?: null | string
  activeStoredSessionId?: null | string
  gatewayState?: string
}

function renderSync(refreshActiveTranscript: () => Promise<void>, options: SyncOptions = {}) {
  return renderHook(
    (props: SyncOptions) =>
      useSyncHarness({
        activeSessionId: ACTIVE_RUNTIME_ID,
        activeStoredSessionId: ACTIVE_STORED_ID,
        refreshActiveTranscript,
        ...props
      }),
    { initialProps: options }
  )
}

beforeEach(() => {
  // visiblePoll only ticks while the window is actively viewed; jsdom's
  // document.hasFocus() is not reliably true, so pin it for these tests.
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
  vi.mocked(getSessionMessages).mockResolvedValue({
    messages: [],
    pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 0 },
    session_id: ACTIVE_STORED_ID
  } as never)
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  resetLiveSync()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  setSessions([])
  setCronSessions([])
  setMessagingSessions([])
  setBusy(false)
  vi.clearAllMocks()
  vi.restoreAllMocks()
  clearAllSessionStates()
  for (const tile of [...$sessionTiles.get()]) {
    discardSessionTile(tile.storedSessionId)
  }

  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests()
  clearAllSessionTodos()
  clearAllTodoContinuations()
  resetLiveRuntimeTracking()
  resetTypingActivityTracking()
})

describe('resolveActiveTranscriptSession', () => {
  it('uses a unique hidden owner hint for hydration and refresh scope', async () => {
    const ownerRoute = {
      connectionId: 'hidden-remote',
      profile: 'connection-profile',
      targetProfile: 'bot-profile',
      mode: 'remote' as const
    }

    setSessionOwnerHint(ACTIVE_STORED_ID, ownerRoute)
    const scope = profileScopeForTranscriptSession(resolveActiveTranscriptSession(ACTIVE_STORED_ID, ACTIVE_RUNTIME_ID))

    expect(scope).toEqual({ connectionId: ownerRoute.connectionId, profile: ownerRoute.targetProfile })
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('hint-owned answer'))
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, scope)
    expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'hint-owned answer'
    })
  })

  it('does not promote a different runtime tile into the active transcript owner', () => {
    const ownerRoute = { connectionId: 'local', profile: 'other-profile', mode: 'local' as const }
    // SAFETY: Ownership resolution reads only identity and profile fields; omitted session metadata is unused.
    setSessions([{ id: 'shared', profile: 'default', source: 'desktop' } as never])
    $sessionTiles.set([{ storedSessionId: 'shared', runtimeId: 'other-runtime', ownerRoute }])

    expect(resolveActiveTranscriptSession('shared', 'active-runtime')).toEqual({ profile: 'default' })
  })

  it('does not treat an ownerless active tile as corroboration for a stale hint', () => {
    // SAFETY: Ownership resolution reads only identity and profile fields; omitted session metadata is unused.
    setSessions([{ id: 'shared', profile: 'default', source: 'desktop' } as never])
    setSessionOwnerHint('shared', { connectionId: 'stale-connection', profile: 'stale-profile', mode: 'remote' })
    $sessionTiles.set([{ storedSessionId: 'shared', runtimeId: 'active-runtime' }])

    expect(resolveActiveTranscriptSession('shared', 'active-runtime')).toEqual({ profile: 'default' })
  })
})

describe('tile transcript owner routing', () => {
  const storedSessionId = 'stored-owner-route-tile'
  const runtimeSessionId = 'runtime-owner-route-tile'

  it('reads a secondary local profile through the tile owner route', async () => {
    const ownerRoute = {
      connectionId: 'local',
      mode: 'local' as const,
      profile: 'secondary-local'
    }

    bindRealTile(storedSessionId, runtimeSessionId, ownerRoute)
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(transcript('local answer', storedSessionId) as never)
    const { request, updateSessionState } = reconcileRealTiles()

    await request

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(getLatestSessionMessages).toHaveBeenCalledWith(
      storedSessionId,
      {
        connectionId: ownerRoute.connectionId,
        profile: ownerRoute.profile
      },
      { passive: true }
    )
    expect(updateSessionState).toHaveBeenCalledTimes(1)
  })

  it('reads a remote target profile through the exact tile owner route', async () => {
    const ownerRoute = {
      connectionId: 'ssh-owner',
      mode: 'remote' as const,
      profile: 'desktop-bot-route',
      targetProfile: 'remote-worker'
    }

    bindRealTile(storedSessionId, runtimeSessionId, ownerRoute)
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(transcript('remote answer', storedSessionId) as never)
    const { request, updateSessionState } = reconcileRealTiles()

    await request

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(getLatestSessionMessages).toHaveBeenCalledWith(
      storedSessionId,
      {
        connectionId: ownerRoute.connectionId,
        profile: ownerRoute.targetProfile
      },
      { passive: true }
    )
    expect(updateSessionState).toHaveBeenCalledTimes(1)
  })

  it('keeps a rebound owner signature after the old owner read retires', async () => {
    const staleRead = deferred<ReturnType<typeof transcript>>()

    const ownerA = {
      connectionId: 'ssh-owner-a',
      mode: 'remote' as const,
      profile: 'desktop-owner-a',
      targetProfile: 'worker-a'
    }

    const ownerB = {
      connectionId: 'ssh-owner-b',
      mode: 'remote' as const,
      profile: 'desktop-owner-b',
      targetProfile: 'worker-b'
    }

    const requestSequenceRef = { current: 0 }
    const signatureRef = { current: new Map<string, string>() }
    const updateSessionState = makePublishingTileUpdater()
    const identical = transcript('same bytes on both owners', storedSessionId)

    bindRealTile(storedSessionId, runtimeSessionId, ownerA)
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(staleRead.promise as never)
    const staleRequest = reconcileRealTiles({ requestSequenceRef, signatureRef, updateSessionState }).request

    patchSessionTile(storedSessionId, { ownerRoute: ownerB })
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(identical as never)
    await reconcileRealTiles({ requestSequenceRef, signatureRef, updateSessionState }).request
    expect(updateSessionState).toHaveBeenCalledTimes(1)

    staleRead.resolve(identical)
    await staleRequest

    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(identical as never)
    await reconcileRealTiles({ requestSequenceRef, signatureRef, updateSessionState }).request

    expect(updateSessionState).toHaveBeenCalledTimes(1)
    expect(signatureRef.current.size).toBe(1)
    expect(vi.mocked(getLatestSessionMessages).mock.calls).toEqual([
      [storedSessionId, { connectionId: ownerA.connectionId, profile: ownerA.targetProfile }, { passive: true }],
      [storedSessionId, { connectionId: ownerB.connectionId, profile: ownerB.targetProfile }, { passive: true }],
      [storedSessionId, { connectionId: ownerB.connectionId, profile: ownerB.targetProfile }, { passive: true }]
    ])
  })
})

describe('tile reconciliation lifecycle authority', () => {
  const storedSessionId = 'stored-lifecycle-tile'
  const runtimeSessionId = 'runtime-lifecycle-tile'

  it('rejects a pending read after the tile closes without recreating Todo or session state', async () => {
    const read = deferred<ReturnType<typeof carrierTranscript>>()

    bindRealTile(storedSessionId, runtimeSessionId)
    publishSessionState(runtimeSessionId, createClientSessionState(storedSessionId))
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(read.promise as never)
    const { request, signatureRef, updateSessionState } = reconcileRealTiles()

    closeSessionTile(storedSessionId)
    expect($sessionStates.get()[runtimeSessionId]).toBeUndefined()

    read.resolve(carrierTranscript(storedSessionId))
    await request

    expect(updateSessionState).not.toHaveBeenCalled()
    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
    expect($sessionStates.get()[runtimeSessionId]).toBeUndefined()
    expect($openStoredSessionIds.get().has(storedSessionId)).toBe(false)
    expect(signatureRef.current.has(`tile:${storedSessionId}`)).toBe(false)
  })

  it('treats close then reopen of the same identifiers as a new incarnation and permits its later retry', async () => {
    const staleRead = deferred<ReturnType<typeof transcript>>()

    bindRealTile(storedSessionId, runtimeSessionId)
    publishSessionState(runtimeSessionId, createClientSessionState(storedSessionId))
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(staleRead.promise as never)
    const requestSequenceRef = { current: 0 }
    const signatureRef = { current: new Map<string, string>() }
    const updateSessionState = makePublishingTileUpdater()
    const staleRequest = reconcileRealTiles({ requestSequenceRef, signatureRef, updateSessionState }).request

    closeSessionTile(storedSessionId)
    bindRealTile(storedSessionId, runtimeSessionId)
    staleRead.resolve(transcript('pre-close response', storedSessionId))
    await staleRequest

    expect(updateSessionState).not.toHaveBeenCalled()
    expect($sessionStates.get()[runtimeSessionId]).toBeUndefined()
    expect(signatureRef.current.has(`tile:${storedSessionId}`)).toBe(false)

    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(
      transcript('new incarnation response', storedSessionId) as never
    )
    await reconcileRealTiles({ requestSequenceRef, signatureRef, updateSessionState }).request

    expect(updateSessionState).toHaveBeenCalledTimes(1)
    expect($sessionStates.get()[runtimeSessionId]?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'new incarnation response'
    })
  })

  it('rejects an old-runtime read after the stored tile rebinds', async () => {
    const read = deferred<ReturnType<typeof carrierTranscript>>()
    const reboundRuntimeId = 'runtime-lifecycle-rebound'

    bindRealTile(storedSessionId, runtimeSessionId)
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(read.promise as never)
    const { request, signatureRef, updateSessionState } = reconcileRealTiles()

    patchSessionTile(storedSessionId, { runtimeId: reboundRuntimeId })
    read.resolve(carrierTranscript(storedSessionId))
    await request

    expect(updateSessionState).not.toHaveBeenCalled()
    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
    expect($todosBySession.get()[reboundRuntimeId]).toBeUndefined()
    expect($sessionStates.get()[runtimeSessionId]).toBeUndefined()
    expect(signatureRef.current.has(`tile:${storedSessionId}`)).toBe(false)
    expect(_todoHydrationAuthorityStatsForTests()).toMatchObject({
      activeSessionCount: 0,
      activeTokenCount: 0,
      latestSessionCount: 0
    })
  })

  it.each(['unmount', 'profile', 'connection'] as const)(
    'retires an in-flight tile read when its background-sync owner changes by %s',
    async retirement => {
      const read = deferred<ReturnType<typeof carrierTranscript>>()

      bindRealTile(storedSessionId, runtimeSessionId)
      $changeEventsAvailable.set(true)
      vi.mocked(getLatestSessionMessages).mockReturnValueOnce(read.promise as never)
      const updateSessionState = makePublishingTileUpdater()

      const stable = {
        activeIsMessaging: false,
        activeSessionId: null,
        activeStoredSessionId: null,
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript: vi.fn(async () => undefined),
        refreshCronJobs: vi.fn(async () => undefined),
        refreshCurrentModel: vi.fn(async () => undefined),
        refreshHermesConfig: vi.fn(async () => undefined),
        refreshMessagingSessions: vi.fn(async () => undefined),
        refreshSessions: vi.fn(async () => undefined),
        requestGateway: vi.fn(async () => ({ sessions: [] })) as never,
        updateSessionState
      }

      const hook = renderHook(
        ({ activeConnectionId, activeGatewayProfile }) =>
          useBackgroundSync({ activeConnectionId, activeGatewayProfile, ...stable }),
        { initialProps: { activeConnectionId: 'connection-a' as string | null, activeGatewayProfile: 'profile-a' } }
      )

      act(() => notifySessionsChanged())
      expect(getLatestSessionMessages).toHaveBeenCalledWith(storedSessionId, undefined, { passive: true })

      if (retirement === 'unmount') {
        hook.unmount()
      } else {
        hook.rerender({
          activeConnectionId: retirement === 'connection' ? 'connection-b' : 'connection-a',
          activeGatewayProfile: retirement === 'profile' ? 'profile-b' : 'profile-a'
        })
      }

      await act(async () => {
        read.resolve(carrierTranscript(storedSessionId))
        await read.promise
        await Promise.resolve()
      })

      expect(updateSessionState).not.toHaveBeenCalled()
      expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
      expect($sessionStates.get()[runtimeSessionId]).toBeUndefined()
    }
  )

  it('hydrates an unchanged still-open tile once', async () => {
    bindRealTile(storedSessionId, runtimeSessionId)
    vi.mocked(getLatestSessionMessages).mockResolvedValueOnce(carrierTranscript(storedSessionId) as never)
    const { request, updateSessionState } = reconcileRealTiles()

    await request

    expect(updateSessionState).toHaveBeenCalledTimes(1)
    expect($todosBySession.get()[runtimeSessionId]).toEqual(carrierTodos)
    expect($sessionStates.get()[runtimeSessionId]).toBeDefined()
  })

  it('rejects a late read when the tile becomes busy', async () => {
    const read = deferred<ReturnType<typeof carrierTranscript>>()

    bindRealTile(storedSessionId, runtimeSessionId)
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(read.promise as never)
    const { request, signatureRef, updateSessionState } = reconcileRealTiles()

    publishSessionState(runtimeSessionId, {
      ...createClientSessionState(storedSessionId),
      busy: true,
      turnLive: true
    })
    read.resolve(carrierTranscript(storedSessionId))
    await request

    expect(updateSessionState).not.toHaveBeenCalled()
    expect($todosBySession.get()[runtimeSessionId]).toBeUndefined()
    expect($sessionStates.get()[runtimeSessionId]).toMatchObject({ busy: true, turnLive: true })
    expect(signatureRef.current.has(`tile:${storedSessionId}`)).toBe(false)
  })
})

describe('active transcript refresh', () => {
  beforeEach(() => {
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('answer') as never)
  })

  it('refreshes a hidden session through its unique complete owner route', async () => {
    const hiddenStoredSessionId = 'hidden-bot-chat'

    const ownerRoute = {
      connectionId: 'ssh-bot-owner',
      mode: 'remote' as const,
      profile: 'bot-route',
      targetProfile: 'bot-profile'
    }

    $changeEventsAvailable.set(true)
    $activeSessionId.set(ACTIVE_RUNTIME_ID)
    $selectedStoredSessionId.set(hiddenStoredSessionId)
    setSessionOwnerHint(hiddenStoredSessionId, ownerRoute)
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    fixture.selectedStoredSessionIdRef.current = hiddenStoredSessionId
    vi.mocked(getLatestSessionMessages).mockResolvedValue(
      transcript('hidden external answer', hiddenStoredSessionId) as never
    )

    renderSync(fixture.refresh, { activeStoredSessionId: hiddenStoredSessionId })

    act(() => notifySessionsChanged())

    await waitFor(() =>
      expect(getLatestSessionMessages).toHaveBeenCalledWith(hiddenStoredSessionId, {
        connectionId: ownerRoute.connectionId,
        profile: ownerRoute.targetProfile
      })
    )
    expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'hidden external answer'
    })
  })

  it('reconciles a workspace TILE transcript when sessions.changed ticks (#94255 review: behavior, not source-grep)', async () => {
    $changeEventsAvailable.set(true)
    // The tile's runtime differs from the active session — it is NOT the main
    // pane surface, so only the tile reconcile path may update it.
    const TILE_RUNTIME_ID = 'runtime-tile'
    const TILE_STORED_ID = 'stored-tile'
    $activeSessionId.set('runtime-something-else')
    $selectedStoredSessionId.set('stored-other')

    const states = new Map<string, ReturnType<typeof createClientSessionState>>()
    states.set(TILE_RUNTIME_ID, createClientSessionState(TILE_STORED_ID))

    let updaterCallCount = 0

    const updateSessionState: Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState'] = vi.fn(
      (sessionId, updater) => {
        updaterCallCount += 1
        const current = {} as Parameters<typeof updater>[0]

        return updater(current)
      }
    )

    void updateSessionState

    const signatureRef = { current: new Map<string, string>() }
    const requestSequenceRef = { current: 0 }

    vi.mocked(getLatestSessionMessages).mockImplementation(async (storedId: string) => {
      if (storedId === TILE_STORED_ID) {
        return {
          messages: [
            { content: 'tile question', role: 'user', timestamp: 1 },
            { content: 'background delivery answer', role: 'assistant', timestamp: 2 }
          ],
          session_id: TILE_STORED_ID
        } as never
      }

      return transcript('main-pane answer') as never
    })

    // Seed a tile so reconcileTileTranscripts has a target.
    setSessions([]) // bot chats are hidden from $sessions — the whole point

    await act(async () => {
      await reconcileTileTranscriptsForTest({
        tiles: [{ storedSessionId: TILE_STORED_ID, runtimeId: TILE_RUNTIME_ID }],
        requestSequenceRef,
        signatureRef,
        updateSessionState
      })
    })

    // Behavior assertions:
    expect(updaterCallCount).toBeGreaterThan(0)
    expect(getLatestSessionMessages).toHaveBeenCalledWith(TILE_STORED_ID, undefined, { passive: true })
  })

  it('reconciles an idle tile while the main pane is busy', async () => {
    const runtimeId = 'runtime-idle-tile'
    const storedId = 'stored-idle-tile'
    const idleState = createClientSessionState(storedId)

    setBusy(true)
    publishSessionState(runtimeId, idleState)
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('idle tile update', storedId) as never)

    const updateSessionState = vi.fn((sessionId: string, updater: (state: typeof idleState) => typeof idleState) => {
      expect(sessionId).toBe(runtimeId)

      return updater(idleState)
    })

    await reconcileTileTranscriptsForTest({
      tiles: [{ runtimeId, storedSessionId: storedId }],
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      updateSessionState
    })

    expect(getLatestSessionMessages).toHaveBeenCalledWith(storedId, undefined, { passive: true })
    expect(updateSessionState).toHaveBeenCalledTimes(1)
  })

  it('does not reconcile a busy tile when the main pane is idle', async () => {
    const runtimeId = 'runtime-busy-tile'
    const storedId = 'stored-busy-tile'
    const liveState = createClientSessionState(storedId)

    liveState.busy = true
    liveState.messages = [
      {
        id: 'live-assistant',
        parts: [{ text: 'streaming answer', type: 'text' }],
        pending: true,
        role: 'assistant'
      }
    ]
    publishSessionState(runtimeId, liveState)
    vi.mocked(getLatestSessionMessages).mockResolvedValue({ messages: [], session_id: storedId } as never)

    const updateSessionState = vi.fn()

    await reconcileTileTranscriptsForTest({
      tiles: [{ runtimeId, storedSessionId: storedId }],
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      updateSessionState
    })

    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(updateSessionState).not.toHaveBeenCalled()
  })

  it('discards a tile snapshot when the tile closes during the read', async () => {
    const runtimeId = 'runtime-closing-tile'
    const storedId = 'stored-closing-tile'
    let resolveRead: (value: unknown) => void = () => undefined

    $sessionTiles.set([{ runtimeId, storedSessionId: storedId }])
    publishSessionState(runtimeId, createClientSessionState(storedId))
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(
      new Promise(resolve => {
        resolveRead = resolve
      }) as never
    )

    const updateSessionState = vi.fn()

    const reconcile = reconcileTileTranscriptsForTest({
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      updateSessionState
    })

    $sessionTiles.set([])
    resolveRead(transcript('stale tile answer', storedId))
    await reconcile

    expect(updateSessionState).not.toHaveBeenCalled()
  })

  it('isolates tile transcript reads by connection and profile while preserving the legacy local path', async () => {
    vi.mocked(getLatestSessionMessages).mockImplementation(async storedId => transcript(storedId, storedId) as never)

    const updateSessionState: Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState'] = vi.fn(
      (_sessionId, updater) => updater({} as Parameters<typeof updater>[0])
    )

    await reconcileTileTranscriptsForTest({
      tiles: [
        {
          ownerRoute: {
            connectionId: 'connection-a',
            mode: 'remote',
            profile: 'shared-profile',
            targetProfile: 'target-a'
          },
          runtimeId: 'runtime-a',
          storedSessionId: 'stored-a'
        },
        {
          ownerRoute: { connectionId: 'connection-b', mode: 'remote', profile: 'shared-profile' },
          runtimeId: 'runtime-b',
          storedSessionId: 'stored-b'
        },
        { runtimeId: 'runtime-local', storedSessionId: 'stored-local' }
      ],
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map<string, string>() },
      updateSessionState
    })

    // Every tile read is passive: it may serve from a warm owner backend but
    // must never cold-start one (#103375).
    expect(getLatestSessionMessages).toHaveBeenCalledWith(
      'stored-a',
      { connectionId: 'connection-a', profile: 'target-a' },
      { passive: true }
    )
    expect(getLatestSessionMessages).toHaveBeenCalledWith(
      'stored-b',
      { connectionId: 'connection-b', profile: 'shared-profile' },
      { passive: true }
    )
    expect(getLatestSessionMessages).toHaveBeenCalledWith('stored-local', undefined, { passive: true })
    expect(updateSessionState).toHaveBeenCalledWith('runtime-a', expect.any(Function), 'stored-a')
    expect(updateSessionState).toHaveBeenCalledWith('runtime-b', expect.any(Function), 'stored-b')
    expect(updateSessionState).toHaveBeenCalledWith('runtime-local', expect.any(Function), 'stored-local')
  })

  it('restores carrier-only todo state during a background tile reconciliation', async () => {
    const runtimeId = 'runtime-carrier-tile'
    const storedId = 'stored-carrier-tile'
    const updateSessionState = vi.fn((_sessionId, updater) =>
      updater(createClientSessionState(storedId))
    ) as Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState']

    vi.mocked(getLatestSessionMessages).mockResolvedValue(carrierTranscript(storedId) as never)

    await reconcileTileTranscriptsForTest({
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      tiles: [{ runtimeId, storedSessionId: storedId }],
      updateSessionState
    })

    expect($todosBySession.get()[runtimeId]).toEqual(carrierTodos)
  })

  it('restores row 277757 beyond a full 120-message owner-scoped tile tail', async () => {
    const runtimeId = 'runtime-paged-tile'
    const storedId = 'stored-paged-tile'
    const ownerRoute = {
      connectionId: 'remote-paged-tile',
      mode: 'remote' as const,
      profile: 'tile-route',
      targetProfile: 'tile-target'
    }
    const scope = { connectionId: ownerRoute.connectionId, profile: ownerRoute.targetProfile }
    const updateSessionState = vi.fn((_sessionId, updater) =>
      updater(createClientSessionState(storedId))
    ) as Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState']

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => ({
        content: `ordinary tile tail message ${index}`,
        role: 'assistant' as const,
        timestamp: index + 1
      })),
      session_id: storedId
    } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [{ ...legacyEvidenceMessage277757, session_id: storedId }],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 1 },
      session_id: storedId
    } as never)

    await reconcileTileTranscriptsForTest({
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      tiles: [{ ownerRoute, runtimeId, storedSessionId: storedId }],
      updateSessionState
    })

    expect(getSessionMessages).toHaveBeenCalledWith(storedId, scope, {
      projection: 'todo-state'
    })
    expect($todosBySession.get()[runtimeId]).toEqual(LEGACY_TODOS_277757)
  })

  it('clears tile Todo state after the bounded candidate domain is genuinely exhausted', async () => {
    const runtimeId = 'runtime-absent-tile'
    const storedId = 'stored-absent-tile'
    const updateSessionState = vi.fn((_sessionId, updater) =>
      updater(createClientSessionState(storedId))
    ) as Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState']

    setSessionTodos(runtimeId, [{ content: 'stale tile Todo', id: 'stale', status: 'in_progress' }])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => ({
        content: `ordinary absent tile tail ${index}`,
        role: 'assistant' as const,
        timestamp: index + 1
      })),
      session_id: storedId
    } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 0 },
      session_id: storedId
    } as never)

    await reconcileTileTranscriptsForTest({
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      tiles: [{ runtimeId, storedSessionId: storedId }],
      updateSessionState
    })

    expect(getSessionMessages).toHaveBeenCalledWith(storedId, undefined, {
      projection: 'todo-state'
    })
    expect($todosBySession.get()[runtimeId]).toBeUndefined()
  })

  it('restores carrier-only todo state during an active background reconciliation', async () => {
    const fixture = makeRefresh()

    vi.mocked(getLatestSessionMessages).mockResolvedValue(carrierTranscript() as never)

    await fixture.refresh()

    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toEqual(carrierTodos)
  })

  it('restores row 277757 beyond a full 120-message active background tail', async () => {
    const fixture = makeRefresh()

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => ({
        content: `ordinary tail message ${index}`,
        role: 'assistant' as const,
        timestamp: index + 1
      })),
      session_id: ACTIVE_STORED_ID
    } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [legacyEvidenceMessage277757],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 1 },
      session_id: ACTIVE_STORED_ID
    } as never)

    await fixture.refresh()

    expect(getSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, 'default', {
      projection: 'todo-state'
    })
    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toEqual(LEGACY_TODOS_277757)
  })

  it('clears active Todo state after the bounded candidate domain is genuinely exhausted', async () => {
    const fixture = makeRefresh()

    setSessionTodos(ACTIVE_RUNTIME_ID, [{ content: 'stale active Todo', id: 'stale', status: 'in_progress' }])
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => ({
        content: `ordinary absent active tail ${index}`,
        role: 'assistant' as const,
        timestamp: index + 1
      })),
      session_id: ACTIVE_STORED_ID
    } as never)
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [],
      pagination: { exhausted: true, has_more: false, limit: 32, next_before_id: null, returned: 0 },
      session_id: ACTIVE_STORED_ID
    } as never)

    await fixture.refresh()

    expect(getSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, 'default', {
      projection: 'todo-state'
    })
    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toBeUndefined()
  })

  it('replays legacy evidence row 277757 during a background tile reconciliation', async () => {
    const runtimeId = 'runtime-legacy-carrier-tile'
    const storedId = 'stored-legacy-carrier-tile'
    const updateSessionState = vi.fn((_sessionId, updater) =>
      updater(createClientSessionState(storedId))
    ) as Parameters<typeof reconcileTileTranscriptsForTest>[0]['updateSessionState']

    vi.mocked(getLatestSessionMessages).mockResolvedValue(legacyEvidenceTranscript(storedId) as never)

    await reconcileTileTranscriptsForTest({
      requestSequenceRef: { current: 0 },
      signatureRef: { current: new Map() },
      tiles: [{ runtimeId, storedSessionId: storedId }],
      updateSessionState
    })

    expect($todosBySession.get()[runtimeId]).toEqual(LEGACY_TODOS_277757)
  })

  it('replays legacy evidence row 277757 during active background reconciliation', async () => {
    const fixture = makeRefresh()

    vi.mocked(getLatestSessionMessages).mockResolvedValue(legacyEvidenceTranscript() as never)

    await fixture.refresh()

    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toEqual(LEGACY_TODOS_277757)
  })

  it('skips the tile fetch entirely when nothing changed (signature-gated)', async () => {
    $changeEventsAvailable.set(true)

    const TILE_RUNTIME_ID = 'runtime-tile-2'
    const TILE_STORED_ID = 'stored-tile-2'

    const signatureRef = { current: new Map<string, string>() }

    // Pre-seed the signature with what the mock returns → no-change tick.
    const pre = {
      messages: [
        { content: 'q', role: 'user', timestamp: 1 },
        { content: 'a', role: 'assistant', timestamp: 2 }
      ],
      session_id: TILE_STORED_ID
    }

    vi.mocked(getLatestSessionMessages).mockResolvedValue(pre as never)

    // Compute the same signature the reconcile will compute, and pre-seed it.
    const preSignature = sessionMessagesSignature(pre.messages as never)

    signatureRef.current.set(`tile:${TILE_STORED_ID}`, preSignature)

    const updateSessionState = vi.fn()
    const requestSequenceRef = { current: 0 }

    await act(async () => {
      await reconcileTileTranscriptsForTest({
        tiles: [{ storedSessionId: TILE_STORED_ID, runtimeId: TILE_RUNTIME_ID }],
        requestSequenceRef,
        signatureRef,
        updateSessionState
      })
    })

    expect(updateSessionState).not.toHaveBeenCalled()
  })

  it('refreshes a local/Desktop session when sessions.changed ticks', async () => {
    $changeEventsAvailable.set(true)
    $activeSessionId.set(ACTIVE_RUNTIME_ID)
    $selectedStoredSessionId.set(ACTIVE_STORED_ID)
    setSessionOwnerHint(ACTIVE_STORED_ID, {
      connectionId: 'stale-owner',
      mode: 'remote',
      profile: 'wrong-profile',
      targetProfile: 'wrong-target'
    })
    setSessions([
      {
        connectionId: 'future-visible-owner',
        id: ACTIVE_STORED_ID,
        profile: 'desktop-profile',
        source: 'desktop',
        targetProfile: 'must-not-rewrite-visible-row'
      } as never
    ])
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('external answer') as never)

    renderSync(fixture.refresh)

    act(() => notifySessionsChanged())

    await waitFor(() =>
      expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
        text: 'external answer'
      })
    )
    expect(getLatestSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, 'desktop-profile')
  })

  it('does not add a periodic transcript poll to local/Desktop sessions', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh)
    // Exactly the one connect-time pull (#94779) — no timer after it.
    expect(refresh).toHaveBeenCalledTimes(1)

    await act(async () => {
      vi.advanceTimersByTime(60_000)
      await Promise.resolve()
    })

    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('retains the existing periodic backstop for messaging sessions', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh, { activeIsMessaging: true })
    expect(refresh).toHaveBeenCalledTimes(1)
    await act(async () => Promise.resolve())
    refresh.mockClear()

    await act(async () => {
      vi.advanceTimersByTime(30_000)
      await Promise.resolve()
    })

    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('only defers an external tick while busy, then refreshes once after idle', async () => {
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh)
    refresh.mockClear() // drop the connect-time pull; this test is about busy transitions

    act(() => setBusy(true))
    act(() => setBusy(false))
    expect(refresh).not.toHaveBeenCalled()
    act(() => setBusy(true))

    act(() => {
      notifySessionsChanged()
      notifySessionsChanged()
    })
    expect(refresh).not.toHaveBeenCalled()

    act(() => setBusy(false))
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
  })

  it('pulls the open transcript once per (re)connect, not on session switches (#94779)', () => {
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    const { rerender } = renderSync(refresh, { gatewayState: 'connecting' })
    expect(refresh).not.toHaveBeenCalled()

    rerender({ gatewayState: 'open' })
    expect(refresh).toHaveBeenCalledTimes(1)

    rerender({ activeSessionId: 'runtime-other', activeStoredSessionId: 'stored-other', gatewayState: 'open' })
    expect(refresh).toHaveBeenCalledTimes(1)

    rerender({ activeSessionId: 'runtime-other', activeStoredSessionId: 'stored-other', gatewayState: 'closed' })
    rerender({ activeSessionId: 'runtime-other', activeStoredSessionId: 'stored-other', gatewayState: 'open' })
    expect(refresh).toHaveBeenCalledTimes(2)
  })

  it('coalesces a burst of global session-change ticks', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh)
    refresh.mockClear() // drop the connect-time pull; this test is about tick coalescing

    act(() => {
      for (let index = 0; index < 20; index += 1) {
        notifySessionsChanged()
      }
    })
    expect(refresh).toHaveBeenCalledTimes(1)

    await act(async () => {
      vi.advanceTimersByTime(9_999)
      await Promise.resolve()
    })

    expect(refresh).toHaveBeenCalledTimes(1)
  })

  it('refreshes the project tree on a sessions.changed tick, alongside the sessions list (#100354)', async () => {
    $changeEventsAvailable.set(true)

    renderSync(vi.fn(async () => undefined))

    act(() => notifySessionsChanged())

    await waitFor(() => expect(refreshProjectTree).toHaveBeenCalledTimes(1))
  })
})

describe('reconcileActiveTranscript', () => {
  it('keeps one failed assistant bubble when refresh rebuilds the same tail turn under a new id', async () => {
    const fixture = makeRefresh()
    fixture.state.messages = [
      {
        id: 'optimistic-user',
        parts: [{ text: 'question', type: 'text' }],
        role: 'user'
      },
      {
        error: 'connection lost after completion',
        errorSurface: { code: 'transport_lost', layer: 'streaming', retryable: true },
        id: 'assistant-live-before-refresh',
        parts: [{ text: 'answer', type: 'text' }],
        role: 'assistant'
      }
    ]
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('answer') as never)

    await fixture.refresh()

    const messages = fixture.states.get(ACTIVE_RUNTIME_ID)?.messages ?? []
    const assistants = messages.filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(1)
    expect(assistants[0]).toMatchObject({
      error: 'connection lost after completion',
      errorSurface: { code: 'transport_lost', layer: 'streaming', retryable: true },
      pending: false
    })
  })

  it('resolves and hydrates a messaging session from the messaging sessions store', async () => {
    setSessionOwnerHint(ACTIVE_STORED_ID, {
      connectionId: 'stale-messaging-owner',
      mode: 'remote',
      profile: 'wrong-profile',
      targetProfile: 'wrong-target'
    })
    setMessagingSessions([{ id: ACTIVE_STORED_ID, profile: 'messaging-profile', source: 'telegram' } as never])
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('telegram answer') as never)

    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, 'messaging-profile')
    expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'telegram answer'
    })
  })

  it('resolves and hydrates a cron session from the cron sessions store', async () => {
    setCronSessions([{ id: ACTIVE_STORED_ID, profile: 'cron-profile', source: 'cron' } as never])
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('cron progress') as never)

    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, 'cron-profile')
    expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'cron progress'
    })
  })

  it('fails closed when a hidden session id has multiple owner hints', async () => {
    const ambiguousStoredSessionId = 'ambiguous-hidden-chat'
    setSessionOwnerHint(ambiguousStoredSessionId, {
      connectionId: 'owner-a',
      mode: 'remote',
      profile: 'bot'
    })
    setSessionOwnerHint(ambiguousStoredSessionId, {
      connectionId: 'owner-b',
      mode: 'remote',
      profile: 'bot'
    })
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    fixture.selectedStoredSessionIdRef.current = ambiguousStoredSessionId

    await fixture.refresh()

    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(fixture.updateSessionState).not.toHaveBeenCalled()
  })

  it('keeps the active named-profile owner when a visible default duplicate shares the stored id', async () => {
    const S = ACTIVE_STORED_ID
    const namedOwner = { connectionId: 'remote', mode: 'remote' as const, profile: 'omar' }

    $activeSessionId.set(ACTIVE_RUNTIME_ID)
    $selectedStoredSessionId.set(S)
    setSessionOwnerHint(S, namedOwner)
    // Bot Chat lives in a workspace tile (often hidden from $sessions) while a
    // root-DB duplicate of the same stored id remains visible as `default`.
    $sessionTiles.set([
      {
        storedSessionId: S,
        runtimeId: ACTIVE_RUNTIME_ID,
        ownerRoute: namedOwner
      }
    ])
    // SAFETY: Ownership resolution reads only identity and profile fields; omitted session metadata is unused.
    setSessions([{ id: S, profile: 'default', source: 'desktop', connection_id: 'local' } as never])
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('named-profile answer'))

    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(S, expect.objectContaining({ profile: 'omar' }))
    expect(getLatestSessionMessages).not.toHaveBeenCalledWith(S, 'default')
    expect(getLatestSessionMessages).not.toHaveBeenCalledWith(S, expect.objectContaining({ profile: 'default' }))
  })

  it('uses the presentation profile when a hidden owner has no target profile', async () => {
    const hiddenStoredSessionId = 'hidden-no-target'
    setSessionOwnerHint(hiddenStoredSessionId, {
      connectionId: 'owner-no-target',
      mode: 'remote',
      profile: 'presentation-profile'
    })
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    fixture.selectedStoredSessionIdRef.current = hiddenStoredSessionId

    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledWith(hiddenStoredSessionId, {
      connectionId: 'owner-no-target',
      profile: 'presentation-profile'
    })
  })

  it('reads and publishes only the active hidden owner when another owner coexists', async () => {
    const ownerAStoredSessionId = 'owner-a-chat'
    const ownerBStoredSessionId = 'owner-b-hidden-chat'

    const ownerBRoute = {
      connectionId: 'owner-b',
      mode: 'remote' as const,
      profile: 'bot-route',
      targetProfile: 'bot-b'
    }

    setSessions([{ id: ownerAStoredSessionId, profile: 'bot-a', source: 'desktop' } as never])
    setSessionOwnerHint(ownerAStoredSessionId, {
      connectionId: 'owner-a',
      mode: 'remote',
      profile: 'bot-route',
      targetProfile: 'bot-a'
    })
    setSessionOwnerHint(ownerBStoredSessionId, ownerBRoute)
    const fixture = makeRefresh(resolveActiveTranscriptSession)
    fixture.selectedStoredSessionIdRef.current = ownerBStoredSessionId
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('owner B answer', ownerBStoredSessionId) as never)

    await fixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(1)
    expect(getLatestSessionMessages).toHaveBeenCalledWith(ownerBStoredSessionId, {
      connectionId: ownerBRoute.connectionId,
      profile: ownerBRoute.targetProfile
    })
    expect(fixture.updateSessionState).toHaveBeenCalledWith(
      ACTIVE_RUNTIME_ID,
      expect.any(Function),
      ownerBStoredSessionId
    )
    expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
      text: 'owner B answer'
    })
  })

  it.each([
    [
      'remote',
      {
        connectionId: 'remote-active-a',
        mode: 'remote' as const,
        profile: 'desktop-a',
        targetProfile: 'backend-a'
      },
      {
        connectionId: 'remote-active-b',
        mode: 'remote' as const,
        profile: 'desktop-b',
        targetProfile: 'backend-b'
      }
    ],
    [
      'local profile',
      { connectionId: 'local', mode: 'local' as const, profile: 'profile-a' },
      { connectionId: 'local', mode: 'local' as const, profile: 'profile-b' }
    ]
  ] as const)(
    'rejects an active %s owner response after same-ID/runtime rebind and permits the current-owner retry',
    async (_label, ownerA, ownerB) => {
      const staleRead = deferred<Awaited<ReturnType<typeof getLatestSessionMessages>>>()
      const identicalResponse = {
        ...carrierTranscript(),
        messages: [
          ...carrierTranscript().messages,
          { content: 'current owner visible transcript', role: 'assistant' as const, timestamp: 4 }
        ]
      } as Awaited<ReturnType<typeof getLatestSessionMessages>>
      let currentOwner: ActiveTranscriptRefreshDeps['resolveSession'] extends (...args: never[]) => infer R
        ? R
        : never = {
        ownerRoute: ownerA,
        profile: ownerA.profile
      }
      const fixture = makeRefresh(() => currentOwner)

      vi.mocked(getLatestSessionMessages)
        .mockReturnValueOnce(staleRead.promise as never)
        .mockResolvedValueOnce(identicalResponse as never)

      const staleRequest = fixture.refresh()

      expect(getLatestSessionMessages).toHaveBeenCalledWith(ACTIVE_STORED_ID, {
        connectionId: ownerA.connectionId,
        profile: 'targetProfile' in ownerA ? ownerA.targetProfile : ownerA.profile
      })

      currentOwner = { ownerRoute: ownerB, profile: ownerB.profile }
      staleRead.resolve(identicalResponse)
      await staleRequest

      expect(fixture.updateSessionState).not.toHaveBeenCalled()
      expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages).toEqual([])
      expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toBeUndefined()
      expect(fixture.signatureRef.current.size).toBe(0)

      await fixture.refresh()

      expect(getLatestSessionMessages).toHaveBeenNthCalledWith(2, ACTIVE_STORED_ID, {
        connectionId: ownerB.connectionId,
        profile: 'targetProfile' in ownerB ? ownerB.targetProfile : ownerB.profile
      })
      expect(fixture.updateSessionState).toHaveBeenCalledTimes(1)
      expect(fixture.states.get(ACTIVE_RUNTIME_ID)?.messages.at(-1)?.parts[0]).toMatchObject({
        text: 'current owner visible transcript'
      })
      expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toEqual(carrierTodos)
      expect(fixture.signatureRef.current.size).toBe(1)
    }
  )

  it('retries identical transcript hydration after an older publication is rejected and the newer read fails', async () => {
    const olderFixture = makeRefresh()
    const newerFixture = makeRefresh()
    const olderRead = deferred<ReturnType<typeof carrierTranscript>>()

    vi.mocked(getLatestSessionMessages)
      .mockReturnValueOnce(olderRead.promise as never)
      .mockRejectedValueOnce(new Error('newer read failed'))
      .mockResolvedValueOnce(carrierTranscript() as never)

    const olderRequest = olderFixture.refresh()
    await newerFixture.refresh()

    olderRead.resolve(carrierTranscript())
    await olderRequest

    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toBeUndefined()

    await olderFixture.refresh()

    expect(getLatestSessionMessages).toHaveBeenCalledTimes(3)
    expect($todosBySession.get()[ACTIVE_RUNTIME_ID]).toEqual(carrierTodos)
  })

  it('publishes changed authoritative messages once without duplicates', async () => {
    const fixture = makeRefresh()
    vi.mocked(getLatestSessionMessages).mockResolvedValue(transcript('new answer') as never)

    await fixture.refresh()

    expect(fixture.updateSessionState).toHaveBeenCalledTimes(1)
    const messages = fixture.states.get(ACTIVE_RUNTIME_ID)?.messages ?? []
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant'])
    expect(new Set(messages.map(message => message.id)).size).toBe(messages.length)

    await fixture.refresh()

    expect(fixture.updateSessionState).toHaveBeenCalledTimes(1)
  })

  it('preserves a local assistant error while hydrating authoritative messages', async () => {
    const fixture = makeRefresh()
    fixture.state.messages = [
      { id: '1-0-user', parts: [{ text: 'question', type: 'text' }], role: 'user' },
      { error: 'local failure', id: 'assistant-error', parts: [], role: 'assistant' }
    ]
    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [{ content: 'question', role: 'user', timestamp: 1 }],
      session_id: ACTIVE_STORED_ID
    } as never)

    await fixture.refresh()

    const messages = fixture.states.get(ACTIVE_RUNTIME_ID)?.messages ?? []
    expect(messages.map(message => message.id)).toEqual(['1-0-user', 'assistant-error'])
    expect(messages.at(-1)?.error).toBe('local failure')
  })

  it('does not clobber a busy stream', async () => {
    const fixture = makeRefresh()
    fixture.busyRef.current = true

    await fixture.refresh()

    expect(getLatestSessionMessages).not.toHaveBeenCalled()
    expect(fixture.updateSessionState).not.toHaveBeenCalled()
  })

  it('discards a response when the active session changes in flight', async () => {
    const fixture = makeRefresh()
    let resolve: ((value: unknown) => void) | undefined
    vi.mocked(getLatestSessionMessages).mockReturnValueOnce(
      new Promise(currentResolve => {
        resolve = currentResolve
      }) as never
    )

    const request = fixture.refresh()
    fixture.selectedStoredSessionIdRef.current = 'stored-other'
    fixture.activeSessionIdRef.current = 'runtime-other'
    resolve?.(transcript('stale answer'))
    await request

    expect(fixture.updateSessionState).not.toHaveBeenCalled()
  })
})

describe('windowIsActivelyViewed', () => {
  it('requires both DOM visibility and keyboard focus', () => {
    expect(windowIsActivelyViewed({ focused: true, visibilityState: 'visible' })).toBe(true)
    expect(windowIsActivelyViewed({ focused: false, visibilityState: 'visible' })).toBe(false)
    expect(windowIsActivelyViewed({ focused: true, visibilityState: 'hidden' })).toBe(false)
  })
})

describe('rehydrateLiveSessionStatuses', () => {
  it('restores running sessions after reconnect without opening them', () => {
    const now = 1_800_000_000_000

    rehydrateLiveSessionStatuses(
      {
        sessions: [
          {
            id: 'runtime-overnight',
            last_active: (now - SESSION_WATCHDOG_TIMEOUT_MS - 1_000) / 1000,
            session_key: 'overnight-exam-learning',
            status: 'working'
          },
          {
            id: 'runtime-cleanup',
            last_active: now / 1000,
            session_key: 'temporary-file-cleanup',
            status: 'working'
          }
        ]
      },
      now
    )

    expect($workingSessionIds.get()).toEqual(['overnight-exam-learning', 'temporary-file-cleanup'])
    expect($stalledSessionIds.get()).toEqual(['overnight-exam-learning'])
    expect($attentionSessionIds.get()).toEqual([])
  })

  it('restores a waiting turn as working and needing attention', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-needs-user', session_key: 'needs-user', status: 'waiting' }]
    })

    expect($workingSessionIds.get()).toEqual(['needs-user'])
    expect($attentionSessionIds.get()).toEqual(['needs-user'])
    expect($stalledSessionIds.get()).toEqual([])
  })

  it('reaps stale unfinished todos without disturbing finished linger or authoritative continuations', () => {
    setSessionTodos('runtime-stale', [{ content: 'stale task', id: 'stale', status: 'in_progress' }])
    setSessionTodos('runtime-finished', [{ content: 'done task', id: 'done', status: 'completed' }])
    setSessionTodos('runtime-active-goal', [{ content: 'continuing task', id: 'active', status: 'pending' }])
    setSessionTodos('runtime-paused-goal', [{ content: 'paused task', id: 'paused', status: 'in_progress' }])
    applyTodoContinuationSnapshot('runtime-active-goal', { revision: 1, state: 'active' })
    applyTodoContinuationSnapshot('runtime-paused-goal', { revision: 1, state: 'paused' })

    rehydrateLiveSessionStatuses({
      sessions: [
        { id: 'runtime-stale', session_key: 'stored-stale', status: 'working' },
        { id: 'runtime-finished', session_key: 'stored-finished', status: 'working' },
        { id: 'runtime-active-goal', session_key: 'stored-active-goal', status: 'working' },
        { id: 'runtime-paused-goal', session_key: 'stored-paused-goal', status: 'working' }
      ]
    })

    expect($todoProgressBySession.get()['stored-stale']).toBe('0/1')

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($todosBySession.get()['runtime-stale']).toBeUndefined()
    expect($todoProgressBySession.get()['stored-stale']).toBeUndefined()
    expect($todosBySession.get()['runtime-finished']).toHaveLength(1)
    expect($todosBySession.get()['runtime-active-goal']).toHaveLength(1)
    expect($todosBySession.get()['runtime-paused-goal']).toHaveLength(1)
  })

  it('ignores idle, starting, and malformed live-session rows', () => {
    rehydrateLiveSessionStatuses({
      sessions: [
        { id: 'runtime-idle', session_key: 'idle-session', status: 'idle' },
        { id: 'runtime-starting', session_key: 'starting-session', status: 'starting' },
        { id: 'runtime-malformed', status: 'working' }
      ]
    })

    expect($workingSessionIds.get()).toEqual([])
    expect($attentionSessionIds.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
  })
})

describe('typing-aware sessions.changed deferral', () => {
  // Dedicated harness: the sessions-list spy must be the exact fn handed to
  // the hook (the shared harness above wires inner vi.fn()s and its outer spy
  // observes the transcript path instead), and EVERY param must keep a stable
  // identity across the tick-driven re-renders — an unstable prop would
  // re-run the connect-reseed effect and re-subscribe the throttle each
  // render, polluting the counts under observation.
  function renderTypingSync(refreshSessions: () => Promise<void>) {
    const stable = {
      refreshActiveTranscript: async () => undefined,
      refreshCronJobs: vi.fn(),
      refreshCurrentModel: vi.fn(),
      refreshHermesConfig: vi.fn(),
      refreshMessagingSessions: vi.fn(),
      requestGateway: vi.fn(async () => ({ sessions: [] })) as never,
      // Required by the hook's params. This harness never drives the
      // transcript path, so the updater just runs against a throwaway state —
      // but it must live in `stable` like every other prop, since a fresh
      // identity per render would re-run the connect-reseed effect.
      updateSessionState: vi.fn(
        (
          _sessionId: string,
          updater: (state: ReturnType<typeof createClientSessionState>) => ReturnType<typeof createClientSessionState>
        ) => updater(createClientSessionState(ACTIVE_STORED_ID))
      )
    }

    return renderHook(() => {
      useBackgroundSync({
        activeConnectionId: 'local',
        activeGatewayProfile: 'default',
        activeIsMessaging: false,
        activeSessionId: null,
        activeStoredSessionId: null,
        freshDraftReady: false,
        gatewayState: 'open',
        ...stable,
        refreshSessions
      })
    })
  }

  const typeKey = (): void => {
    window.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'a' }))
  }

  /** Mount, land one full throttle cycle so lastRunAt sits at a known clock
   *  position, then clear the spy. */
  async function primeThrottle(refreshSessions: ReturnType<typeof vi.fn>): Promise<void> {
    act(() => notifySessionsChanged())
    await act(async () => {
      // One SESSIONS_LIST_TICK_GAP_MS covers both the immediate first tick
      // and any trailing timer the burst armed.
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
    })
    refreshSessions.mockClear()
  }

  it('holds the trailing sessions.changed refresh while a typing burst is live, then lands it once after the keyboard quiets', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refreshSessions = vi.fn(async () => undefined)

    renderTypingSync(refreshSessions)
    await primeThrottle(refreshSessions)

    // A ~6s continuous burst: keys every 200ms, broadcasts every ~1s. The
    // first broadcast finds the throttle gap already elapsed (primed), so the
    // deferral engages immediately and must hold for the whole burst.
    for (let index = 0; index < 30; index += 1) {
      typeKey()

      if (index % 5 === 0) {
        act(() => notifySessionsChanged())
      }

      await act(async () => {
        vi.advanceTimersByTime(200)
        await Promise.resolve()
      })
    }

    // The heavy list pass must not have landed under the keystrokes.
    expect(refreshSessions).not.toHaveBeenCalled()

    // Last key at ~5.8s; quiet threshold elapses ~7.3s → the held pass lands
    // exactly once shortly after.
    await act(async () => {
      vi.advanceTimersByTime(2_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)

    // ...and nothing extra afterwards without further broadcasts — mid-burst
    // ticks must not have stacked trailing timers behind the promised pass.
    await act(async () => {
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('holds through a burst longer than the throttle gap and lands once after the keyboard quiets', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refreshSessions = vi.fn(async () => undefined)

    renderTypingSync(refreshSessions)
    await primeThrottle(refreshSessions)

    // Keys every 200ms for ~22s — longer than SESSIONS_LIST_TICK_GAP_MS.
    // Broadcasts keep flowing; the heavy pass must not land under them.
    for (let index = 0; index < 110; index += 1) {
      typeKey()

      if (index % 10 === 0) {
        act(() => notifySessionsChanged())
      }

      await act(async () => {
        vi.advanceTimersByTime(200)
        await Promise.resolve()
      })
    }

    expect(refreshSessions).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(2_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)

    await act(async () => {
      vi.advanceTimersByTime(10_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('does not defer anything when the keyboard has been idle', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refreshSessions = vi.fn(async () => undefined)

    renderTypingSync(refreshSessions)
    await primeThrottle(refreshSessions)

    act(() => notifySessionsChanged())

    await act(async () => {
      vi.advanceTimersByTime(11_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })
})

describe('isTypingBurstActive', () => {
  it('marks a burst warm for the quiet threshold and cold at it', () => {
    resetTypingActivityTracking()

    // No keyboard history → nothing to defer for.
    expect(isTypingBurstActive(1_000_000)).toBe(false)

    noteRendererKeyboardActivity(1_000_000)
    expect(isTypingBurstActive(1_000_000)).toBe(true)
    expect(isTypingBurstActive(1_000_000 + 1_499)).toBe(true)

    // Exactly one quiet threshold after the last key the keyboard is cold.
    expect(isTypingBurstActive(1_000_000 + 1_500)).toBe(false)
  })
})
