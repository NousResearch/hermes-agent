import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $changeEventsAvailable, notifySessionsChanged, resetLiveSync } from '@/store/live-sync'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  setBusy,
  setMessagingSessions,
  setSessionOwnerHint,
  setSessions
} from '@/store/session'
import {
  $attentionSessionIds,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  SESSION_WATCHDOG_TIMEOUT_MS
} from '@/store/session-states'

import {
  type ActiveTranscriptRefreshDeps,
  reconcileActiveTranscript,
  rehydrateLiveSessionStatuses,
  resolveActiveTranscriptSession,
  useBackgroundSync,
  windowIsActivelyViewed
} from './use-background-sync'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')

const ACTIVE_RUNTIME_ID = 'runtime-active'
const ACTIVE_STORED_ID = 'stored-active'

function transcript(answer: string, sessionId = ACTIVE_STORED_ID) {
  return {
    messages: [
      { content: 'question', role: 'user', timestamp: 1 },
      { content: answer, role: 'assistant', timestamp: 2 }
    ],
    session_id: sessionId
  }
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

  return { activeSessionIdRef, busyRef, refresh, selectedStoredSessionIdRef, state, states, updateSessionState }
}

function useSyncHarness({
  activeIsMessaging = false,
  activeSessionId,
  activeStoredSessionId,
  refreshActiveTranscript
}: {
  activeIsMessaging?: boolean
  activeSessionId: string | null
  activeStoredSessionId: string | null
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
    gatewayState: 'open',
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

function renderSync(
  refreshActiveTranscript: () => Promise<void>,
  options: { activeIsMessaging?: boolean; activeSessionId?: null | string; activeStoredSessionId?: null | string } = {}
) {
  return renderHook(() =>
    useSyncHarness({
      activeSessionId: ACTIVE_RUNTIME_ID,
      activeStoredSessionId: ACTIVE_STORED_ID,
      refreshActiveTranscript,
      ...options
    })
  )
}

beforeEach(() => {
  // visiblePoll only ticks while the window is actively viewed; jsdom's
  // document.hasFocus() is not reliably true, so pin it for these tests.
  vi.spyOn(document, 'hasFocus').mockReturnValue(true)
})

afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  resetLiveSync()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  setSessions([])
  setMessagingSessions([])
  setBusy(false)
  vi.clearAllMocks()
  vi.restoreAllMocks()
  clearAllSessionStates()
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
    expect(refresh).not.toHaveBeenCalled()

    await act(async () => {
      vi.advanceTimersByTime(60_000)
      await Promise.resolve()
    })

    expect(refresh).not.toHaveBeenCalled()
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
    setBusy(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh)

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

  it('coalesces a burst of global session-change ticks', async () => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    const refresh = vi.fn(async () => undefined)

    renderSync(refresh)

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
})

describe('reconcileActiveTranscript', () => {
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
    vi.mocked(getLatestSessionMessages).mockResolvedValue(
      transcript('owner B answer', ownerBStoredSessionId) as never
    )

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
