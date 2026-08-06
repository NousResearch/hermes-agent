import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { $sessionsChangeTick, notifySessionsChanged, setChangeEventsAvailable } from '@/store/live-sync'
import {
  $attentionSessionIds,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  SESSION_WATCHDOG_TIMEOUT_MS
} from '@/store/session-states'

import { reconcileActiveTranscript, rehydrateLiveSessionStatuses, useBackgroundSync } from './use-background-sync'

describe('rehydrateLiveSessionStatuses', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.clearAllTimers()
    vi.useRealTimers()
    setChangeEventsAvailable(false)
    $sessionsChangeTick.set(0)
    clearAllSessionStates()
  })

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

describe('active stored transcript sync', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setChangeEventsAvailable(true)
  })

  afterEach(() => {
    cleanup()
    vi.clearAllTimers()
    vi.useRealTimers()
    setChangeEventsAvailable(false)
    $sessionsChangeTick.set(0)
  })

  it('coalesces sessions.changed and refreshes an open desktop-source transcript', async () => {
    const refreshActiveTranscript = vi.fn()

    renderHook(() =>
      useBackgroundSync({
        activeGatewayProfile: 'default',
        activeIsMessaging: false,
        activeSessionId: 'runtime-desktop',
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript,
        refreshCronJobs: vi.fn(),
        refreshCurrentModel: vi.fn(),
        refreshHermesConfig: vi.fn(),
        refreshMessagingSessions: vi.fn(),
        refreshSessions: vi.fn(),
        requestGateway: vi.fn(async () => ({ sessions: [] })) as never
      })
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })
    refreshActiveTranscript.mockClear()

    act(() => {
      notifySessionsChanged()
      notifySessionsChanged()
      notifySessionsChanged()
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(249)
    })
    expect(refreshActiveTranscript).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(refreshActiveTranscript).toHaveBeenCalledTimes(1)
  })

  it('dedupes the persisted external user row while preserving a live assistant tail', () => {
    const history: ChatMessage[] = [
      { id: 'stored-user-1', parts: [{ text: 'old prompt', type: 'text' }], role: 'user' },
      { id: 'stored-assistant-1', parts: [{ text: 'old reply', type: 'text' }], role: 'assistant' }
    ]

    const stored = [
      ...history,
      { id: 'stored-iphone-user', parts: [{ text: 'sent from iPhone', type: 'text' }], role: 'user' }
    ] satisfies ChatMessage[]

    const current = [
      ...history,
      { id: 'optimistic-user', parts: [{ text: 'sent from iPhone', type: 'text' }], role: 'user' },
      {
        id: 'assistant-stream-1',
        parts: [{ text: 'working', type: 'text' }],
        pending: true,
        role: 'assistant'
      }
    ] satisfies ChatMessage[]

    const reconciled = reconcileActiveTranscript(stored, current)

    expect(reconciled.filter(message => chatMessageText(message) === 'sent from iPhone')).toHaveLength(1)
    expect(reconciled.at(-1)).toMatchObject({ id: 'assistant-stream-1', pending: true })
  })
})
