// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import {
  $attentionSessionIds,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  SESSION_WATCHDOG_TIMEOUT_MS
} from '@/store/session-states'

import { rehydrateLiveSessionStatuses, useBackgroundSync } from './use-background-sync'

describe('useBackgroundSync', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' })
    $connection.set({ baseUrl: 'http://shared-gateway', mode: 'remote', profile: 'san' } as never)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('refreshes local sessions while a shared gateway is open', () => {
    const refreshSessions = vi.fn().mockResolvedValue(undefined)

    renderHook(() =>
      useBackgroundSync({
        activeGatewayProfile: 'san',
        activeIsMessaging: false,
        activeSessionId: null,
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript: vi.fn().mockResolvedValue(undefined),
        refreshCronJobs: vi.fn().mockResolvedValue(undefined),
        refreshCurrentModel: vi.fn().mockResolvedValue(undefined),
        refreshHermesConfig: vi.fn().mockResolvedValue(undefined),
        refreshMessagingSessions: vi.fn().mockResolvedValue(undefined),
        refreshSessions,
        requestGateway: vi.fn().mockResolvedValue({ sessions: [] })
      })
    )

    expect(refreshSessions).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(2_000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(2)
    const pollOptions = refreshSessions.mock.calls[1][0]

    expect(pollOptions?.refreshCronJobs).toBe(false)
    expect(pollOptions?.skipIfBusy).toBe(true)
    expect(pollOptions?.shouldApply?.()).toBe(true)

    act(() => {
      $connection.set({ baseUrl: 'http://different-gateway', mode: 'remote', profile: 'work' } as never)
    })

    expect(pollOptions?.shouldApply?.()).toBe(false)
  })

  it('refreshes the open local transcript while a shared gateway is visible', () => {
    const refreshActiveTranscript = vi.fn().mockResolvedValue(undefined)

    renderHook(() =>
      useBackgroundSync({
        activeGatewayProfile: 'san',
        activeIsMessaging: false,
        activeSessionId: 'runtime-local',
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript,
        refreshCronJobs: vi.fn().mockResolvedValue(undefined),
        refreshCurrentModel: vi.fn().mockResolvedValue(undefined),
        refreshHermesConfig: vi.fn().mockResolvedValue(undefined),
        refreshMessagingSessions: vi.fn().mockResolvedValue(undefined),
        refreshSessions: vi.fn().mockResolvedValue(undefined),
        requestGateway: vi.fn().mockResolvedValue({ sessions: [] })
      })
    )

    expect(refreshActiveTranscript).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(2_000)
    })

    expect(refreshActiveTranscript).toHaveBeenCalledTimes(1)
  })

  it('coalesces slow open-transcript refreshes', async () => {
    let release: (() => void) | undefined

    const refreshActiveTranscript = vi.fn(
      () =>
        new Promise<void>(resolve => {
          release = resolve
        })
    )

    renderHook(() =>
      useBackgroundSync({
        activeGatewayProfile: 'san',
        activeIsMessaging: false,
        activeSessionId: 'runtime-local',
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript,
        refreshCronJobs: vi.fn().mockResolvedValue(undefined),
        refreshCurrentModel: vi.fn().mockResolvedValue(undefined),
        refreshHermesConfig: vi.fn().mockResolvedValue(undefined),
        refreshMessagingSessions: vi.fn().mockResolvedValue(undefined),
        refreshSessions: vi.fn().mockResolvedValue(undefined),
        requestGateway: vi.fn().mockResolvedValue({ sessions: [] })
      })
    )

    await act(async () => {
      vi.advanceTimersByTime(4_000)
    })

    expect(refreshActiveTranscript).toHaveBeenCalledTimes(1)

    await act(async () => {
      release?.()
      await Promise.resolve()
      vi.advanceTimersByTime(2_000)
    })

    expect(refreshActiveTranscript).toHaveBeenCalledTimes(2)
  })

  it('does not poll local-only sessions', () => {
    $connection.set({ mode: 'local' } as never)
    const refreshSessions = vi.fn().mockResolvedValue(undefined)
    const refreshActiveTranscript = vi.fn().mockResolvedValue(undefined)

    renderHook(() =>
      useBackgroundSync({
        activeGatewayProfile: 'san',
        activeIsMessaging: false,
        activeSessionId: 'runtime-local',
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript,
        refreshCronJobs: vi.fn().mockResolvedValue(undefined),
        refreshCurrentModel: vi.fn().mockResolvedValue(undefined),
        refreshHermesConfig: vi.fn().mockResolvedValue(undefined),
        refreshMessagingSessions: vi.fn().mockResolvedValue(undefined),
        refreshSessions,
        requestGateway: vi.fn().mockResolvedValue({ sessions: [] })
      })
    )

    act(() => {
      vi.advanceTimersByTime(2_000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(refreshActiveTranscript).not.toHaveBeenCalled()
  })
})

describe('rehydrateLiveSessionStatuses', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
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
