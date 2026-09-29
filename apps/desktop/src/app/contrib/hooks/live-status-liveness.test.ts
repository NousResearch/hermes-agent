import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $changeEventsAvailable, resetLiveSync } from '@/store/live-sync'
import { $onBattery } from '@/store/power'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  noteSessionEvent,
  publishSessionState
} from '@/store/session-states'

import { resetLiveRuntimeTracking, useBackgroundSync } from './use-background-sync'

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal()),
  refreshActiveProfile: vi.fn(async () => undefined)
}))

beforeEach(() => {
  vi.useFakeTimers()
  $changeEventsAvailable.set(true)
  $activeSessionId.set('runtime-live')
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  resetLiveSync()
  $onBattery.set(false)
  $activeSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

it.each([
  { focused: false, visibility: 'hidden', battery: false },
  { focused: false, visibility: 'visible', battery: false },
  { focused: true, visibility: 'visible', battery: true },
  { focused: false, visibility: 'hidden', battery: true }
] as const)('keeps a healthy quiet turn alive with %j, but still detects a dead backend', async options => {
  vi.spyOn(document, 'hasFocus').mockReturnValue(options.focused)
  vi.spyOn(document, 'visibilityState', 'get').mockReturnValue(options.visibility)
  $onBattery.set(options.battery)

  let working = false

  const requestGateway = vi.fn(async () => ({
    sessions: working ? [{ id: 'runtime-live', session_key: 'stored-live', status: 'working' }] : []
  }))

  const noop = vi.fn(async () => undefined)
  renderHook(() =>
    useBackgroundSync({
      activeConnectionId: 'local',
      activeGatewayProfile: 'default',
      activeIsMessaging: false,
      activeSessionId: null,
      activeStoredSessionId: null,
      freshDraftReady: false,
      gatewayState: 'open',
      refreshActiveTranscript: noop,
      refreshCronJobs: noop,
      refreshCurrentModel: noop,
      refreshHermesConfig: noop,
      refreshMessagingSessions: noop,
      refreshSessions: noop,
      requestGateway: requestGateway as never,
      updateSessionState: vi.fn()
    })
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1_000)
  })

  // Start after the idle poll was installed: battery cadence must shorten now.
  working = true
  act(() => {
    publishSessionState('runtime-live', {
      ...createClientSessionState('stored-live'),
      busy: true,
      awaitingResponse: true,
      turnLive: true
    })
    noteSessionEvent('runtime-live')
  })

  for (let i = 0; i < 5; i++) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect($workingSessionIds.get()).toContain('stored-live')
    expect($sessionStates.get()['runtime-live'].messages.some(message => message.errorSurface)).toBe(false)
  }

  // Failed RPCs must not count as proof of life, even with keepalive polling.
  requestGateway.mockRejectedValue(new Error('backend unavailable'))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(45_000)
  })
  expect($workingSessionIds.get()).not.toContain('stored-live')
  expect($sessionStates.get()['runtime-live'].messages.some(message => message.errorSurface?.retryable)).toBe(true)

  requestGateway.mockClear()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(requestGateway).not.toHaveBeenCalled()
})
