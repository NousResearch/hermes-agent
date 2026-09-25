import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $sidebarShowArchived } from '@/store/layout'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery } from '@/store/power'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState
} from '@/store/session-states'
import { loadArchivedSessions } from '@/store/sidebar-archive'

import { useBackgroundSync } from './use-background-sync'

vi.mock('@/store/sidebar-archive', () => ({
  loadArchivedSessions: vi.fn()
}))

const noop = () => undefined
const requestGateway = async () => ({ sessions: [] })

function render(
  activeGatewayProfile: string,
  activeConnectionId: string,
  refreshSessions: () => Promise<void>,
  gatewayRequest = requestGateway
) {
  return renderHook(
    ({ connectionId, profile }: { connectionId: string; profile: string }) => {
      useBackgroundSync({
        activeConnectionId: connectionId,
        activeGatewayProfile: profile,
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
        refreshSessions,
        requestGateway: gatewayRequest
      })
    },
    { initialProps: { connectionId: activeConnectionId, profile: activeGatewayProfile } }
  )
}

describe('useBackgroundSync profile-scoped session refresh', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $activeSessionId.set(null)
    $changeEventsAvailable.set(false)
    $cronChangeTick.set(0)
    $sessionsChangeTick.set(0)
    $sidebarShowArchived.set(false)
    vi.mocked(loadArchivedSessions).mockReset()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('coalesces change ticks while the live status request is pending', async () => {
    $changeEventsAvailable.set(true)
    let release!: (value: { sessions: [] }) => void

    const pending = new Promise<{ sessions: [] }>(resolve => {
      release = resolve
    })

    const request = vi.fn(() => pending)
    render('default', 'local', async () => undefined, request)
    await act(async () => undefined)

    for (let tick = 1; tick <= 8; tick += 1) {
      await act(async () => {
        $sessionsChangeTick.set(tick)
      })
    }

    expect(request).toHaveBeenCalledTimes(1)
    await act(async () => {
      release({ sessions: [] })
    })
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('refreshes the session list after the active gateway profile changes', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const hook = render('default', 'local', refreshSessions)

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
    refreshSessions.mockClear()

    hook.rerender({ connectionId: 'local', profile: 'nova' })

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('refreshes the session list when the backend changes but the profile name does not', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    const hook = render('default', 'work', refreshSessions)

    await act(async () => undefined)
    refreshSessions.mockClear()

    hook.rerender({ connectionId: 'homelab', profile: 'default' })

    await act(async () => undefined)
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('reloads archived sessions after an external session change while the Archived view is open', async () => {
    $changeEventsAvailable.set(true)
    $sidebarShowArchived.set(true)
    render('default', 'local', async () => undefined)

    await act(async () => {
      $sessionsChangeTick.set(1)
    })

    expect(loadArchivedSessions).toHaveBeenCalledTimes(1)
  })
})

describe('useBackgroundSync keeps a quiet working turn live', () => {
  // A long tool call emits no events; only the live-status poll says the turn
  // is still running. Past the silence window plus any confirmation grace.
  const PAST_SILENCE_MS = LIVE_TURN_EVENT_SILENCE_MS + 20_000

  const working = async () => ({
    sessions: [{ id: 'rt-quiet', last_active: Date.now() / 1000, session_key: 's-quiet', status: 'working' }]
  })

  function startQuietTurn() {
    publishSessionState('rt-quiet', {
      ...createClientSessionState('s-quiet'),
      awaitingResponse: true,
      busy: true,
      sawAssistantPayload: true,
      turnLive: true,
      turnStartedAt: Date.now()
    })
    noteSessionEvent('rt-quiet')
  }

  // The silence settle marks the turn interrupted, which drops its later events.
  const forceSettled = () => $sessionStates.get()['rt-quiet']?.interrupted === true

  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $activeSessionId.set(null)
    $changeEventsAvailable.set(true)
    $onBattery.set(false)
    $sessionsChangeTick.set(0)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    clearAllSessionStates()
    $onBattery.set(false)
    vi.useRealTimers()
  })

  it('while the window is not focused', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    startQuietTurn()
    render('default', 'local', async () => undefined, working)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_SILENCE_MS)
    })

    expect($sessionStates.get()['rt-quiet']?.busy).toBe(true)
    expect(forceSettled()).toBe(false)
  })

  it('on battery, where the backstop poll is slower than the silence window', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    $onBattery.set(true)
    startQuietTurn()
    render('default', 'local', async () => undefined, working)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_SILENCE_MS)
    })

    expect($sessionStates.get()['rt-quiet']?.busy).toBe(true)
    expect(forceSettled()).toBe(false)
  })

  it('but still settles it when the backend stops answering', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    startQuietTurn()
    const request = vi.fn(working)
    render('default', 'local', async () => undefined, request)
    await act(async () => undefined)
    request.mockImplementation(async () => {
      throw new Error('gateway gone')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_SILENCE_MS)
    })

    expect($sessionStates.get()['rt-quiet']?.busy).toBe(false)
    expect(forceSettled()).toBe(true)
  })
})
