import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $sidebarShowArchived } from '@/store/layout'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery } from '@/store/power'
import { $activeSessionId } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
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
    // The safety-net interval polls require focus, and jsdom's
    // document.hasFocus() is not reliably true, so pin it (visibility stays
    // at jsdom's "visible" default for the heavy pass).
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
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
  // A foreground tool emits nothing between tool.start and tool.complete; the
  // backend is the only witness that the turn is still running.
  const LONG_TOOL_MS = LIVE_TURN_EVENT_SILENCE_MS * 4

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

  const cardShown = () => Boolean($sessionStates.get()['rt-quiet']?.messages.some(message => message.errorSurface))

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
      await vi.advanceTimersByTimeAsync(LONG_TOOL_MS)
    })

    expect($workingSessionIds.get()).toContain('s-quiet')
    expect($sessionStates.get()['rt-quiet']?.interrupted).toBeFalsy()
    expect(cardShown()).toBe(false)
  })

  it('on battery, where the backstop poll is slower than the silence window', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(true)
    $onBattery.set(true)
    startQuietTurn()
    render('default', 'local', async () => undefined, working)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LONG_TOOL_MS)
    })

    expect($workingSessionIds.get()).toContain('s-quiet')
    expect(cardShown()).toBe(false)
  })

  it('while the gateway stops answering', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    startQuietTurn()
    const request = vi.fn(working)
    render('default', 'local', async () => undefined, request)
    await act(async () => undefined)
    request.mockImplementation(async () => {
      throw new Error('Hermes gateway unavailable')
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LONG_TOOL_MS)
    })

    // No answer is not an ending: the stall hint and Stop stay available.
    expect($workingSessionIds.get()).toContain('s-quiet')
    expect(cardShown()).toBe(false)
  })

  it('and settles it once the backend reports it over', async () => {
    vi.spyOn(document, 'hasFocus').mockReturnValue(false)
    startQuietTurn()
    const request = vi.fn(working)
    render('default', 'local', async () => undefined, request)
    await act(async () => undefined)
    request.mockImplementation(async () => ({ sessions: [] }))

    await act(async () => {
      await vi.advanceTimersByTimeAsync(LIVE_TURN_EVENT_SILENCE_MS)
    })

    expect($workingSessionIds.get()).not.toContain('s-quiet')
    expect($sessionStates.get()['rt-quiet']?.interrupted).toBeFalsy()
  })
})

describe('useBackgroundSync active-view gating', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $changeEventsAvailable.set(true)
    $cronChangeTick.set(0)
    $sessionsChangeTick.set(0)
    $sidebarShowArchived.set(false)
    vi.spyOn(globalThis.document, 'hasFocus').mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  const hideWindow = () => {
    const visibility = vi.spyOn(globalThis.document, 'visibilityState', 'get').mockReturnValue('hidden')

    return () => visibility.mockReturnValue('visible')
  }

  const changeVisibility = async () => {
    await act(async () => {
      globalThis.document.dispatchEvent(new Event('visibilitychange'))
    })
  }

  it('holds the coalesced pass while the window is hidden and catches up when shown', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    render('default', 'local', refreshSessions)
    await act(async () => undefined)
    refreshSessions.mockClear()

    const show = hideWindow()

    await act(async () => {
      $sessionsChangeTick.set(1)
    })

    await act(async () => {
      vi.advanceTimersByTime(60_000)
      await Promise.resolve()
    })

    expect(refreshSessions).not.toHaveBeenCalled()

    show()
    await changeVisibility()

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('does not stack passes when visibility flips repeatedly', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    render('default', 'local', refreshSessions)
    await act(async () => undefined)
    refreshSessions.mockClear()

    const show = hideWindow()

    await act(async () => {
      $sessionsChangeTick.set(1)
    })

    show()
    await changeVisibility()
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    await act(async () => {
      globalThis.document.dispatchEvent(new Event('visibilitychange'))
      window.dispatchEvent(new Event('focus'))
      globalThis.document.dispatchEvent(new Event('visibilitychange'))
    })
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('keeps an owed pass across an effect re-creation while hidden and defers it to the gap floor', async () => {
    const refreshSessions = vi.fn(async () => undefined)
    render('default', 'local', refreshSessions)
    await act(async () => undefined)
    refreshSessions.mockClear()

    // First tick while visible: the pass runs and sets the gap floor.
    await act(async () => {
      $sessionsChangeTick.set(1)
    })
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    const show = hideWindow()

    // In-gap tick while hidden: flagged as owed (no timer is armed while hidden).
    await act(async () => {
      $sessionsChangeTick.set(2)
    })

    // Re-create the heavy effect (the shape of an active-session switch)
    // while still hidden: the owed flag must survive - a closure timer would
    // be cancelled - and the gap floor must not reset.
    await act(async () => {
      $changeEventsAvailable.set(false)
    })
    await act(async () => {
      $changeEventsAvailable.set(true)
    })

    show()
    await changeVisibility()
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    await act(async () => {
      vi.advanceTimersByTime(30_000)
      await Promise.resolve()
    })
    expect(refreshSessions).toHaveBeenCalledTimes(2)
  })
})
