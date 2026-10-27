import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sidebarShowArchived } from '@/store/layout'
import { $changeEventsAvailable, $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $onBattery } from '@/store/power'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState
} from '@/store/session-states'
import { loadArchivedSessions } from '@/store/sidebar-archive'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

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

describe('useBackgroundSync live-status poll with no session open', () => {
  // A profile picked on top of the shared primary socket (a Webapp tab's live pick, a global remote)
  // is a view of that socket: a request that names no profile is answered for the socket's own
  // profile, so the sidebar dot would show that profile's turns instead of the selected one's.
  const sharedResearch = { profile: 'research', sharedPrimary: true } as HermesConnection

  async function polledParams(connection: HermesConnection | null) {
    $connection.set(connection)
    const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ sessions: [] }))
    render('research', 'local', async () => undefined, request as typeof requestGateway)
    await act(async () => undefined)

    return request.mock.calls.filter(([method]) => method === 'session.active_list').map(([, params]) => params)
  }

  beforeEach(() => {
    vi.useFakeTimers()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $changeEventsAvailable.set(false)
    $sessionsChangeTick.set(0)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    vi.useRealTimers()
  })

  it('names the selected profile when it rides the shared primary socket', async () => {
    expect(await polledParams(sharedResearch)).toEqual([{ profile: 'research' }])
  })

  it('asks what it asked before while a session is open (the session owner routes it)', async () => {
    $selectedStoredSessionId.set('s-open')

    expect(await polledParams(sharedResearch)).toEqual([{}])
  })

  it('names no profile on a socket of the selected profile’s own', async () => {
    expect(await polledParams({ profile: 'research' } as HermesConnection)).toEqual([{}])
    cleanup()
    expect(await polledParams(null)).toEqual([{}])
  })

  // A reconnect re-resolves the socket's own route and publishes the primary's descriptor
  // (sharedPrimary false) while the picked profile stays selected; the poll must still name it.
  it('keeps naming the picked profile after a reconnect republishes the primary descriptor', async () => {
    const win = window as unknown as Window & { __HERMES_DASHBOARD_PROFILE__?: string }
    win.__HERMES_DASHBOARD_PROFILE__ = 'default'
    seedWebappSession('served-token')
    window.history.replaceState(null, '', '/?profile=research#/')

    try {
      expect(installBrowserDesktopBridge()).toBe(true)
      const desktop = window.hermesDesktop!

      const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ sessions: [] }))
      $connection.set(await desktop.getConnection('research'))
      render('research', 'local', async () => undefined, request as typeof requestGateway)
      await act(async () => undefined)

      $connection.set(await desktop.getConnection())
      expect($connection.get()?.sharedPrimary).toBeFalsy()
      await act(async () => {
        $sessionsChangeTick.set(1)
      })
      await act(async () => undefined)

      expect(
        request.mock.calls.filter(([method]) => method === 'session.active_list').map(([, params]) => params)
      ).toEqual([{ profile: 'research' }, { profile: 'research' }])
    } finally {
      delete win.__HERMES_DASHBOARD_PROFILE__
      clearWebappSession()
      Reflect.deleteProperty(window, 'hermesDesktop')
      window.history.replaceState(null, '', '/#/')
    }
  })

  describe('on a native local gateway', () => {
    // Native local profiles ride the one host backend too (multiplex-only routing): main
    // resolves 'research' to the primary socket with sharedPrimary, and the primary's own
    // descriptor to an unflagged local one.
    const getConnection = vi.fn(async (profile?: string) =>
      profile === 'research'
        ? ({ mode: 'local', profile: 'research', sharedPrimary: true } as HermesConnection)
        : ({ mode: 'local', profile: 'default' } as HermesConnection)
    )

    beforeEach(() => {
      getConnection.mockClear()
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { getConnection } })
    })

    afterEach(() => {
      Reflect.deleteProperty(window, 'hermesDesktop')
    })

    it('keeps naming the picked profile after a reconnect republishes the primary local descriptor', async () => {
      const request = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ sessions: [] }))
      $connection.set(await getConnection('research'))
      render('research', 'local', async () => undefined, request as typeof requestGateway)
      await act(async () => undefined)

      $connection.set(await getConnection())
      await act(async () => {
        $sessionsChangeTick.set(1)
      })
      await act(async () => undefined)

      expect(
        request.mock.calls.filter(([method]) => method === 'session.active_list').map(([, params]) => params)
      ).toEqual([{ profile: 'research' }, { profile: 'research' }])
    })

    it('names no profile and asks main nothing on the selected profile’s own local descriptor', async () => {
      expect(await polledParams({ mode: 'local', profile: 'research' } as HermesConnection)).toEqual([{}])
      expect(getConnection).not.toHaveBeenCalled()
    })
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
