import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getLatestSessionMessages, type SessionInfo } from '@/hermes'
import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { createClientSessionState } from '@/lib/chat-runtime'
import * as gateways from '@/store/gateway'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { _resetSessionOwnerHintsForTests, setConnection, setSessions } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

// Every socket the store builds besides the tab's own.
const sockets: unknown[] = []

vi.mock('@/app/contrib/hooks/use-background-sync', () => ({ resetLiveRuntimeTracking: vi.fn() }))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'work-chat' })),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    close = vi.fn()
    onEvent = () => () => undefined
    onState = () => () => undefined
    request = vi.fn(async () => ({}))

    constructor() {
      sockets.push(this)
    }
  }
}))

type ServedWindow = Window & { __HERMES_DASHBOARD_PROFILE__?: string; hermesDesktop?: Window['hermesDesktop'] }

const win = window as unknown as ServedWindow

const workRow = { id: 'work-chat', profile: 'work', title: 'work chat', message_count: 2 }

// One multiplexer serving `default` (the tab's own profile) and `work`.
function stubGateway() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: URL) => {
      const path = url.pathname

      const body = path.endsWith('/api/status')
        ? { gateway_running: true, gateway_shared_with: ['default', 'work'] }
        : path.endsWith('/api/profiles')
          ? { profiles: [{ name: 'default' }, { name: 'work' }] }
          : path.endsWith('/api/profiles/sessions')
            ? { sessions: [workRow], total: 1 }
            : path.endsWith('/api/sessions/work-chat')
              ? workRow
              : {}

      return new Response(JSON.stringify(body), { status: 200 })
    })
  )
}

type Resume = ReturnType<typeof useSessionActions>['resumeSession']

function Harness({ onReady }: { onReady: (resume: Resume) => void }) {
  const states = useRef(new Map<string, ClientSessionState>())
  const activeSessionIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)
  const creatingSessionRef = useRef(false)
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>())
  const selectedStoredSessionIdRef = useRef<string | null>(null)

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef,
    busyRef,
    creatingSessionRef,
    ensureSessionState: () => createClientSessionState(null),
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: vi.fn(async () => ({}) as never),
    resetViewSync: vi.fn(),
    routedSessionId: null,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef,
    sessionStateByRuntimeIdRef: states,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (id, updater, storedId) => {
      const next = updater(states.current.get(id) ?? createClientSessionState(storedId ?? null))
      states.current.set(id, next)
      publishSessionState(id, next)

      return next
    }
  })

  useEffect(() => onReady(actions.resumeSession), [actions.resumeSession, onReady])

  return null
}

describe('a Webapp tab resuming a row its sidebar listed', () => {
  beforeEach(() => {
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    $showAllProfiles.set(false)
    $activeGatewayProfile.set('default')
    gateways.configureGatewayRegistry({ onEvent: vi.fn(), onActiveRouteChanged: p => $activeGatewayProfile.set(p) })
  })

  afterEach(() => {
    cleanup()
    gateways.closeSecondaryGateways()
    gateways.setPrimaryGateway(null)
    setConnection(null)
    setSessions([])
    clearAllSessionStates()
    _resetSessionOwnerHintsForTests()
    delete win.__HERMES_DASHBOARD_PROFILE__
    clearWebappSession()
    Reflect.deleteProperty(win, 'hermesDesktop')
    document.documentElement.removeAttribute('data-hermes-desktop-host')
    sockets.length = 0
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("opens that profile's session on the tab's own socket, routed by the row's gateway", async () => {
    stubGateway()
    win.__HERMES_DASHBOARD_PROFILE__ = 'default'
    seedWebappSession('served-token')
    expect(installBrowserDesktopBridge()).toBe(true)
    const desktop = win.hermesDesktop!
    await desktop.getProfileRoutes?.(['default', 'work'])

    // The tab's socket, published the way boot does: the tab's connection on its own profile.
    const primary = { connectionState: 'open', request: vi.fn(async () => ({})) }
    const tab = await desktop.getConnection!()
    gateways.setPrimaryGateway(primary as never, 'default')
    gateways.setPrimaryGatewayConnection(tab)
    setConnection(tab)

    // The sidebar's listing, read through the tab's REST door.
    const listed = await desktop.api<{ sessions: SessionInfo[] }>({ path: '/api/profiles/sessions?limit=40' })
    expect(listed.sessions[0]).toMatchObject({ connection_id: 'local', profile: 'work' })
    setSessions(listed.sessions)

    let resume!: Resume
    render(
      <Harness
        onReady={ready => {
          resume = ready
        }}
      />
    )
    await act(() => resume('work-chat', true))

    expect(primary.request).toHaveBeenCalledWith(
      'session.resume',
      expect.objectContaining({ profile: 'work', session_id: 'work-chat' })
    )
    expect(getLatestSessionMessages).toHaveBeenCalledWith('work-chat', { connectionId: 'local', profile: 'work' })
    expect(sockets).toHaveLength(0)
    expect($activeGatewayProfile.get()).toBe('work')

    // A tagged row leaves no remembered owner behind, so a later open without the listing (a
    // reload, a pasted link) resolves the owner from the row's own detail read, tagged the same.
    cleanup()
    setSessions([])
    primary.request.mockClear()
    vi.mocked(getLatestSessionMessages).mockClear()
    $activeGatewayProfile.set('default')
    render(
      <Harness
        onReady={ready => {
          resume = ready
        }}
      />
    )
    await act(() => resume('work-chat', true))

    expect(primary.request).toHaveBeenCalledWith(
      'session.resume',
      expect.objectContaining({ profile: 'work', session_id: 'work-chat' })
    )
    expect(getLatestSessionMessages).toHaveBeenCalledWith('work-chat', { connectionId: 'local', profile: 'work' })
    expect(sockets).toHaveLength(0)
  })
})
