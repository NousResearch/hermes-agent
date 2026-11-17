import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $desktopBoot } from '@/store/boot'
import { $connectionsRegistry, _resetConnectionsForTests } from '@/store/connections'
import { closeSecondaryGateways, isActivePrimary } from '@/store/gateway'
import { $gatewaySwitching, endGatewaySwitch } from '@/store/gateway-switch'
import { clearNotifications } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, ensureGatewayAgent, ensureGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $gatewayState,
  $sessions,
  $sessionsLoading,
  setActiveSessionId
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

// A plugin moves ONE profile to a backend it runs and back
// (host.setProfileBackend / clearProfileBackend). Main retires the old route
// and sends connection-applied naming the profile; the window must redial that
// profile in place, never blank the page, wipe the session list or hop to
// another profile.

type ConnectionAppliedDetail = { profile?: string }

let connectionApplied: null | ((detail?: ConnectionAppliedDetail) => void) = null

const primaryConn = {
  authMode: 'token' as const,
  baseUrl: 'https://vps.example.com',
  connectionId: 'primary-vps',
  profile: 'default',
  token: 't',
  wsUrl: 'wss://vps.example.com/api/ws?token=t'
}

const coderConn = {
  authMode: 'token' as const,
  baseUrl: 'https://coder.example.com',
  connectionId: 'coder-remote',
  profile: 'coder',
  token: 'c',
  wsUrl: 'wss://coder.example.com/api/ws?token=c'
}

// The sandbox launcher's address for the primary profile once it is re-routed.
const launcherConn = {
  ...primaryConn,
  baseUrl: 'http://127.0.0.1:47972',
  connectionId: undefined,
  token: 'l',
  wsUrl: 'ws://127.0.0.1:47972/api/ws?token=l'
}

function fakeDesktop() {
  return {
    getConnection: vi.fn(async (profile?: null | string): Promise<Record<string, unknown>> => {
      const key = (profile ?? '').trim()

      return !key || key === 'default' ? primaryConn : coderConn
    }),
    // Main mints a socket URL for a PROFILE's current route.
    getGatewayWsUrl: vi.fn(async (_profile?: unknown): Promise<string> => primaryConn.wsUrl),
    getBootProgress: vi.fn(async () => ({
      error: null as null | string,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      retryable: false as boolean,
      running: true as boolean,
      timestamp: Date.now()
    })),
    onBootProgress: vi.fn(() => () => undefined),
    onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn(callback => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn(() => () => undefined),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(() => () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness({
  beforeConnectionSwitch = () => undefined,
  refreshSessions = async () => undefined
}: {
  beforeConnectionSwitch?: () => void
  refreshSessions?: (shouldPublish?: () => boolean) => Promise<void>
} = {}) {
  useGatewayBoot({
    beforeConnectionSwitch,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

function dropSurvivor() {
  // Vitest keeps import.meta.hot truthy, so the boot effect's cleanup parks an
  // open gateway (the real HMR path): close it so each case boots fresh.
  try {
    takeGatewaySurvivor()?.gateway.close()
  } catch {
    // ignore
  }
}

beforeEach(() => {
  dropSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  connectionApplied = null
  clearNotifications()
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = fakeDesktop()
  $gatewayState.set('idle')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: '',
    phase: 'init',
    progress: 0,
    running: true,
    timestamp: Date.now(),
    visible: true
  })
})

afterEach(() => {
  cleanup()
  dropSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  setActiveSessionId(null)
  endGatewaySwitch()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

// Let pending microtasks (awaits) AND the queued 0ms socket open/error fire.
async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

describe('a plugin re-routes one profile', () => {
  it('a pooled profile re-routed by a plugin re-dials only its own socket and never leaves it', async () => {
    const beforeConnectionSwitch = vi.fn()

    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayProfile('coder')
    vi.useFakeTimers()
    expect($activeGatewayProfile.get()).toBe('coder')
    const primarySocket = FakeWebSocket.instances.find(socket => socket.readyState === 1)
    const socketsBefore = FakeWebSocket.instances.length
    const seen = new Set<string>()
    const offProfile = $activeGatewayProfile.subscribe(profile => void seen.add(profile))

    vi.useRealTimers()
    act(() => connectionApplied?.({ profile: 'coder' }))

    for (
      let i = 0;
      i < 100 && (FakeWebSocket.instances.length === socketsBefore || $gatewayState.get() !== 'open');
      i += 1
    ) {
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    await new Promise(resolve => setTimeout(resolve, 20))
    vi.useFakeTimers()
    offProfile()

    // Only the re-routed profile got a new socket; the primary's stayed open,
    // nothing wiped the window, and the window never showed another profile.
    expect(FakeWebSocket.instances.slice(socketsBefore)).toHaveLength(1)
    expect(primarySocket?.readyState).toBe(1)
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect($gatewaySwitching.get()).toBe(false)
    expect([...seen]).toEqual(['coder'])
    expect($activeGatewayProfile.get()).toBe('coder')
    expect(isActivePrimary()).toBe(false)
  })

  it('a profile shown through the shared primary socket moves to its new route at once', async () => {
    const beforeConnectionSwitch = vi.fn()
    const desktop = fakeDesktop()
    // The one local host backend serves every profile: `work` rides the primary
    // socket (main answers its route with sharedPrimary) until a plugin takes it.
    const sharedWork = { ...primaryConn, profile: 'work', sharedPrimary: true }
    const launcherWork = { ...launcherConn, profile: 'work' }
    let sandboxed = false

    desktop.getConnection.mockImplementation(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      if (!key || key === 'default') {
        return primaryConn
      }

      return sandboxed ? launcherWork : sharedWork
    })
    desktop.getGatewayWsUrl.mockImplementation(async () => primaryConn.wsUrl)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayProfile('work')
    vi.useFakeTimers()
    expect($activeGatewayProfile.get()).toBe('work')
    // Riding the primary: no socket of its own.
    expect(isActivePrimary()).toBe(true)
    const primarySocket = FakeWebSocket.instances.find(socket => socket.readyState === 1)
    const socketsBefore = FakeWebSocket.instances.length

    // The sandbox launcher took `work` over.
    sandboxed = true
    vi.useRealTimers()
    act(() => connectionApplied?.({ profile: 'work' }))

    for (let i = 0; i < 100 && isActivePrimary(); i += 1) {
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    await new Promise(resolve => setTimeout(resolve, 20))
    vi.useFakeTimers()

    // `work` got its own socket and the window's route is on it: its next
    // session.create goes to the launcher, not to the host backend that no
    // longer serves it. The primary socket stays open; nothing was wiped.
    expect(isActivePrimary()).toBe(false)
    expect(FakeWebSocket.instances.slice(socketsBefore).filter(socket => socket.readyState === 1)).toHaveLength(1)
    expect(primarySocket?.readyState).toBe(1)
    expect($activeGatewayProfile.get()).toBe('work')
    expect($connection.get()?.baseUrl).toBe(launcherConn.baseUrl)
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('a sandboxed profile switched off moves back onto the shared primary and the window stays on it', async () => {
    // Window layout of the tryit runs: `default` is the window's primary, `work`
    // is selected and runs in the sandbox on its own socket (the launcher).
    // Switching it off hands `work` back to the one local host backend, which
    // serves it through the primary socket again (main answers sharedPrimary).
    const beforeConnectionSwitch = vi.fn()
    const desktop = fakeDesktop()
    const sharedWork = { ...primaryConn, profile: 'work', sharedPrimary: true }
    const launcherWork = { ...launcherConn, profile: 'work' }
    let sandboxed = true

    desktop.getConnection.mockImplementation(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      if (!key || key === 'default') {
        return primaryConn
      }

      return sandboxed ? launcherWork : sharedWork
    })
    desktop.getGatewayWsUrl.mockImplementation(async (profile?: unknown) =>
      profile === 'work' && sandboxed ? launcherWork.wsUrl : primaryConn.wsUrl
    )
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayProfile('work')
    vi.useFakeTimers()
    expect($activeGatewayProfile.get()).toBe('work')
    expect(isActivePrimary()).toBe(false)
    const primarySocket = FakeWebSocket.instances.find(s => s.url === primaryConn.wsUrl && s.readyState === 1)
    const launcherSocket = FakeWebSocket.instances.find(s => s.url === launcherWork.wsUrl && s.readyState === 1)
    expect(launcherSocket).toBeDefined()
    // Changes only: the window's profile must not move at all, not even for a frame.
    const profiles: string[] = []
    const switching: boolean[] = []
    const offProfile = $activeGatewayProfile.listen(value => void profiles.push(value))
    const offSwitching = $gatewaySwitching.listen(value => void switching.push(value))

    // The plugin switched `work` off: main cleared its override.
    sandboxed = false
    vi.useRealTimers()
    act(() => connectionApplied?.({ profile: 'work' }))

    for (let i = 0; i < 100 && !isActivePrimary(); i += 1) {
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    await new Promise(resolve => setTimeout(resolve, 20))
    vi.useFakeTimers()
    offProfile()
    offSwitching()

    // `work` rides the primary socket again, the launcher socket is gone, and
    // the window still shows `work` with its host descriptor (not the stopped
    // launcher's address). No connection-wide switch ran.
    expect(isActivePrimary()).toBe(true)
    expect(launcherSocket?.readyState).toBe(FakeWebSocket.CLOSED)
    expect(primarySocket?.readyState).toBe(1)
    expect(profiles).toEqual([])
    expect($activeGatewayProfile.get()).toBe('work')
    expect($connection.get()).toMatchObject({ baseUrl: primaryConn.baseUrl, profile: 'work', sharedPrimary: true })
    expect($apiRequestScope.get()).toEqual({ connectionId: primaryConn.connectionId, profile: 'work' })
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect(switching).toEqual([])
  })

  it('switched off after the sandbox already stopped, the window and its requests stay on the profile', async () => {
    // The plugin stops the sandbox before main clears the override, so `work`'s
    // launcher socket drops and every redial fails ("Stopping sandbox…") until
    // connection-applied arrives. The plugin's page and the session list then
    // ask the host backend for `work`; asking for `default` showed default's
    // sandbox settings under a rail that still had `work` selected.
    const desktop = fakeDesktop()
    const sharedWork = { ...primaryConn, profile: 'work', sharedPrimary: true }
    const launcherWork = { ...launcherConn, profile: 'work' }
    let sandboxed = true

    desktop.getConnection.mockImplementation(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      if (!key || key === 'default') {
        return primaryConn
      }

      return sandboxed ? launcherWork : sharedWork
    })
    desktop.getGatewayWsUrl.mockImplementation(async (profile?: unknown) =>
      profile === 'work' && sandboxed ? launcherWork.wsUrl : primaryConn.wsUrl
    )
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayProfile('work')
    vi.useFakeTimers()
    const launcherSocket = FakeWebSocket.instances.find(s => s.url === launcherWork.wsUrl && s.readyState === 1)
    expect(launcherSocket).toBeDefined()

    // The sandbox stops: the launcher socket drops and its redials fail.
    FakeWebSocket.mode = 'fail'
    launcherSocket?.drop()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000)
    })
    expect(FakeWebSocket.instances.filter(s => s.url === launcherWork.wsUrl).length).toBeGreaterThan(1)
    expect($activeGatewayProfile.get()).toBe('work')

    // Main cleared the override.
    sandboxed = false
    FakeWebSocket.mode = 'open'
    vi.useRealTimers()
    act(() => connectionApplied?.({ profile: 'work' }))

    for (let i = 0; i < 100 && !isActivePrimary(); i += 1) {
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    await new Promise(resolve => setTimeout(resolve, 20))
    vi.useFakeTimers()

    expect(isActivePrimary()).toBe(true)
    expect($activeGatewayProfile.get()).toBe('work')
    expect($connection.get()).toMatchObject({ baseUrl: primaryConn.baseUrl, profile: 'work', sharedPrimary: true })
    expect($apiRequestScope.get()).toEqual({ connectionId: primaryConn.connectionId, profile: 'work' })
  })

  it('a primary-wide apply still adopts the primary profile', async () => {
    render(<Harness />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayProfile('coder')
    vi.useFakeTimers()
    expect($activeGatewayProfile.get()).toBe('coder')

    act(() => connectionApplied?.())
    await flushAsync()

    expect($activeGatewayProfile.get()).toBe('default')
    expect(isActivePrimary()).toBe(true)
  })

  it('the window’s PRIMARY profile re-dials its socket in place: no switch, no wipe, no fresh draft', async () => {
    const beforeConnectionSwitch = vi.fn()
    const refreshSessions = vi.fn(async () => undefined)

    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} refreshSessions={refreshSessions} />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect(isActivePrimary()).toBe(true)

    const oldPrimary = FakeWebSocket.instances.find(socket => socket.readyState === 1)
    const socketsBefore = FakeWebSocket.instances.length
    // Changes only (listen, not subscribe): the barrier, the session-list
    // skeleton and the window's profile must not move at all.
    const switching: boolean[] = []
    const loading: boolean[] = []
    const profiles: string[] = []
    const offSwitching = $gatewaySwitching.listen(value => void switching.push(value))
    const offLoading = $sessionsLoading.listen(value => void loading.push(value))
    const offProfile = $activeGatewayProfile.listen(value => void profiles.push(value))
    refreshSessions.mockClear()

    // Main switched `default` (the primary) to the sandbox launcher: its route
    // and the WebSocket URL main mints for it both name the launcher now.
    const routeOf = (profile?: null | string) =>
      !(profile ?? '').trim() || profile === 'default' ? launcherConn : coderConn

    desktop.getConnection.mockImplementation(async profile => routeOf(profile))
    desktop.getGatewayWsUrl.mockImplementation(async profile =>
      typeof profile === 'string' || profile == null ? routeOf(profile).wsUrl : primaryConn.wsUrl
    )

    vi.useRealTimers()
    act(() => connectionApplied?.({ profile: 'default' }))

    for (
      let i = 0;
      i < 100 && !FakeWebSocket.instances.some(s => s.url === launcherConn.wsUrl && s.readyState === 1);
      i += 1
    ) {
      await new Promise(resolve => setTimeout(resolve, 10))
    }

    await new Promise(resolve => setTimeout(resolve, 20))
    vi.useFakeTimers()
    offSwitching()
    offLoading()
    offProfile()

    // One new socket, on the launcher; the old primary socket is closed.
    expect(FakeWebSocket.instances.slice(socketsBefore).map(socket => socket.url)).toEqual([launcherConn.wsUrl])
    expect(oldPrimary?.readyState).toBe(FakeWebSocket.CLOSED)
    expect($gatewayState.get()).toBe('open')
    expect($connection.get()?.baseUrl).toBe(launcherConn.baseUrl)
    // No connection-wide switch: no fresh draft, no session-list skeleton, the
    // window never showed another profile; the list is refreshed in place.
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect(switching).toEqual([])
    expect(loading).not.toContain(true)
    expect(profiles.filter(profile => profile !== 'default')).toEqual([])
    expect($activeGatewayProfile.get()).toBe('default')
    expect(isActivePrimary()).toBe(true)
    expect(refreshSessions).toHaveBeenCalled()
  })
})

// connection-applied { profile } names a profile of THIS computer. A registered
// gateway (another machine) can serve a profile of the same name: a window on
// it holds no socket of the local profile, so nothing of it may change.
describe('a local profile moves while the window is on another gateway', () => {
  // The registered gateway's own `work`, as main answers a window opened on it
  // (registry-scoped: both identity fields).
  const teamWork = {
    authMode: 'token' as const,
    baseUrl: 'https://team.example.com',
    connectionId: 'team-gw',
    mode: 'remote' as const,
    profile: 'work',
    registryScoped: true,
    token: 'w',
    wsUrl: 'wss://team.example.com/api/ws?token=w'
  }

  const teamDefault = { ...teamWork, profile: 'default' }

  // This computer's `work`: Hermes's own backend, then the plugin's launcher.
  const hostWork = {
    ...launcherConn,
    baseUrl: 'http://127.0.0.1:9120',
    profile: 'work',
    wsUrl: 'ws://127.0.0.1:9120/api/ws?token=h'
  }

  const launcherWork = { ...launcherConn, profile: 'work' }

  // `local.sandboxed`: where this computer's profiles run now.
  function teamDesktop(primary: Record<string, unknown>, local = { sandboxed: true }) {
    return {
      ...fakeDesktop(),
      // Main answers a profile-less request from the window's own route; an
      // explicit profile is this computer's (the legacy door).
      getConnection: vi.fn(async (profile?: null | string): Promise<Record<string, unknown>> => {
        const key = (profile ?? '').trim()

        return key ? { ...(local.sandboxed ? launcherWork : hostWork), profile: key } : primary
      }),
      getConnectionFor: vi.fn(
        async ({ profile }: { connectionId: string; profile: string }): Promise<Record<string, unknown>> => ({
          ...primary,
          profile
        })
      ),
      getGatewayWsUrl: vi.fn(async () => (local.sandboxed ? launcherWork.wsUrl : hostWork.wsUrl)),
      getGatewayWsUrlFor: vi.fn(async (): Promise<string> => teamWork.wsUrl),
      profile: { get: vi.fn(async () => ({ profile: String(primary.profile) })) }
    }
  }

  // What a connection-wide switch does to a window: every piece is watched.
  function watchWindow() {
    const switching: boolean[] = []
    const loading: boolean[] = []
    const profiles: string[] = []

    const offs = [
      $gatewaySwitching.listen(value => void switching.push(value)),
      $sessionsLoading.listen(value => void loading.push(value)),
      $activeGatewayProfile.listen(value => void profiles.push(value))
    ]

    return { loading, profiles, stop: () => offs.forEach(off => off()), switching }
  }

  afterEach(() => $sessions.set([]))

  async function applied(profile: string) {
    vi.useRealTimers()
    act(() => connectionApplied?.({ profile }))
    await new Promise(resolve => setTimeout(resolve, 150))
    vi.useFakeTimers()
  }

  it.each(['work', 'default'])(
    'a window on that gateway’s own "%s" keeps its chat, session list, terminals and socket',
    async name => {
      // Closing the terminals and replacing the open chat with a draft is the
      // window's beforeConnectionSwitch (wiring.tsx).
      const beforeConnectionSwitch = vi.fn()
      const refreshSessions = vi.fn(async () => undefined)

      const desktop = teamDesktop({ ...teamWork, profile: name })

      ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
      $activeGatewayProfile.set(name)
      render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} refreshSessions={refreshSessions} />)
      await flushAsync()
      expect($gatewayState.get()).toBe('open')
      expect($connection.get()).toMatchObject({ connectionId: 'team-gw', profile: name, registryScoped: true })
      expect($activeGatewayProfile.get()).toBe(name)

      const rows = [{ id: 'open-chat' }] as never
      $sessions.set(rows)
      setActiveSessionId('open-chat')
      const socket = FakeWebSocket.instances.find(candidate => candidate.readyState === 1)
      const socketsBefore = FakeWebSocket.instances.length
      const dials = () => desktop.getConnection.mock.calls.length + desktop.getConnectionFor.mock.calls.length
      const dialsBefore = dials()
      const seen = watchWindow()
      beforeConnectionSwitch.mockClear()
      refreshSessions.mockClear()

      // A plugin moved THIS computer's profile of that name (another window flipped its switch).
      await applied(name)
      seen.stop()

      expect(beforeConnectionSwitch).not.toHaveBeenCalled()
      expect(seen.switching).toEqual([])
      expect(seen.loading).toEqual([])
      expect(seen.profiles).toEqual([])
      expect($sessions.get()).toBe(rows)
      expect($activeSessionId.get()).toBe('open-chat')
      expect(refreshSessions).not.toHaveBeenCalled()
      // The unchanged gateway is not asked for again and its socket stays.
      expect(dials()).toBe(dialsBefore)
      expect(FakeWebSocket.instances).toHaveLength(socketsBefore)
      expect(socket?.readyState).toBe(1)
      expect($gatewayState.get()).toBe('open')
      expect($connection.get()).toMatchObject({ baseUrl: teamWork.baseUrl, connectionId: 'team-gw', profile: name })
    }
  )

  it('a same-named profile shown through that gateway’s socket is not dialed on this computer', async () => {
    const beforeConnectionSwitch = vi.fn()
    const desktop = teamDesktop(teamDefault)

    // One remote host serves every profile of the gateway on the window's socket.
    desktop.getConnectionFor.mockImplementation(async ({ profile }) => ({
      ...teamDefault,
      profile,
      sharedRemote: profile !== 'default'
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    vi.useRealTimers()
    await ensureGatewayAgent('team-gw', 'work')
    vi.useFakeTimers()
    expect($activeGatewayProfile.get()).toBe('work')
    expect(isActivePrimary()).toBe(true)
    expect($apiRequestScope.get()).toEqual({ connectionId: 'team-gw', profile: 'work' })

    const socketsBefore = FakeWebSocket.instances.length
    const localDials = () => desktop.getConnection.mock.calls.filter(([profile]) => profile === 'work').length
    const localDialsBefore = localDials()
    const seen = watchWindow()

    await applied('work')
    seen.stop()

    // Nothing asked main for this computer's `work`, and the window is still
    // on the gateway's: same socket, same connection, same request scope.
    expect(localDials()).toBe(localDialsBefore)
    expect(FakeWebSocket.instances).toHaveLength(socketsBefore)
    expect(isActivePrimary()).toBe(true)
    expect(seen.profiles).toEqual([])
    expect(seen.switching).toEqual([])
    expect($activeGatewayProfile.get()).toBe('work')
    expect($connection.get()).toMatchObject({ baseUrl: teamWork.baseUrl, connectionId: 'team-gw', profile: 'work' })
    expect($apiRequestScope.get()).toEqual({ connectionId: 'team-gw', profile: 'work' })
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
  })

  it('a window still starting on that gateway goes on starting', async () => {
    const beforeConnectionSwitch = vi.fn()
    const desktop = teamDesktop(teamWork)
    let mint: (url: string) => void = () => undefined

    // Main answered what the window connects to; its socket address is still being minted.
    desktop.getGatewayWsUrlFor.mockImplementation(() => new Promise<string>(resolve => void (mint = resolve)))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    $activeGatewayProfile.set('work')
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledTimes(1)
    const dialsBefore = desktop.getConnection.mock.calls.length
    const seen = watchWindow()

    await applied('work')

    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    expect(seen.switching).toEqual([])
    expect(desktop.getConnection.mock.calls).toHaveLength(dialsBefore)
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledTimes(1)

    await act(async () => {
      mint(teamWork.wsUrl)
      await vi.advanceTimersByTimeAsync(0)
    })
    await flushAsync()
    seen.stop()

    expect(FakeWebSocket.instances.map(socket => socket.url)).toEqual([teamWork.wsUrl])
    expect($gatewayState.get()).toBe('open')
    expect($connection.get()).toMatchObject({ connectionId: 'team-gw', profile: 'work' })
    expect(seen.switching).toEqual([])
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
  })

  it('the socket such a window dialed for the local profile itself is still re-dialed', async () => {
    const beforeConnectionSwitch = vi.fn()
    const local = { sandboxed: false }

    const desktop = teamDesktop(teamDefault, local)

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    // This computer's `work`, picked in the window: its own (legacy) socket.
    vi.useRealTimers()
    await ensureGatewayProfile('work')
    vi.useFakeTimers()
    expect(isActivePrimary()).toBe(false)
    const teamSocket = FakeWebSocket.instances.find(socket => socket.url === teamWork.wsUrl)
    const hostSocket = FakeWebSocket.instances.find(socket => socket.url === hostWork.wsUrl)
    expect(hostSocket?.readyState).toBe(1)
    const socketsBefore = FakeWebSocket.instances.length
    const seen = watchWindow()

    local.sandboxed = true
    await applied('work')
    seen.stop()

    expect(FakeWebSocket.instances.slice(socketsBefore).map(socket => socket.url)).toEqual([launcherWork.wsUrl])
    expect(hostSocket?.readyState).toBe(FakeWebSocket.CLOSED)
    expect(teamSocket?.readyState).toBe(1)
    expect($connection.get()?.baseUrl).toBe(launcherWork.baseUrl)
    expect($activeGatewayProfile.get()).toBe('work')
    expect(seen.profiles).toEqual([])
    expect(seen.switching).toEqual([])
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
  })

  it('a window opened on "This device" through the gateway list is on the local profile and is re-dialed', async () => {
    // Registry-scoped too, but its connection is this computer.
    const localScoped = { ...hostWork, connectionId: 'local', registryScoped: true }
    const desktop = teamDesktop(localScoped)

    desktop.getGatewayWsUrlFor.mockImplementation(async () => hostWork.wsUrl)
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    $activeGatewayProfile.set('work')
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const dials = () => desktop.getConnection.mock.calls.length + desktop.getConnectionFor.mock.calls.length
    const dialsBefore = dials()

    await applied('work')
    await flushAsync()

    expect(dials()).toBeGreaterThan(dialsBefore)
    expect($gatewayState.get()).toBe('open')
  })

  it.each(['default', 'work'])(
    'a gateway’s same-named profile shown on its own socket stays in front (the window’s own profile is "%s")',
    async own => {
      const beforeConnectionSwitch = vi.fn()
      const desktop = { ...fakeDesktop(), getConnectionFor: vi.fn(), getGatewayWsUrlFor: vi.fn() }
      const ownConn = { ...primaryConn, profile: own }

      // The window itself is on this computer; the gateway's `work` is a second socket.
      desktop.getConnection.mockImplementation(async () => ownConn)
      desktop.getConnectionFor.mockImplementation(async () => teamWork)
      desktop.getGatewayWsUrlFor.mockImplementation(async () => teamWork.wsUrl)
      desktop.profile.get.mockImplementation(async () => ({ profile: own }))
      ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
      $activeGatewayProfile.set(own)
      render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
      await flushAsync()
      vi.useRealTimers()
      await ensureGatewayAgent('team-gw', 'work')
      vi.useFakeTimers()
      const teamSocket = FakeWebSocket.instances.find(socket => socket.url === teamWork.wsUrl)
      expect(teamSocket?.readyState).toBe(1)
      expect(isActivePrimary()).toBe(false)
      expect($connection.get()).toMatchObject({ connectionId: 'team-gw', profile: 'work' })
      const seen = watchWindow()

      await applied('work')
      seen.stop()

      expect(teamSocket?.readyState).toBe(1)
      expect(isActivePrimary()).toBe(false)
      expect($activeGatewayProfile.get()).toBe('work')
      expect($connection.get()).toMatchObject({ baseUrl: teamWork.baseUrl, connectionId: 'team-gw', profile: 'work' })
      expect($apiRequestScope.get()).toEqual({ connectionId: 'team-gw', profile: 'work' })
      expect(seen.profiles).toEqual([])
      expect(seen.switching).toEqual([])
      expect(beforeConnectionSwitch).not.toHaveBeenCalled()
    }
  )

  it('a window that has not been told its gateway yet still starts on it', async () => {
    const desktop = teamDesktop(teamWork)
    let answer: (connection: Record<string, unknown>) => void = () => undefined

    desktop.getConnection.mockImplementationOnce(() => new Promise(resolve => void (answer = resolve)))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    $activeGatewayProfile.set('work')
    render(<Harness />)
    await flushAsync()
    expect(FakeWebSocket.instances).toHaveLength(0)

    await applied('work')
    await act(async () => {
      answer(teamWork)
      await vi.advanceTimersByTimeAsync(0)
    })
    await flushAsync()

    expect(FakeWebSocket.instances.filter(socket => socket.readyState === 1).map(socket => socket.url)).toEqual([
      teamWork.wsUrl
    ])
    expect($gatewayState.get()).toBe('open')
    expect($connection.get()).toMatchObject({ connectionId: 'team-gw', profile: 'work' })
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('a window on another profile of this computer is left alone', async () => {
    const beforeConnectionSwitch = vi.fn()

    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    expect($activeGatewayProfile.get()).toBe('default')
    const socket = FakeWebSocket.instances.find(candidate => candidate.readyState === 1)
    const socketsBefore = FakeWebSocket.instances.length
    const dialsBefore = desktop.getConnection.mock.calls.length
    const seen = watchWindow()

    await applied('work')
    seen.stop()

    expect(desktop.getConnection.mock.calls).toHaveLength(dialsBefore)
    expect(FakeWebSocket.instances).toHaveLength(socketsBefore)
    expect(socket?.readyState).toBe(1)
    expect(seen.switching).toEqual([])
    expect(seen.loading).toEqual([])
    expect(seen.profiles).toEqual([])
    expect(beforeConnectionSwitch).not.toHaveBeenCalled()
  })
})
