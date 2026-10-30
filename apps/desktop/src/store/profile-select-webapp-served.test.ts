import { afterEach, describe, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { UnservedProfileError } from '@/lib/browser-served-profiles'
import { $gateway, activeGatewayProfileKey, setPrimaryGateway } from '@/store/gateway'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

// Every socket the store builds besides the tab's own (the primary each test installs).
const sockets: Array<{ connect: ReturnType<typeof vi.fn> }> = []

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    request = vi.fn(async () => ({}))
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})

    constructor() {
      sockets.push(this)
    }
  }
}))

type ServedWindow = Window & { __HERMES_DASHBOARD_PROFILE__?: string; hermesDesktop?: Window['hermesDesktop'] }

const win = window as unknown as ServedWindow

// One multiplexer: `/api/profiles` lists every installed profile, `/api/status` the ones it serves.
// `holdProfiles` leaves every `/api/profiles` read unanswered until its own deadline aborts it.
interface Gateway {
  holdProfiles?: boolean
  profiles: string[]
  served: string[]
}

function stubGateway(gateway: Gateway) {
  const fetchMock = vi.fn(async (url: URL, init?: RequestInit) => {
    if (url.pathname.endsWith('/api/status')) {
      return new Response(JSON.stringify({ gateway_running: true, gateway_shared_with: gateway.served }), {
        status: 200
      })
    }

    if (url.pathname.endsWith('/api/webapp/socket-ticket')) {
      return new Response(JSON.stringify({ ticket: 'one-use' }), { status: 200 })
    }

    if (gateway.holdProfiles) {
      return new Promise<Response>((_resolve, reject) =>
        init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
      )
    }

    return new Response(JSON.stringify({ profiles: gateway.profiles.map(name => ({ name })) }), { status: 200 })
  })

  vi.stubGlobal('fetch', fetchMock)

  return fetchMock
}

const callsTo = (fetchMock: ReturnType<typeof stubGateway>, path: string) =>
  fetchMock.mock.calls.filter(([url]) => url.pathname.endsWith(path)).length

// The tab's bridge, its served set already read once (as the profile rail's inventory read does),
// and the tab's own socket open as the primary.
async function openTab(gateway: Gateway) {
  const fetchMock = stubGateway(gateway)
  win.__HERMES_DASHBOARD_PROFILE__ = 'default'
  seedWebappSession('served-token')
  expect(installBrowserDesktopBridge()).toBe(true)
  await win.hermesDesktop!.getProfileRoutes?.(gateway.profiles)

  const primary = { connectionState: 'open', request: vi.fn(async () => ({})) }
  setPrimaryGateway(primary as never, 'default')

  return { fetchMock, primary }
}

afterEach(() => {
  delete win.__HERMES_DASHBOARD_PROFILE__
  clearWebappSession()
  Reflect.deleteProperty(win, 'hermesDesktop')
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('selecting a profile in a Webapp tab', () => {
  it('rides the tab socket for a profile created since the last inventory read', async () => {
    const gateway: Gateway = { profiles: ['default', 'solo'], served: ['default'] }
    const { primary } = await openTab(gateway)
    const built = sockets.length

    // A picker creates the profile and selects it at once: the multiplexer adopted it before the
    // create returned, but the tab's cached set predates it.
    gateway.profiles = [...gateway.profiles, 'fresh']
    gateway.served = ['default', 'fresh']
    await ensureGatewayProfile('fresh')

    expect(sockets).toHaveLength(built)
    expect($gateway.get()).toBe(primary)
    expect(activeGatewayProfileKey()).toBe('default')
    expect($activeGatewayProfile.get()).toBe('fresh')
    expect($connection.get()).toMatchObject({ profile: 'fresh', sharedPrimary: true })
  })

  it('still refuses and parks an unserved profile at once when the inventory read stalls', async () => {
    vi.useFakeTimers()
    const gateway: Gateway = { profiles: ['default', 'solo'], served: ['default'] }
    const { fetchMock } = await openTab(gateway)
    const built = sockets.length

    gateway.holdProfiles = true
    let outcome: unknown = 'pending'
    void ensureGatewayProfile('solo').then(
      () => (outcome = 'activated'),
      (error: unknown) => (outcome = error)
    )

    // The pick waits for the inventory at most 5 s, then routes on the set it has.
    await vi.advanceTimersByTimeAsync(4_900)
    expect(outcome).toBe('pending')
    await vi.advanceTimersByTimeAsync(100)
    expect(outcome).toBeInstanceOf(UnservedProfileError)

    // Parked: nothing redials it, so no ticket is minted and no socket ever connects.
    const reads = callsTo(fetchMock, '/api/profiles')
    await vi.advanceTimersByTimeAsync(120_000)
    expect(callsTo(fetchMock, '/api/profiles')).toBe(reads)
    expect(callsTo(fetchMock, '/api/webapp/socket-ticket')).toBe(0)
    expect(sockets.slice(built).every(socket => socket.connect.mock.calls.length === 0)).toBe(true)
  })
})
