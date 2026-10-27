import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways, isActivePrimary, requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// A Webapp tab has one socket, and the serving gateway answers it for the profile it was launched as
// whenever a request names no profile, exactly as its REST does. A tab loaded or reloaded on
// `?profile=<name>` (or opened on the origin's saved default) therefore keeps that launch profile as its
// primary and selects <name> on top of it, the way a live switch does: every request for <name> carries
// `profile`. These tests boot the real hook against the real browser bridge.

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

type WebappWindow = Window & { __HERMES_DASHBOARD_PROFILE__?: string }

function Harness() {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

function closeSurvivor() {
  try {
    takeGatewaySurvivor()?.gateway.close()
  } catch {
    // ignore
  }
}

const originalWebSocket = globalThis.WebSocket
const mints: URL[] = []
// What the gateway reports: every installed profile on /api/profiles, the ones it serves on
// /api/status (null: a backend that does not report the set).
let installed: string[] = []
let served: null | string[] = null
const frames: { method?: string; params?: Record<string, unknown> }[] = []

async function flush() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

/** Boot a Webapp tab at `url` against a gateway launched as `launch`; resolves once boot is done. */
async function bootWebappTab(url: string, launch = 'default') {
  ;(window as WebappWindow).__HERMES_DASHBOARD_PROFILE__ = launch
  window.history.replaceState(null, '', url)
  seedWebappSession('served-token')
  expect(installBrowserDesktopBridge()).toBe(true)
  render(<Harness />)

  for (let i = 0; i < 10 && $desktopBoot.get().running; i += 1) {
    await flush()
  }

  expect($desktopBoot.get().running).toBe(false)
}

/** The params the tab's socket carried for one request to `profile`. */
async function sentFor(profile: string, method: string) {
  frames.length = 0
  void requestGatewayForProfile(profile, method, {}).catch(() => undefined)
  await flush()

  return frames.find(frame => frame.method === method)?.params
}

beforeEach(() => {
  closeSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  mints.length = 0
  frames.length = 0
  installed = ['default', 'research']
  served = null
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  vi.spyOn(FakeWebSocket.prototype, 'send').mockImplementation(function (this: FakeWebSocket, data: string) {
    frames.push(JSON.parse(data) as (typeof frames)[number])
  })
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: URL) => {
      if (url.pathname.endsWith('/api/webapp/socket-ticket')) {
        mints.push(url)

        return new Response(JSON.stringify({ ticket: `ticket-${mints.length}` }), { status: 200 })
      }

      if (url.pathname.endsWith('/api/status')) {
        return new Response(JSON.stringify({ gateway_running: true, gateway_shared_with: served }), { status: 200 })
      }

      return new Response(JSON.stringify({ profiles: installed.map(name => ({ name })) }), { status: 200 })
    })
  )
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
  closeSurvivor()
  closeSecondaryGateways()
  $connection.set(null)
  $activeGatewayProfile.set('default')
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete (window as WebappWindow).__HERMES_DASHBOARD_PROFILE__
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  clearWebappSession()
  window.localStorage.clear()
  window.history.replaceState(null, '', '/#/')
})

it('a tab loaded on ?profile=<name> asks its socket for that profile, by name', async () => {
  await bootWebappTab('/?profile=research#/')
  expect($activeGatewayProfile.get()).toBe('research')

  // The live-status poll and every other request for the tab's profile name it on the wire...
  expect(await sentFor('research', 'session.active_list')).toEqual({ profile: 'research' })
  // ...through the owner door a session's silence probe takes as well...
  frames.length = 0
  void requestGatewayForAgent('local', 'research', 'session.active_list', {}).catch(() => undefined)
  await flush()
  expect(frames.find(frame => frame.method === 'session.active_list')?.params).toEqual({ profile: 'research' })
  // ...and the gateway's own profile is the one a request without `profile` reaches.
  expect(await sentFor('default', 'session.active_list')).toEqual({})

  // One socket, minted for the gateway's own profile: the mint names no other.
  expect(FakeWebSocket.instances).toHaveLength(1)
  expect(mints.map(url => url.searchParams.get('profile'))).toEqual([null])
  expect($connection.get()).toMatchObject({ profile: 'research', sharedPrimary: true })
  expect(isActivePrimary()).toBe(true)
})

it('a bare tab on a gateway launched as a named profile names default when it asks for it', async () => {
  await bootWebappTab('/#/', 'research')

  expect($activeGatewayProfile.get()).toBe('research')
  expect(await sentFor('research', 'session.active_list')).toEqual({})
  expect(await sentFor('default', 'session.active_list')).toEqual({ profile: 'default' })
})

it('a bare tab opens on the saved default the same way, and records it in its address as a switch does', async () => {
  window.localStorage.setItem('hermes:webapp:default-profile:/', JSON.stringify({ connectionId: 'local', profile: 'research' }))
  await bootWebappTab('/#/')

  expect($activeGatewayProfile.get()).toBe('research')
  expect(await sentFor('research', 'session.active_list')).toEqual({ profile: 'research' })
  expect(mints.map(url => url.searchParams.get('profile'))).toEqual([null])
  // Files, uploads and the terminal fall back to the address's profile whenever the connection
  // descriptor names none (a reconnect publishes the tab's own socket), so it must name this one.
  expect(new URLSearchParams(window.location.search).get('profile')).toBe('research')
})

it("a tab loaded on a profile the gateway no longer has stays on the gateway's own and names it in its address", async () => {
  await bootWebappTab('/?profile=gone#/')

  expect($activeGatewayProfile.get()).toBe('default')
  expect(await sentFor('default', 'session.active_list')).toEqual({})
  // A reload lands on the profile the tab is really on, not on the stale link's.
  expect(new URLSearchParams(window.location.search).get('profile')).toBe('default')
})

it("a tab loaded on a profile its gateway does not serve stays on the gateway's own", async () => {
  installed = ['default', 'research', 'solo']
  served = ['default', 'research']
  await bootWebappTab('/?profile=solo#/')

  expect($activeGatewayProfile.get()).toBe('default')
  expect(new URLSearchParams(window.location.search).get('profile')).toBe('default')
  expect(FakeWebSocket.instances).toHaveLength(1)
})

it('picking a profile the gateway does not serve is refused with the reason and dials nothing', async () => {
  installed = ['default', 'research', 'solo']
  served = ['default', 'research']
  await bootWebappTab('/?profile=research#/')
  expect($activeGatewayProfile.get()).toBe('research')

  await expect(ensureGatewayProfile('solo')).rejects.toThrow(
    'This gateway does not serve profile "solo", so this tab cannot open it.'
  )
  expect($activeGatewayProfile.get()).toBe('research')
  expect(FakeWebSocket.instances).toHaveLength(1)
  expect(mints.map(url => url.searchParams.get('profile'))).toEqual([null])
})

it('a refused pick is final: the store does not redial the profile the gateway does not serve', async () => {
  installed = ['default', 'research', 'solo']
  served = ['default', 'research']
  await bootWebappTab('/?profile=research#/')
  const dials = vi.spyOn(window.hermesDesktop!, 'getGatewayWsUrl')

  await expect(ensureGatewayProfile('solo')).rejects.toThrow('This gateway does not serve profile "solo"')
  // Well past the redial ladder's ceiling: every scheduled redial would have run by now.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(120_000)
  })

  expect(dials.mock.calls.filter(([profile]) => profile === 'solo')).toHaveLength(1)
  expect(FakeWebSocket.instances).toHaveLength(1)
})
