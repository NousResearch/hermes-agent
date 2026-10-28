import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { clearWebappSession, seedWebappSession } from '@/test/webapp-session'

// A live profile switch in a Webapp tab is recorded in THAT tab's URL, so a
// reload lands on the profile the user is in, and a generic New session
// follows it. It is not the origin's saved default (only "Set as default"
// writes that), no other tab hears about it, and the tab's own socket — the
// one boot and every reconnect dial — stays the one its gateway answers for
// the profile it runs as; the URL's profile is selected on top of it.

const ensureGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const ensureGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => true)
const activeGatewayConnectionId = vi.fn<() => null | string>(() => null)

vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>({ id: 'live-socket' }),
  activeGatewayConnectionId,
  activeGatewayProfileKey: () => ensureGatewayForProfile.mock.lastCall?.[0] ?? $activeGatewayProfile.get(),
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))

const { $activeGatewayProfile, selectProfile } = await import('./profile')
const { installBrowserDesktopBridge } = await import('@/lib/browser-desktop-bridge')

const bootstrap = window as Window & { __HERMES_BASE_PATH__?: string }
const savedDefault = { connectionId: 'local', profile: 'research' }
const urlProfile = () => new URLSearchParams(window.location.search).get('profile')

beforeEach(async () => {
  bootstrap.__HERMES_BASE_PATH__ = '/hermes'
  seedWebappSession('served-token')
  window.history.replaceState(null, '', '/hermes?profile=arrived#/')
  expect(installBrowserDesktopBridge()).toBe(true)
  await window.hermesDesktop.profile.setDefault(savedDefault)
  $activeGatewayProfile.set('arrived')
})

afterEach(() => {
  Reflect.deleteProperty(window, 'hermesDesktop')
  delete bootstrap.__HERMES_BASE_PATH__
  clearWebappSession()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  window.localStorage.clear()
  window.history.replaceState(null, '', '/#/')
  vi.clearAllMocks()
})

// A tab that booted on the saved default dials it as the registry `local`
// route; it is still this tab's only source.
it.each([
  { boot: 'its URL', source: null },
  { boot: 'the saved default route', source: 'local' }
])('records a live switch in the tab that booted on $boot, and only there', async ({ source }) => {
  activeGatewayConnectionId.mockReturnValue(source)
  const defaultChanged = vi.fn()
  const unsubscribe = window.hermesDesktop.profile.onDefaultChanged(defaultChanged)

  selectProfile('ops')
  await vi.waitFor(() => expect(urlProfile()).toBe('ops'))

  // Default is written out, not dropped: a bare URL would boot the saved default.
  selectProfile('default')
  await vi.waitFor(() => expect(urlProfile()).toBe('default'))

  expect(window.location.pathname).toBe('/hermes')
  expect(window.location.hash).toBe('#/')
  await expect(window.hermesDesktop.profile.getDefault()).resolves.toEqual(savedDefault)
  expect(defaultChanged).not.toHaveBeenCalled()
  const own = await window.hermesDesktop.getConnection()
  expect(own.profile).toBeUndefined()
  expect(own.sharedPrimary).toBe(false)
  unsubscribe()
})
