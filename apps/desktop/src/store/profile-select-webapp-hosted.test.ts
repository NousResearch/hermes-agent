import { afterEach, expect, it, vi } from 'vitest'

import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { _resetAdoptedLaunchProfileForTests } from '@/lib/platform'
import { $gateway, setPrimaryGateway } from '@/store/gateway'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'

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
  }
}))

const agent = (id: string) => ({ authMode: 'oauth', headerNames: [], id, kind: 'cloud', label: id, url: `https://${id}.example` })
const profiles: Record<string, string[]> = { alpha: ['default'], beta: ['default', 'research'] }

afterEach(() => {
  for (const key of ['hermesDesktop', '__HERMES_AUTH_REQUIRED__', '__HERMES_HOST_AGENT__', '__HERMES_UI_SURFACE__']) {
    Reflect.deleteProperty(window, key)
  }

  document.documentElement.removeAttribute('data-hermes-desktop-host')
  $connection.set(null)
  _resetAdoptedLaunchProfileForTests()
  vi.unstubAllGlobals()
})

// The hosted dashboard: no agent of its own; the window's socket opened on the list's primary (Alpha).
it('keeps a profile named with no agent on the agent the window socket opened on, after the window moved', async () => {
  let primaryId = 'alpha'
  const registry = () => ({ connections: [agent('alpha'), agent('beta')], primary: primaryId, version: 2 })

  vi.stubGlobal('fetch', vi.fn(async (url: URL) => {
    const [, id, rest] = url.pathname.match(/^\/api\/relay\/([^/]+)(\/.*)$/) ?? []

    if (url.pathname === '/api/gateway-connections/beta/primary') {
      primaryId = 'beta'
    }

    const body = !id
      ? url.pathname === '/api/gateway-connections'
        ? registry()
        : url.pathname.endsWith('/primary')
          ? { ok: true, registry: registry() }
          : { ticket: 'relay-ticket' }
      : rest === '/api/profiles'
        ? { profiles: profiles[id].map(name => ({ name })) }
        : { current: 'default', from: id }

    return new Response(JSON.stringify(body))
  }))
  Object.assign(window, { __HERMES_AUTH_REQUIRED__: true, __HERMES_HOST_AGENT__: false, __HERMES_UI_SURFACE__: 'webapp' })
  expect(installBrowserDesktopBridge()).toBe(true)
  const desktop = window.hermesDesktop!
  const primary = { connectionState: 'open', request: vi.fn(async () => ({})) }
  setPrimaryGateway(primary as never, 'default')
  $connection.set(await desktop.getConnection())
  // Beta becomes the account's primary and the window moves to Beta's research profile; then
  // `/profile default` takes the bare profile door.
  await desktop.connections!.setPrimary('beta')
  $connection.set(await desktop.getConnectionFor!({ connectionId: 'beta', profile: 'research' }))
  $activeGatewayProfile.set('research')

  await ensureGatewayProfile('default')

  expect($gateway.get()).toBe(primary)
  expect($connection.get()).toMatchObject({ connectionId: 'alpha', profile: 'default' })
  expect(await desktop.api({ path: '/api/config' })).toMatchObject({ from: 'alpha' })
})
