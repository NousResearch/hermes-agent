import { afterAll, expect, it, vi } from 'vitest'

import { profileScoped } from '@/api/client'
import { getApiRequestProfile, hermesApi } from '@/hermes'
import { installBrowserDesktopBridge } from '@/lib/browser-desktop-bridge'
import { browserLaunchProfile } from '@/lib/platform'
import { $activeGatewayProfile } from '@/store/profile'

// The hosted dashboard runs no agent and names no profile in its page: which profile the window's
// agent runs as is known only once boot reads it through the relay. A request sent before that must
// name no profile (the agent answers for its own), never `default`, which may be another profile there.
type PageGlobals = { __HERMES_AUTH_REQUIRED__?: boolean; __HERMES_HOST_AGENT__?: boolean; __HERMES_UI_SURFACE__?: string }

vi.hoisted(() => {
  Object.assign(window, { __HERMES_AUTH_REQUIRED__: true, __HERMES_HOST_AGENT__: false, __HERMES_UI_SURFACE__: 'webapp' })
})

afterAll(() => {
  const page = window as Window & PageGlobals
  delete page.__HERMES_AUTH_REQUIRED__
  delete page.__HERMES_HOST_AGENT__
  delete page.__HERMES_UI_SURFACE__
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.unstubAllGlobals()
})

const alpha = { authMode: 'oauth', headerNames: [], id: 'alpha', kind: 'cloud', label: 'Alpha', url: 'https://alpha.example' }

it('names no profile on a request sent before boot learns the profile the agent runs as', async () => {
  const sent: string[] = []

  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: URL) => {
      sent.push(`${url.pathname}${url.search}`)

      const body =
        url.pathname === '/api/gateway-connections'
          ? { connections: [alpha], primary: 'alpha', version: 2 }
          : url.pathname === '/api/relay/alpha/api/profiles/active'
            ? { current: 'research' }
            : {}

      return new Response(JSON.stringify(body))
    })
  )
  expect(installBrowserDesktopBridge()).toBe(true)

  expect(getApiRequestProfile()).toBeNull()
  await hermesApi({ ...profileScoped(), path: '/api/config' })
  expect(sent).toContain('/api/relay/alpha/api/config')

  // Boot reads the agent's profile, then the window's route moves onto it.
  await window.hermesDesktop.getConnection()
  $activeGatewayProfile.set(browserLaunchProfile())
  expect(getApiRequestProfile()).toBe('research')

  // A user's own pick of `default` is named.
  $activeGatewayProfile.set('default')
  expect(getApiRequestProfile()).toBe('default')
})
