import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

import { deferred } from '../test/deferred'

const { connect } = vi.hoisted(() => ({ connect: vi.fn() }))

// Keep the real profile/gateway stores; only the transport and view effects are inert.
vi.mock('@/hermes', () => ({
  setApiRequestProfile: vi.fn(),
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async (url: string) => {
      await connect(url)
      this.connectionState = 'open'
    }
    close = () => { this.connectionState = 'closed' }
    onEvent = () => () => {}
    onState = () => () => {}
  }
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const gateway = await import('./gateway')
const { $activeGatewayProfile, ensureGatewayProfile } = await import('./profile')
const { $connection } = await import('./session')

const descriptor = (profile: string): HermesConnection => ({
  authMode: 'token',
  isFullscreen: false,
  nativeOverlayWidth: 0,
  logs: [],
  windowButtonPosition: null,
  baseUrl: `https://${profile}.invalid`,
  mode: 'remote',
  profile,
  token: 'test-token',
  wsUrl: `wss://${profile}.invalid/ws`
})

afterEach(() => {
  gateway.closeSecondaryGateways()
  vi.unstubAllGlobals()
})

it('returns to A when A is requested while B is still dialing', async () => {
  const dial = deferred<void>()
  connect.mockImplementationOnce(() => dial.promise)
  const primary = { connectionState: 'open' }
  gateway.configureGatewayRegistry({
    onEvent: vi.fn(),
    onActiveRouteChanged: profile => $activeGatewayProfile.set(profile)
  })
  gateway.setPrimaryGateway(primary as never, 'default')
  await gateway.ensureGatewayForProfile('default')
  $activeGatewayProfile.set('default')
  const getConnection = vi.fn(async (profile: string) => descriptor(profile))
  vi.stubGlobal('window', { hermesDesktop: { getConnection } })

  const b = ensureGatewayProfile('work')
  await vi.waitFor(() => expect(connect).toHaveBeenCalledOnce())
  const a = ensureGatewayProfile('default')
  dial.resolve()
  await Promise.all([b, a])

  expect($activeGatewayProfile.get()).toBe('default')
  expect($connection.get()?.profile).toBe('default')
  expect(gateway.activeGateway()).toBe(primary)
  const calls = getConnection.mock.calls.length
  await ensureGatewayProfile('default')
  expect(getConnection).toHaveBeenCalledTimes(calls) // The post-mutex no-op still avoids redialing.
})
