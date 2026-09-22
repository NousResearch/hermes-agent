import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { startOAuthLogin } from '@/api/config'
import type { HermesApiRequest } from '@/global'
import type * as HermesApi from '@/hermes'
import {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  requestGatewayForAgent,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} from '@/store/gateway'
import type * as OnboardingStore from '@/store/onboarding'
import {
  $desktopOnboarding,
  closeManualOnboarding,
  type OnboardingContext,
  startManualProviderOAuth
} from '@/store/onboarding'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { FlowPanel } from './flow'

import { DesktopOnboardingOverlay } from '.'

stubResizeObserver()
stubMenuDomApis()

// Boundary test: real overlay/API helpers/gateway router; no OAuth
// account or production backend. Intercept acquisition to inspect its context.
const mocks = vi.hoisted(() => ({
  begin: vi.fn(),
  rpc: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/onboarding', async importOriginal => ({
  ...(await importOriginal<typeof OnboardingStore>()),
  startProviderOAuth: mocks.begin
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async () => {
      this.connectionState = 'open'
    }
    request = mocks.rpc
    close = () => {
      this.connectionState = 'closed'
    }
    onEvent = () => () => {}
    onState = () => () => {}
  }
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const provider = { ...makeOAuthProvider('openai-codex'), flow: 'device_code' as const }

afterEach(() => {
  cleanup()
  closeManualOnboarding()
  closeSecondaryGateways()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

it.each([
  { profile: 'leverage-ai', sharedRemote: true },
  { profile: 'custom', sharedRemote: true },
  { profile: 'mara', sharedRemote: false }
])('keeps Settings readiness on its owner ($profile, shared=$sharedRemote)', async ({ profile, sharedRemote }) => {
  const localRequest = vi.fn(async () => ({ ok: true }))

  const localDial = vi.fn(async () => {
    throw new Error('Diagnostic local backend has no remote-only profile')
  })

  const remoteDial = vi.fn(async ({ connectionId, profile: routeProfile }) => ({
    connectionId,
    profile: routeProfile,
    sharedRemote,
    ...(sharedRemote ? {} : { remoteKind: 'ssh', remoteProfile: 'remote-research' }),
    mode: 'remote',
    authMode: 'token',
    baseUrl: 'https://athena.invalid',
    wsUrl: 'wss://athena.invalid/api/ws',
    token: 'fixture-only'
  }))

  const api = vi.fn(async () => ({ providers: [provider] }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api,
      getConnection: localDial,
      getConnectionFor: remoteDial,
      touchBackend: vi.fn(async () => undefined)
    }
  })
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open', request: localRequest } as never, 'default')
  setPrimaryGatewayConnection({ connectionId: 'local' })
  await ensureGatewayForAgent('athena', profile)
  setApiRequestConnection('athena')
  setApiRequestProfile(profile)
  localDial.mockClear()

  await act(async () => {
    startManualProviderOAuth('openai-codex', profile)
  })
  render(
    <DesktopOnboardingOverlay
      enabled={false}
      profile={profile}
      requestGateway={(method, params) => requestGatewayForAgent('athena', profile, method, params)}
    />
  )
  await waitFor(() => expect(mocks.begin).toHaveBeenCalledOnce())
  const context = mocks.begin.mock.calls[0][1] as OnboardingContext

  // Switching the foreground device must not retarget a pending setup.
  setApiRequestConnection('local')
  setApiRequestProfile('default')
  await startOAuthLogin('openai-codex', context.scope)
  expect(api).toHaveBeenLastCalledWith(
    expect.objectContaining({
      connectionId: 'athena',
      profile,
      path: '/api/providers/oauth/openai-codex/start'
    })
  )
  await context.requestGateway('setup.status')
  await context.requestGateway('setup.runtime_check', { provider: 'openai-codex' })
  await context.requestGateway('model.options', { profile, explicit_only: true })
  const params = sharedRemote ? { profile } : {}
  expect(mocks.rpc).toHaveBeenCalledWith('setup.status', params)
  expect(localDial).not.toHaveBeenCalled()
  expect(mocks.rpc).toHaveBeenCalledWith('setup.runtime_check', { provider: 'openai-codex', ...params })
  expect(mocks.rpc).toHaveBeenCalledWith('model.options', { explicit_only: true, ...params })
  expect(localRequest).not.toHaveBeenCalled()
})

it('keeps an explicitly untagged owner through Change → Add provider', async () => {
  const owner = { connectionId: null, profile: 'legacy' }
  const api = vi.fn(async (_request: HermesApiRequest) => ({ providers: [] }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  startManualProviderOAuth('openai-codex', owner)
  setApiRequestConnection('athena')
  setApiRequestProfile('default')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <FlowPanel
        ctx={{ scope: owner, profile: owner.profile, requestGateway: async () => ({ providers: [] }) as never }}
        flow={{
          status: 'confirming_model',
          currentModel: 'fixture-model',
          providerSlug: 'openai-codex',
          label: 'Codex',
          saving: false
        }}
        leaving={false}
        onBegin={() => undefined}
      />
    </QueryClientProvider>
  )
  fireEvent.click(screen.getByRole('button', { name: 'Change' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Add provider' }))
  expect($desktopOnboarding.get().targetScope).toEqual(owner)
  await waitFor(() =>
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/providers/oauth', profile: 'legacy' }))
  )
  expect(api.mock.calls.every(([request]) => !(request as { connectionId?: string }).connectionId)).toBe(true)
  client.clear()
})

it.each([null, undefined])('pins a registry owner when its profile is %s', async profile => {
  const localRequest = vi.fn(async (_method: string, _params?: Record<string, unknown>) => ({ ok: true }))

  const localDial = vi.fn(async () => {
    throw new Error('A registry-owned setup must not dial the legacy connection')
  })

  const remoteDial = vi.fn(async () => ({
    connectionId: 'athena',
    profile: 'default',
    sharedRemote: true,
    mode: 'remote',
    authMode: 'token',
    baseUrl: 'https://athena.invalid',
    wsUrl: 'wss://athena.invalid/api/ws',
    token: 'fixture-only'
  }))

  const api = vi.fn(async () => ({ providers: [provider] }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api,
      getConnection: localDial,
      getConnectionFor: remoteDial,
      touchBackend: vi.fn(async () => undefined)
    }
  })
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open', request: localRequest } as never, 'default')
  setPrimaryGatewayConnection({ connectionId: 'local' })
  setApiRequestConnection('local')
  setApiRequestProfile('default')

  await act(async () => {
    startManualProviderOAuth('openai-codex', { connectionId: 'athena', profile })
  })
  render(
    <DesktopOnboardingOverlay
      enabled={false}
      profile="default"
      requestGateway={async (method, params) => (await localRequest(method, params)) as never}
    />
  )
  await waitFor(() => expect(mocks.begin).toHaveBeenCalledOnce())
  const context = mocks.begin.mock.calls[0][1] as OnboardingContext

  await startOAuthLogin('openai-codex', context.scope)
  expect(api).toHaveBeenLastCalledWith(
    expect.objectContaining({ connectionId: 'athena', path: '/api/providers/oauth/openai-codex/start' })
  )
  await context.requestGateway('setup.status')
  await context.requestGateway('setup.runtime_check', { provider: 'openai-codex' })
  expect(remoteDial).toHaveBeenCalledWith({ connectionId: 'athena', profile: 'default' })
  expect(mocks.rpc).toHaveBeenCalledWith('setup.status', { profile: 'default' })
  expect(mocks.rpc).toHaveBeenCalledWith('setup.runtime_check', { provider: 'openai-codex', profile: 'default' })
  expect(localDial).not.toHaveBeenCalled()
  expect(localRequest).not.toHaveBeenCalled()
})
