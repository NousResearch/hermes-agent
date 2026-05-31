import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OAuthProvider } from '@/types/hermes'

import {
  $desktopOnboarding,
  type DesktopOnboardingState,
  type OnboardingContext,
  refreshOnboarding,
  requestDesktopOnboarding
} from './onboarding'

function provider(id: string, name = id): OAuthProvider {
  return {
    cli_command: `hermes login ${id}`,
    docs_url: `https://example.com/${id}`,
    flow: 'pkce',
    id,
    name,
    status: { logged_in: false }
  }
}

function baseState(overrides: Partial<DesktopOnboardingState> = {}): DesktopOnboardingState {
  return {
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    manual: false,
    ...overrides
  }
}

function installApiMock(api: (request: { path: string }) => Promise<unknown>) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api }
  })
}

function runtimeMismatchGateway(): OnboardingContext['requestGateway'] {
  return async method => {
    if (method === 'setup.status') {
      return { provider_configured: true } as never
    }

    if (method === 'setup.runtime_check') {
      return { error: 'Selected runtime is not available.', ok: false } as never
    }

    throw new Error(`unexpected gateway method: ${method}`)
  }
}

function onboardingContext(requestGateway: OnboardingContext['requestGateway']): OnboardingContext {
  return { requestGateway }
}

describe('refreshOnboarding', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $desktopOnboarding.set(baseState())
  })

  afterEach(() => {
    window.localStorage.clear()
    $desktopOnboarding.set(baseState())
    vi.restoreAllMocks()
  })

  it('refreshes OAuth providers again when onboarding was explicitly requested', async () => {
    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path === '/api/providers/oauth') {
        return { providers: [provider('fresh')] }
      }

      throw new Error(`unexpected api path: ${path}`)
    })

    installApiMock(api)
    $desktopOnboarding.set(baseState({ providers: [provider('cached')] }))
    requestDesktopOnboarding('Need provider setup')

    const ready = await refreshOnboarding(onboardingContext(runtimeMismatchGateway()))

    expect(ready).toBe(false)
    expect(api).toHaveBeenCalledTimes(1)
    expect($desktopOnboarding.get().providers?.map(p => p.id)).toEqual(['fresh'])
    expect($desktopOnboarding.get().reason).toContain('Selected runtime is not available.')
    expect($desktopOnboarding.get().reason).toContain('setup.status reports configured credentials')
  })

  it('keeps cached providers when onboarding was not re-requested', async () => {
    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path === '/api/providers/oauth') {
        return { providers: [provider('fresh')] }
      }

      throw new Error(`unexpected api path: ${path}`)
    })

    installApiMock(api)
    $desktopOnboarding.set(baseState({ providers: [provider('cached')] }))

    const ready = await refreshOnboarding(onboardingContext(runtimeMismatchGateway()))

    expect(ready).toBe(false)
    expect(api).not.toHaveBeenCalled()
    expect($desktopOnboarding.get().providers?.map(p => p.id)).toEqual(['cached'])
  })

  it('deduplicates concurrent provider refresh calls', async () => {
    let resolveProviders!: (value: { providers: OAuthProvider[] }) => void

    const providersPromise = new Promise<{ providers: OAuthProvider[] }>(resolve => {
      resolveProviders = value => {
        resolve(value)
      }
    })

    const api = vi.fn(async ({ path }: { path: string }) => {
      if (path === '/api/providers/oauth') {
        return providersPromise
      }

      throw new Error(`unexpected api path: ${path}`)
    })

    installApiMock(api)
    $desktopOnboarding.set(baseState({ requested: true }))

    const first = refreshOnboarding(onboardingContext(runtimeMismatchGateway()))
    const second = refreshOnboarding(onboardingContext(runtimeMismatchGateway()))

    await vi.waitFor(() => expect(api).toHaveBeenCalledTimes(1))

    resolveProviders({ providers: [provider('shared')] })
    await Promise.all([first, second])

    expect($desktopOnboarding.get().providers?.map(p => p.id)).toEqual(['shared'])
  })
})
