import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import { makeOAuthProvider } from '@/test/oauth-provider'

import {
  $desktopOnboarding,
  closeManualOnboarding,
  type OnboardingContext,
  saveOnboardingLocalEndpoint,
  setOnboardingCode,
  setOnboardingModel,
  startManualProviderOAuth,
  startProviderOAuth,
  submitOnboardingCode
} from './onboarding'
import { captureOnboardingScope } from './onboarding-scope'

const owner = { connectionId: 'athena', profile: 'leverage-ai' }
const requests: HermesApiRequest[] = []
let authFlow: 'device_code' | 'pkce'
let deferStart: (() => Promise<void>) | undefined

beforeEach(() => {
  vi.useFakeTimers()
  authFlow = 'device_code'
  deferStart = undefined
  requests.length = 0
  setApiRequestConnection(owner.connectionId)
  setApiRequestProfile(owner.profile)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      openExternal: vi.fn(async () => undefined),
      api: vi.fn(async (request: HermesApiRequest) => {
        requests.push(request)
        const { path } = request

        if (path === '/api/providers/oauth') {
          return { providers: [makeOAuthProvider('openai-codex')] }
        }

        if (path.endsWith('/start')) {
          await deferStart?.()

          return {
            flow: authFlow,
            session_id: 'fixture-session',
            auth_url: 'https://example.invalid/authorize',
            verification_url: 'https://example.invalid/device',
            user_code: 'fixture-code',
            expires_in: 600
          }
        }

        if (path.includes('/poll/') || path.endsWith('/submit')) {
          return { ok: true, status: 'approved' }
        }

        if (path.startsWith('/api/model/options')) {
          return { providers: [{ slug: 'openai-codex', models: ['fixture-model'] }] }
        }

        if (path.startsWith('/api/model/recommended-default')) {
          return { model: 'fixture-model' }
        }

        if (path === '/api/providers/validate') {
          return { ok: true, reachable: true, models: ['endpoint-model'] }
        }

        return { ok: true }
      })
    }
  })
})

afterEach(() => {
  closeManualOnboarding()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function beginContext(): OnboardingContext {
  startManualProviderOAuth('openai-codex', 'leverage-ai')

  return {
    scope: $desktopOnboarding.get().targetScope,
    requestGateway: vi.fn(
      async method => (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as never
    )
  }
}

function switchForeground() {
  setApiRequestConnection('local')
  setApiRequestProfile('default')
}

function expectOwnerRequests() {
  expect(requests.length).toBeGreaterThan(0)

  for (const request of requests) {
    expect(request).toMatchObject(owner)
  }
}

it.each(['device_code', 'pkce'] as const)(
  'pins %s completion and model edits after a foreground switch',
  async flow => {
    authFlow = flow
    const ctx = beginContext()
    await startProviderOAuth({ ...makeOAuthProvider('openai-codex'), flow }, ctx)
    switchForeground()

    if (flow === 'pkce') {
      setOnboardingCode('fixture-code')
      await submitOnboardingCode(ctx)
    } else {
      await vi.advanceTimersByTimeAsync(3000)
    }

    expect($desktopOnboarding.get().flow.status).toBe('confirming_model')
    expect(ctx.requestGateway).not.toHaveBeenCalledWith('reload.env')
    expect(ctx.requestGateway).toHaveBeenCalledWith('setup.runtime_check', { provider: 'openai-codex' })
    await setOnboardingModel('other-fixture-model', 'openai-codex')
    expect(requests.filter(request => request.path === '/api/model/set')).toHaveLength(2)
    expect(requests.some(request => request.path.includes(flow === 'pkce' ? '/submit' : '/poll/'))).toBe(true)
    expectOwnerRequests()
  }
)

it('cancels on the original owner after the foreground changes', async () => {
  const ctx = beginContext()
  await startProviderOAuth({ ...makeOAuthProvider('openai-codex'), flow: 'device_code' }, ctx)
  switchForeground()
  closeManualOnboarding()
  expect(requests.at(-1)).toMatchObject({ method: 'DELETE', path: '/api/providers/oauth/sessions/fixture-session' })
  expectOwnerRequests()
})

it('cancels a late OAuth start on its retired owner, not a newly opened flow', async () => {
  let release!: () => void
  deferStart = () =>
    new Promise<void>(resolve => {
      release = resolve
    })
  const ctx = beginContext()
  const pending = startProviderOAuth(makeOAuthProvider('openai-codex'), ctx)
  expect(release).toBeTypeOf('function')
  closeManualOnboarding()
  switchForeground()
  startManualProviderOAuth('openai-codex', 'default')
  release()
  await pending
  expect(requests.at(-1)).toMatchObject({ ...owner, method: 'DELETE' })
  expect($desktopOnboarding.get().targetScope).toEqual({ connectionId: 'local', profile: 'default' })
  expect($desktopOnboarding.get().flow.status).toBe('idle')
})

it('pins local-endpoint validation and persistence to the setup owner', async () => {
  const ctx = beginContext()
  switchForeground()
  await saveOnboardingLocalEndpoint('http://127.0.0.1:11434/v1', '', ctx)
  expect(requests.some(request => request.path === '/api/providers/validate')).toBe(true)
  expect(requests.some(request => request.path === '/api/model/set')).toBe(true)
  expectOwnerRequests()
})

it('preserves explicit local and untagged legacy routes instead of filling ambient halves', () => {
  expect(captureOnboardingScope({ connectionId: 'local', profile: 'default' })).toEqual({
    connectionId: 'local',
    profile: 'default'
  })
  expect(captureOnboardingScope({ profile: 'legacy' })).toEqual({ profile: 'legacy' })
  setApiRequestConnection(null)
  const legacy = captureOnboardingScope('legacy')
  setApiRequestConnection('athena')
  expect(legacy).toEqual({ connectionId: null, profile: 'legacy' })
})
