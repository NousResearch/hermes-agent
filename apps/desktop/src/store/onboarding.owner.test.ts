import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { HermesApiRequest } from '@/global'
import type { OAuthProvider, OAuthStartResponse } from '@/types/hermes'

import { $desktopOnboarding, cancelOnboardingFlow, saveOnboardingApiKey, startProviderOAuth } from './onboarding'

afterEach(() => {
  cancelOnboardingFlow()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

it.each(
  (['start', 'browser'] as const).flatMap(stage =>
    (['ambient', 'legacy', 'pinned'] as const).map(scope => ({ stage, scope }))
  )
)('keeps delayed $stage OAuth cleanup on its initiating $scope owner', async ({ stage, scope }) => {
  const provider: OAuthProvider = {
    id: 'fixture',
    name: 'Fixture',
    flow: 'pkce',
    cli_command: '',
    docs_url: '',
    status: { logged_in: false }
  }

  const start: OAuthStartResponse = {
    flow: 'pkce',
    session_id: 'fixture-session',
    auth_url: 'https://example.invalid/sign-in',
    expires_in: 60
  }

  let finishStart!: (value: OAuthStartResponse) => void
  let finishBrowser!: () => void

  const heldStart = new Promise<OAuthStartResponse>(resolve => {
    finishStart = resolve
  })

  const heldBrowser = new Promise<void>(resolve => {
    finishBrowser = resolve
  })

  const api = vi.fn(async (request: HermesApiRequest) => {
    if (request.method === 'POST' && request.path === '/api/providers/oauth/fixture/start') {
      return heldStart
    }

    if (request.method === 'DELETE' && request.path === '/api/providers/oauth/sessions/fixture-session') {
      return { ok: true }
    }

    throw new Error(`Unexpected request: ${request.method} ${request.path}`)
  })

  const openExternal = vi.fn(() => heldBrowser)
  vi.stubGlobal('hermesDesktop', { api, openExternal })
  setApiRequestConnection('fixture-old')
  setApiRequestProfile('research')
  let current = true

  const profile =
    scope === 'pinned'
      ? { connectionId: 'fixture-old', profile: 'research' }
      : scope === 'legacy'
        ? 'research'
        : undefined

  const pending = startProviderOAuth(provider, { profile, isCurrent: () => current, requestGateway: vi.fn() })
  expect(api.mock.calls[0][0]).toMatchObject({ method: 'POST', connectionId: 'fixture-old', profile: 'research' })

  if (stage === 'browser') {
    finishStart(start)
    await vi.waitFor(() => expect(openExternal).toHaveBeenCalledOnce())
  }

  current = false
  setApiRequestConnection('fixture-new')
  cancelOnboardingFlow()
  // A later flow replaces the mutable global owner callback with a live one.
  await startProviderOAuth({ ...provider, flow: 'external' }, { isCurrent: () => true, requestGateway: vi.fn() })
  finishStart(start)
  finishBrowser()
  await pending

  const deletes = api.mock.calls.map(([request]) => request).filter(request => request.method === 'DELETE')
  expect(deletes).toEqual(
    scope === 'pinned'
      ? [
          {
            method: 'DELETE',
            path: '/api/providers/oauth/sessions/fixture-session',
            connectionId: 'fixture-old',
            profile: 'research'
          }
        ]
      : []
  )
  expect($desktopOnboarding.get().flow.status).toBe('external_pending')
  expect(openExternal).toHaveBeenCalledTimes(stage === 'browser' ? 1 : 0)
})

it('retires an onboarding continuation before a model write when its Settings owner changes', async () => {
  let finish!: (value: unknown) => void

  const held = new Promise(resolve => {
    finish = resolve
  })

  const api = vi.fn(async (request: HermesApiRequest) => (request.method === 'PUT' ? held : { providers: [] }))
  vi.stubGlobal('hermesDesktop', { api })
  let current = true

  const save = saveOnboardingApiKey('FIXTURE_API_KEY', 'fixture-value', 'Fixture', {
    profile: { connectionId: 'fixture-lab', profile: 'research' },
    isCurrent: () => current,
    requestGateway: vi.fn(async () => ({}) as never)
  })

  current = false
  finish({ ok: true })
  expect(await save).toEqual({ ok: false })
  expect(api).toHaveBeenCalledTimes(1)
  expect(api.mock.calls[0][0]).toMatchObject({ connectionId: 'fixture-lab', profile: 'research' })
})
