import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import * as config from './config'

const endpoint = { name: 'Fixture endpoint', base_url: 'https://fixture.invalid/v1', model: 'fixture-model' }

const operations = [
  (owner: { connectionId: string; profile: string }) => config.getHermesConfig(owner),
  (owner: { connectionId: string; profile: string }) => config.getHermesConfigRecord(owner),
  (owner: { connectionId: string; profile: string }) => config.getHermesConfigDefaults(owner),
  (owner: { connectionId: string; profile: string }) => config.getHermesConfigSchema(owner),
  (owner: { connectionId: string; profile: string }) => config.saveHermesConfig({ model: 'fixture' }, owner),
  (owner: { connectionId: string; profile: string }) => config.saveHermesConfigRecord({ model: 'fixture' }, owner),
  (owner: { connectionId: string; profile: string }) => config.getEnvVars(owner),
  (owner: { connectionId: string; profile: string }) => config.setEnvVar('FIXTURE_API_KEY', 'fixture-value', owner),
  (owner: { connectionId: string; profile: string }) => config.deleteEnvVar('FIXTURE_API_KEY', owner),
  (owner: { connectionId: string; profile: string }) => config.revealEnvVar('FIXTURE_API_KEY', owner),
  (owner: { connectionId: string; profile: string }) =>
    config.validateProviderCredential('FIXTURE_API_KEY', 'fixture-value', undefined, owner),
  (owner: { connectionId: string; profile: string }) => config.getCustomEndpoints(owner),
  (owner: { connectionId: string; profile: string }) => config.saveCustomEndpoint(endpoint, owner),
  (owner: { connectionId: string; profile: string }) => config.validateCustomEndpoint(endpoint, owner),
  (owner: { connectionId: string; profile: string }) => config.activateCustomEndpoint('fixture-endpoint', owner),
  (owner: { connectionId: string; profile: string }) => config.deleteCustomEndpoint('fixture-endpoint', owner),
  (owner: { connectionId: string; profile: string }) => config.listOAuthProviders(owner),
  (owner: { connectionId: string; profile: string }) => config.disconnectOAuthProvider('fixture-provider', owner),
  (owner: { connectionId: string; profile: string }) => config.startOAuthLogin('fixture-provider', owner),
  (owner: { connectionId: string; profile: string }) =>
    config.submitOAuthCode('fixture-provider', 'fixture-session', 'fixture-code', owner),
  (owner: { connectionId: string; profile: string }) =>
    config.pollOAuthSession('fixture-provider', 'fixture-session', owner),
  (owner: { connectionId: string; profile: string }) => config.cancelOAuthSession('fixture-session', owner)
]

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

describe('settings config transport ownership', () => {
  it.each(['local', 'fixture-remote'])(
    'keeps every config/provider request on the explicit %s owner',
    async connectionId => {
      const owner = { connectionId, profile: 'shared-name' }

      const api = vi.fn(async (request: HermesApiRequest) => {
        expect(request).toMatchObject(owner)
        expect(typeof request.profile).toBe('string')

        return {}
      })

      vi.stubGlobal('hermesDesktop', { api })
      setApiRequestConnection('different-ambient')
      setApiRequestProfile('ambient-profile')

      for (const operation of operations) {
        await operation(owner)
      }

      expect(api).toHaveBeenCalledTimes(operations.length)
    }
  )

  it('retains the distinction between omitted and explicit null profile scopes', async () => {
    const api = vi.fn(async () => ({}))
    vi.stubGlobal('hermesDesktop', { api })
    setApiRequestConnection('fixture-ambient')
    setApiRequestProfile('fixture-active')
    await config.getEnvVars()
    await config.getEnvVars(null)

    expect(api.mock.calls).toEqual([
      [{ connectionId: 'fixture-ambient', profile: 'fixture-active', path: '/api/env' }],
      [{ connectionId: 'fixture-ambient', path: '/api/env' }]
    ])
  })
})
