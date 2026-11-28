import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'
import type { MoaConfigResponse } from '@/types/hermes'

import { type ProfileScope, setApiRequestConnection, setApiRequestProfile } from './client'
import {
  activateCustomEndpoint,
  deleteCustomEndpoint,
  disconnectOAuthProvider,
  getCustomEndpoints,
  getEnvVars,
  getHermesConfig,
  getHermesConfigSchema,
  saveCustomEndpoint,
  validateCustomEndpoint
} from './config'
import { getAuxiliaryModels, getGlobalModelInfo, getMoaModels, saveMoaModels } from './models'
import {
  getElevenLabsVoices,
  getMemoryProviderConfig,
  getMemoryProviderOAuthStatus,
  getMemoryStatus,
  saveMemoryProviderConfig,
  startMemoryProviderOAuth
} from './system'

const endpoint = { base_url: 'https://models.example/v1', model: 'model-a', name: 'Model server' }

const moa: MoaConfigResponse = {
  default_preset: 'custom',
  active_preset: 'custom',
  presets: {},
  aggregator: { provider: 'nous', model: 'model-a' },
  aggregator_temperature: 0.5,
  degraded_reference_policy: 'loud',
  enabled: false,
  reference_models: [],
  reference_temperature: 0.5,
  reference_timeout: null
}

// Only the preload transport is replaced; the wrappers and ambient/explicit
// scope resolution run unchanged, including hermesApi's ambient connection tag.
const operations: Array<{
  name: string
  request: HermesApiRequest
  run: (scope?: ProfileScope) => Promise<unknown>
}> = [
  { name: 'config', request: { path: '/api/config' }, run: getHermesConfig },
  { name: 'schema', request: { path: '/api/config/schema' }, run: getHermesConfigSchema },
  { name: 'credentials', request: { path: '/api/env' }, run: getEnvVars },
  { name: 'endpoints', request: { path: '/api/providers/custom-endpoints' }, run: getCustomEndpoints },
  {
    name: 'save endpoint',
    request: { path: '/api/providers/custom-endpoints', method: 'POST', body: endpoint },
    run: scope => saveCustomEndpoint(endpoint, scope)
  },
  {
    name: 'validate endpoint',
    request: { path: '/api/providers/custom-endpoints/validate', method: 'POST', body: endpoint },
    run: scope => validateCustomEndpoint(endpoint, scope)
  },
  {
    name: 'activate endpoint',
    request: { path: '/api/providers/custom-endpoints/endpoint%2Fa/activate', method: 'POST' },
    run: scope => activateCustomEndpoint('endpoint/a', scope)
  },
  {
    name: 'delete endpoint',
    request: { path: '/api/providers/custom-endpoints/endpoint%2Fa', method: 'DELETE' },
    run: scope => deleteCustomEndpoint('endpoint/a', scope)
  },
  {
    name: 'disconnect provider',
    request: { path: '/api/providers/oauth/nous', method: 'DELETE' },
    run: scope => disconnectOAuthProvider('nous', scope)
  },
  { name: 'model info', request: { path: '/api/model/info' }, run: getGlobalModelInfo },
  { name: 'auxiliary models', request: { path: '/api/model/auxiliary' }, run: getAuxiliaryModels },
  { name: 'MoA models', request: { path: '/api/model/moa' }, run: getMoaModels },
  {
    name: 'save MoA',
    request: { path: '/api/model/moa', method: 'PUT', body: moa },
    run: scope => saveMoaModels(moa, scope)
  },
  {
    name: 'memory provider config',
    request: { path: '/api/memory/providers/honcho/config?surface=declared' },
    run: scope => getMemoryProviderConfig('honcho', scope)
  },
  {
    name: 'save memory provider config',
    request: {
      path: '/api/memory/providers/honcho/config?surface=declared',
      method: 'PUT',
      body: { values: { workspace: 'research' } }
    },
    run: scope => saveMemoryProviderConfig('honcho', { workspace: 'research' }, scope)
  },
  {
    name: 'start memory OAuth',
    request: { path: '/api/memory/providers/honcho/oauth/start', method: 'POST' },
    run: scope => startMemoryProviderOAuth('honcho', scope)
  },
  {
    name: 'poll memory OAuth',
    request: { path: '/api/memory/providers/honcho/oauth/status' },
    run: scope => getMemoryProviderOAuthStatus('honcho', scope)
  },
  { name: 'memory status', request: { path: '/api/memory' }, run: getMemoryStatus },
  { name: 'voice catalog', request: { path: '/api/audio/elevenlabs/voices' }, run: getElevenLabsVoices }
]

describe('settings API owner scope', () => {
  const api = vi.fn(async (_request: HermesApiRequest) => ({}))

  beforeEach(() => {
    api.mockClear()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestConnection('ambient-gateway')
    setApiRequestProfile('ambient-profile')
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it.each(operations)('$name isolates same-named profiles on distinct gateways', async ({ run, request }) => {
    for (const connectionId of ['gateway-a', 'gateway-b', 'local']) {
      const owner = Object.freeze({ connectionId, profile: 'worker' })
      await run(owner)
      expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
        ...request,
        connectionId,
        profile: 'worker',
        priority: 'foreground'
      })
      setApiRequestConnection('another-ambient-gateway')
      setApiRequestProfile('another-ambient-profile')
    }
  })

  it.each(operations)('$name preserves classic omitted, null, and string scopes', async ({ run, request }) => {
    await run()
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
      ...request,
      connectionId: 'ambient-gateway',
      profile: 'ambient-profile'
    })
    expect(api.mock.calls.at(-1)?.[0]).not.toHaveProperty('priority')

    await run(null)
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({ ...request, connectionId: 'ambient-gateway' })
    expect(api.mock.calls.at(-1)?.[0]).not.toHaveProperty('profile')
    expect(api.mock.calls.at(-1)?.[0]).not.toHaveProperty('priority')

    await run(' worker ')
    expect(api.mock.calls.at(-1)?.[0]).toMatchObject({
      ...request,
      connectionId: 'ambient-gateway',
      profile: ' worker ',
      priority: 'foreground'
    })
  })
})
