import { JsonRpcGatewayError } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestGatewayForAgent = vi.fn()

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

const ensureGatewayProfile = vi.fn()
const ensureGatewayAgent = vi.fn()

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: { get: () => 'default' },
  $newChatProfile: { get: () => null },
  ensureGatewayAgent: (...args: unknown[]) => ensureGatewayAgent(...args),
  ensureGatewayProfile: (...args: unknown[]) => ensureGatewayProfile(...args),
  normalizeProfileKey: (value: string) => value,
  resolveNewChatOwnerRoute: () => null
}))

// The composer atoms as syncRuntimeMetadataToView leaves them after a session.info
// heartbeat: $currentProvider holds the runtime's resolved provider class, not a pick.
let composerModel = ''
let composerProvider = ''

vi.mock('@/store/session', () => ({
  $currentCwdExplicit: { get: () => false },
  $currentFastMode: { get: () => false },
  $currentModel: { get: () => composerModel },
  $currentProvider: { get: () => composerProvider },
  $currentReasoningEffort: { get: () => 'low' },
  $currentServiceTier: { get: () => 'normal' },
  getCurrentModelSource: () => 'manual'
}))

import { createGatewaySession, desktopSessionCreateParams } from './session-create-request'

// Verbatim admission rejections from older backends (tui_gateway/contracts/registry.py).
const rejection = (field: string, suffix = '') =>
  new JsonRpcGatewayError(`invalid params for session.create: ${field}: Extra inputs are not permitted${suffix}`, {
    code: 4000
  })

const OUT_OF_SYNC =
  ' — the client and the Hermes backend are out of sync (different versions); run `hermes update` and restart both'

const PRE_122899_REJECTION = rejection('cwd_explicit', OUT_OF_SYNC) // v0.21.4 – v0.21.5 wording
const V0213_REJECTION = rejection('cwd_explicit') // v0.21.3 wording, no out-of-sync suffix

const params = { cols: 96, cwd: '/work/repo', cwd_explicit: true, profile: 'default', source: 'desktop' }
const route = { connectionId: 'cloud', profile: 'default' }

describe('createGatewaySession', () => {
  beforeEach(() => requestGatewayForAgent.mockReset())

  it('resends once without cwd_explicit (cwd kept) when a pre-#122899 backend rejects the flag', async () => {
    const { cwd_explicit: _flag, ...withoutFlag } = params

    requestGatewayForAgent.mockRejectedValueOnce(PRE_122899_REJECTION).mockResolvedValueOnce({ session_id: 's1' })
    await expect(createGatewaySession(route, params, vi.fn())).resolves.toEqual({ session_id: 's1' })
    expect(requestGatewayForAgent).toHaveBeenNthCalledWith(
      2,
      'cloud',
      'default',
      'session.create',
      withoutFlag,
      undefined,
      undefined,
      { spawnPriority: 'foreground' }
    )
    expect(requestGatewayForAgent.mock.calls[0][3]).toEqual(params)

    const requestGateway = vi.fn().mockRejectedValueOnce(V0213_REJECTION).mockResolvedValueOnce({ session_id: 's2' })
    await expect(createGatewaySession(null, params, requestGateway)).resolves.toEqual({ session_id: 's2' })
    expect(requestGateway.mock.calls.map(call => call[1])).toEqual([params, withoutFlag])
  })

  it('drops service_tier (fast kept) for a pre-Ultrafast backend, one rejected field at a time', async () => {
    const tiered = { ...params, fast: true, service_tier: 'ultrafast' }
    const { cwd_explicit: _flag, service_tier: _tier, ...legacy } = tiered

    const requestGateway = vi
      .fn()
      .mockRejectedValueOnce(rejection('service_tier', OUT_OF_SYNC))
      .mockRejectedValueOnce(V0213_REJECTION)
      .mockResolvedValueOnce({ session_id: 's3' })

    await expect(createGatewaySession(null, tiered, requestGateway)).resolves.toEqual({ session_id: 's3' })
    expect(requestGateway.mock.calls.map(call => call[1])).toEqual([tiered, { ...legacy, cwd_explicit: true }, legacy])
  })

  it('never resends on any other failure, even one that mentions the field', async () => {
    const cases: Array<[Error, Record<string, unknown>]> = [
      [new Error('request timed out'), params],
      [new Error('handler error: could not resolve cwd_explicit workspace'), params],
      [rejection('some_newer_field', OUT_OF_SYNC), params],
      // The flag was never sent, so a resend would be identical.
      [PRE_122899_REJECTION, { cwd: '/work/repo' }]
    ]

    for (const [failure, sentParams] of cases) {
      const requestGateway = vi.fn().mockRejectedValue(failure)

      await expect(createGatewaySession(null, sentParams, requestGateway)).rejects.toBe(failure)
      expect(requestGateway).toHaveBeenCalledTimes(1)
    }
  })
})

describe('desktopSessionCreateParams', () => {
  beforeEach(() => {
    ensureGatewayProfile.mockReset().mockResolvedValue(undefined)
    ensureGatewayAgent.mockReset().mockResolvedValue(undefined)
    composerModel = ''
    composerProvider = ''
  })

  it('drops the heartbeat-echoed generic `custom` class instead of shipping a credentialless bare-custom pick', async () => {
    // What syncRuntimeMetadataToView leaves behind after a session.info heartbeat for a
    // session on a named custom provider (custom:*): the resolved class, not a user pick.
    composerModel = 'DeepSeek-V4.1-Flash'
    composerProvider = 'custom'

    const created = await desktopSessionCreateParams('/work/repo')

    expect(created.model).toBe('DeepSeek-V4.1-Flash')
    expect('provider' in created).toBe(false)
  })

  it('ships a real catalog slug and the named custom: form untouched', async () => {
    composerModel = 'GLM-5.3'

    for (const provider of ['myhosted', 'custom:myhosted']) {
      composerProvider = provider

      await expect(desktopSessionCreateParams('/work/repo')).resolves.toMatchObject({ model: 'GLM-5.3', provider })
    }
  })
})
