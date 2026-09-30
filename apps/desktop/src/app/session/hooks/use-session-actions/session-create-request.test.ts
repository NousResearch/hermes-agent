import { JsonRpcGatewayError } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestGatewayForAgent = vi.fn()

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

import { createGatewaySession } from './session-create-request'

// Verbatim admission rejections from backends predating #122899 (tui_gateway/contracts/registry.py).
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
    expect(requestGatewayForAgent.mock.calls.map(call => call[3])).toEqual([params, withoutFlag])

    const requestGateway = vi.fn().mockRejectedValueOnce(V0213_REJECTION).mockResolvedValueOnce({ session_id: 's2' })
    await expect(createGatewaySession(null, params, requestGateway)).resolves.toEqual({ session_id: 's2' })
    expect(requestGateway.mock.calls.map(call => call[1])).toEqual([params, withoutFlag])
  })

  it('never resends on any other failure, even one that mentions the field', async () => {
    const failures = [
      new Error('request timed out'),
      new Error('handler error: could not resolve cwd_explicit workspace'),
      rejection('some_newer_field', OUT_OF_SYNC),
      PRE_122899_REJECTION
    ]

    for (const [index, failure] of failures.entries()) {
      const requestGateway = vi.fn().mockRejectedValue(failure)
      // The last case: the flag was never sent, so a resend would be identical.
      const sentParams = index === failures.length - 1 ? { cwd: '/work/repo' } : params

      await expect(createGatewaySession(null, sentParams, requestGateway)).rejects.toBe(failure)
      expect(requestGateway).toHaveBeenCalledTimes(1)
    }
  })
})
