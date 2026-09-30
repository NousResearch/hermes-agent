import { JsonRpcGatewayError } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestGatewayForAgent = vi.fn()

vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))

import { createGatewaySession } from './session-create-request'

// Verbatim admission rejection from a backend predating #122899 (tui_gateway/contracts/registry.py).
const PRE_122899_REJECTION = new JsonRpcGatewayError(
  'invalid params for session.create: cwd_explicit: Extra inputs are not permitted — the client and the ' +
    'Hermes backend are out of sync (different versions); run `hermes update` and restart both',
  { code: 4000 }
)

const params = { cols: 96, cwd: '/work/repo', cwd_explicit: true, profile: 'default', source: 'desktop' }
const route = { connectionId: 'cloud', profile: 'default' }

describe('createGatewaySession', () => {
  beforeEach(() => requestGatewayForAgent.mockReset())

  it('resends once without cwd_explicit (cwd kept) when a pre-#122899 backend rejects the flag', async () => {
    requestGatewayForAgent.mockRejectedValueOnce(PRE_122899_REJECTION).mockResolvedValueOnce({ session_id: 's1' })

    await expect(createGatewaySession(route, params, vi.fn())).resolves.toEqual({ session_id: 's1' })

    const sent = requestGatewayForAgent.mock.calls.map(call => call[3])
    const { cwd_explicit: _flag, ...withoutFlag } = params

    expect(sent).toEqual([params, withoutFlag])
  })

  it('never resends on any other failure, even one that mentions the field', async () => {
    const failures = [
      new Error('request timed out'),
      new Error('handler error: could not resolve cwd_explicit workspace'),
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
