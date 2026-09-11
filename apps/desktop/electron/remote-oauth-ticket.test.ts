import { describe, expect, it } from 'vitest'

import { isReauthRequiredError } from './backend-health'
import { oauthTicketFailureAuthMessage } from './native-auth-decisions'
import { resolveRemoteOauthTicket, rosterSourceEnumerationTimeoutMs } from './remote-oauth-ticket'

describe('resolveRemoteOauthTicket', () => {
  it('reserves a larger bounded roster dial budget only for OAuth remote sources', () => {
    const defaultBudget = rosterSourceEnumerationTimeoutMs({ kind: 'local' })
    for (const kind of ['remote', 'cloud']) {
      expect(rosterSourceEnumerationTimeoutMs({ kind, authMode: 'oauth' })).toBeGreaterThan(defaultBudget)
      expect(rosterSourceEnumerationTimeoutMs({ kind, authMode: 'token' })).toBe(defaultBudget)
    }
    expect(rosterSourceEnumerationTimeoutMs({ kind: 'ssh', authMode: 'oauth' })).toBe(defaultBudget)
    expect(rosterSourceEnumerationTimeoutMs({ kind: 'remote', authMode: 'oauth' })).toBeLessThanOrEqual(30_000)
    expect(defaultBudget).toBeGreaterThan(0)
  })

  it('mints a fresh ticket on every dial even without a preflight native session', async () => {
    let mints = 0
    const deps = {
      hasNativeSession: () => false,
      mintGatewayWsTicket: async (baseUrl: string, headers: Record<string, string>) => {
        expect(baseUrl).toBe('https://gateway.example.com')
        expect(headers).toEqual({ 'X-Access': 'proxy-token' })
        return `ticket-${++mints}`
      }
    }

    const first = await resolveRemoteOauthTicket('https://gateway.example.com', { 'X-Access': 'proxy-token' }, deps)
    const second = await resolveRemoteOauthTicket('https://gateway.example.com', { 'X-Access': 'proxy-token' }, deps)
    expect(first).not.toBe(second)
    expect(mints).toBe(2)
  })

  it('classifies only confirmed auth rejection as reauth and retains the pre-mint session copy', async () => {
    for (const baseUrl of ['https://gateway.example.com', 'https://lab.agents.nousresearch.com']) {
      for (const hadNativeSession of [false, true]) {
        for (const statusCode of [401, 403, 500, 502, 503, 504, undefined]) {
          let nativeSession = hadNativeSession
          const cause = Object.assign(new Error('ticket request failed'), { statusCode })
          const error = await resolveRemoteOauthTicket(baseUrl, {}, {
            hasNativeSession: () => nativeSession,
            mintGatewayWsTicket: async () => {
              nativeSession = false
              throw cause
            }
          }).catch((failure: Error & { isCloudBackendDown?: boolean; statusCode?: number }) => failure)

          expect(error).toBeInstanceOf(Error)
          if (!(error instanceof Error)) {
            throw new Error('Expected ticket rejection')
          }
          expect(error.cause).toBe(cause)
          expect(error.statusCode).toBe(statusCode)
          const authRejected = statusCode === 401 || statusCode === 403
          expect(isReauthRequiredError(error)).toBe(authRejected)
          expect(error.isCloudBackendDown === true).toBe(
            baseUrl.includes('.agents.nousresearch.com') && [502, 503, 504].includes(statusCode ?? 0)
          )
          if (authRejected) {
            expect(error.message).toBe(oauthTicketFailureAuthMessage(hadNativeSession))
          }
        }
      }
    }
  })
})
