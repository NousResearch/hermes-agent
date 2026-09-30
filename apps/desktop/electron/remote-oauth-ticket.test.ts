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

          const error = await resolveRemoteOauthTicket(
            baseUrl,
            {},
            {
              hasNativeSession: () => nativeSession,
              mintGatewayWsTicket: async () => {
                nativeSession = false
                throw cause
              }
            }
          ).catch((failure: Error & { isCloudBackendDown?: boolean; statusCode?: number }) => failure)

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

  it('distinguishes transport failure classes in user-facing copy (timeout vs refused vs other)', async () => {
    const mintThrowing = (cause: unknown) =>
      resolveRemoteOauthTicket('https://gateway.example.com', {}, {
        hasNativeSession: () => false,
        mintGatewayWsTicket: async () => {
          throw cause
        }
      }).catch((failure: Error) => failure)

    const timeout = await mintThrowing(Object.assign(new Error('timeout of 8000ms exceeded'), { code: 'ETIMEDOUT' }))
    const refused = await mintThrowing(Object.assign(new Error('connect ECONNREFUSED 10.0.0.5:8446'), { code: 'ECONNREFUSED' }))
    const dns = await mintThrowing(Object.assign(new Error('getaddrinfo ENOTFOUND gw.example.com'), { code: 'ENOTFOUND' }))
    const ambiguous = await mintThrowing(new Error('socket hang up'))
    const http500 = await mintThrowing(Object.assign(new Error('500: upstream'), { statusCode: 500 }))

    expect(timeout).toBeInstanceOf(Error)
    expect(refused).toBeInstanceOf(Error)
    expect(dns).toBeInstanceOf(Error)
    expect(ambiguous).toBeInstanceOf(Error)
    expect(http500).toBeInstanceOf(Error)

    if (!(timeout instanceof Error && refused instanceof Error && dns instanceof Error && ambiguous instanceof Error && http500 instanceof Error)) {
      throw new Error('Expected transport rejections')
    }

    // Timeout / DNS / unreachable all name the network path.
    expect(timeout.message).toContain('timed out or the host could not be resolved')
    expect(dns.message).toContain('timed out or the host could not be resolved')
    // Refused names the stopped-gateway case.
    expect(refused.message).toContain('connection was refused')
    expect(refused.message).not.toContain('timed out')
    // Ambiguous / unknown transport keeps the legacy one-liner.
    expect(ambiguous.message).toBe(
      'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.'
    )
    expect(http500.message).toBe(
      'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.'
    )

    // None of the transport classes may ever read as an auth failure.
    for (const err of [timeout, refused, dns, ambiguous, http500]) {
      expect(isReauthRequiredError(err)).toBe(false)
      expect((err as Error & { needsOauthLogin?: boolean }).needsOauthLogin).toBeUndefined()
    }
  })

  it('names the OAuth/basic-auth mismatch when the gateway only advertises password providers', async () => {
    // The pre-mint auth-provider probe is best-effort copy input: when it
    // reports a password-only backend, a rejected mint must point at the
    // auth-mode switch instead of the "Sign in" loop (#105632).
    const cause = Object.assign(new Error('ticket request failed'), { statusCode: 401 })
    const mismatch = /only offers username\/password sign-in[\s\S]*Session token/

    const failure = await resolveRemoteOauthTicket(
      'https://gateway.example.com',
      {},
      {
        hasNativeSession: () => true,
        mintGatewayWsTicket: async () => {
          throw cause
        },
        advertisedAuthProviders: async () => [{ name: 'basic', supports_password: true }]
      }
    ).catch((error: Error) => error)

    expect(failure).toBeInstanceOf(Error)
    expect((failure as Error & { isReauthRequired?: boolean }).isReauthRequired).toBe(true)
    expect((failure as Error).message).toMatch(mismatch)

    // A mixed provider list keeps the strict OAuth copy, and a throwing
    // probe must never change the auth-failure classification.
    const strict = await resolveRemoteOauthTicket(
      'https://gateway.example.com',
      {},
      {
        hasNativeSession: () => false,
        mintGatewayWsTicket: async () => {
          throw cause
        },
        advertisedAuthProviders: async () => {
          throw new Error('providers probe unreachable')
        }
      }
    ).catch((error: Error) => error)

    expect(strict).toBeInstanceOf(Error)
    expect((strict as Error).message).toMatch(/not signed in/)
    expect((strict as Error & { isReauthRequired?: boolean }).isReauthRequired).toBe(true)
  })
})
