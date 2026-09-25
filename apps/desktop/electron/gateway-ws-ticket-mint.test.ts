/**
 * The ws-ticket mint exactly as main.ts wires it (mintGatewayWsTicketWithRetries
 * over the production native-token coordinator), read through the same
 * gatewayTicketFailure wrapper that decides whether the app latches the
 * sign-in overlay. Plus a source guard: a merge once silently restored an
 * older main.ts that swallowed coordinator errors (`.catch(() => null)`) and
 * fell back to the cookie jar, turning a Cloud 429 into a sign-out.
 */

import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { cloudExchangeRateLimitedError, isCloudRateLimited } from './cloud-auth-errors'
import { gatewayTicketFailure, normalizeRemoteBaseUrl } from './connection-config'
import { createNativeAccessTokenCoordinator, NativeAuthChangedError } from './native-access-token'
import { type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'
import { createNativeTokenCoordinatorDeps } from './native-token-coordinator-deps'
import { mintGatewayWsTicketWithRetries } from './oauth-rest-request'

const AGENT_URL = 'https://agent-1.cloud.example.test'
const NOW = 1_000

const agentSet = (overrides: Partial<NativeTokenSet> = {}): NativeTokenSet => ({
  accessToken: 'AGENT-AT-1',
  refreshToken: '',
  expiresAt: 10_000,
  provider: 'hermes-cloud-agent',
  userId: 'agt_1',
  ...overrides
})

interface Scenario {
  stored: NativeTokenSet
  exchange: () => Promise<NativeTokenSet>
  bindingCurrent?: () => boolean
  cookieWorks?: boolean
  mintWithBearer: (bearer: string) => Promise<unknown>
}

async function mintThroughMainWiring(scenario: Scenario) {
  const store = new Map<string, NativeTokenSet>([[AGENT_URL, scenario.stored]])
  const calls = { exchanges: 0, cookieMints: 0, bearerMints: [] as string[], cleared: 0, sleeps: 0 }

  const coordinator = createNativeAccessTokenCoordinator(
    createNativeTokenCoordinatorDeps({
      loadTokens: url => store.get(url) ?? null,
      storeTokens: (url, tokens) => void store.set(url, tokens),
      clearTokens: url => {
        calls.cleared++
        store.delete(url)
      },
      normalizeBaseUrl: normalizeRemoteBaseUrl,
      nowSeconds: () => NOW,
      tokenNeedsRefresh,
      refreshGatewayTokens: async () => {
        throw new Error('a cloud agent never uses /auth/native/refresh')
      },
      isCloudAgentUrl: url => url === AGENT_URL,
      confirmedCloudAgentId: async () => 'agt_1',
      isCloudBindingCurrent: () => scenario.bindingCurrent?.() ?? true,
      exchangeForAgent: async () => {
        calls.exchanges++

        return scenario.exchange()
      },
      hasLivePortalSession: () => true
    })
  )

  let ticket: string | undefined
  let failure: any

  try {
    ticket = await mintGatewayWsTicketWithRetries(
      AGENT_URL,
      {
        ensureNativeAccessToken: coordinator.ensure,
        fetchJson: async (_url, _token, options) => {
          calls.bearerMints.push(options.bearer)

          return scenario.mintWithBearer(options.bearer)
        },
        fetchJsonViaOauthSession: async () => {
          calls.cookieMints++

          if (scenario.cookieWorks) {
            return { ticket: 'COOKIE-TICKET' }
          }

          // Cloud connections never have a cookie session: an empty jar 401s.
          throw httpStatusError(401, 'no_cookie')
        }
      },
      {},
      {
        sleep: async () => {
          calls.sleeps++
        }
      }
    )
  } catch (error) {
    failure = gatewayTicketFailure(error, 'AUTH', 'TRANSPORT')
  }

  return { ticket, failure, calls, stored: store.get(AGENT_URL) }
}

test('a Cloud exchange 429 is a temporary failure, never a sign-in latch', async () => {
  const { ticket, failure, calls, stored } = await mintThroughMainWiring({
    stored: agentSet({ expiresAt: 1 }),
    exchange: async () => {
      throw cloudExchangeRateLimitedError(30)
    },
    mintWithBearer: async () => ({ ticket: 'never' })
  })

  expect(ticket).toBeUndefined()
  expect(failure.isReauthRequired).toBeUndefined()
  expect(failure.needsOauthLogin).toBeUndefined()
  expect(failure.message).toBe('TRANSPORT')
  expect(failure.statusCode).toBe(429)
  expect(isCloudRateLimited(failure.cause)).toBe(true)
  // Not hammered: one exchange, no transient retry loop, and the set is kept.
  expect(calls.exchanges).toBe(1)
  expect(calls.sleeps).toBe(0)
  expect(calls.cleared).toBe(0)
  expect(stored?.accessToken).toBe('AGENT-AT-1')
})

test('a structured 401 on an unexpired bearer forces exactly one re-exchange, then mints', async () => {
  const { ticket, failure, calls, stored } = await mintThroughMainWiring({
    stored: agentSet(),
    exchange: async () => agentSet({ accessToken: 'AGENT-AT-2' }),
    mintWithBearer: async bearer => {
      if (bearer === 'AGENT-AT-1') {
        throw httpStatusError(401, JSON.stringify({ error: 'invalid_token' }))
      }

      return { ticket: `TICKET-FOR-${bearer}` }
    }
  })

  expect(failure).toBeUndefined()
  expect(ticket).toBe('TICKET-FOR-AGENT-AT-2')
  expect(calls.exchanges).toBe(1)
  expect(calls.bearerMints).toEqual(['AGENT-AT-1', 'AGENT-AT-2'])
  expect(calls.cookieMints).toBe(0)
  expect(stored?.accessToken).toBe('AGENT-AT-2')
})

test('a binding change mid-exchange never crosses into the cookie jar or latches sign-in', async () => {
  for (const cookieWorks of [true, false]) {
    const { ticket, failure, calls } = await mintThroughMainWiring({
      stored: agentSet({ expiresAt: 1 }),
      exchange: async () => agentSet({ accessToken: 'AGENT-AT-2' }),
      bindingCurrent: () => false,
      cookieWorks,
      mintWithBearer: async () => ({ ticket: 'never' })
    })

    expect(ticket).toBeUndefined()
    expect(failure.cause).toBeInstanceOf(NativeAuthChangedError)
    expect(failure.isReauthRequired).toBeUndefined()
    expect(failure.needsOauthLogin).toBeUndefined()
    expect(calls.cookieMints).toBe(0)
    expect(calls.sleeps).toBe(0)
  }
})

test('main.ts routes every native-token request through the non-swallowing OAuth helpers', () => {
  const source = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8')

  // No site may swallow a coordinator error and silently downgrade to cookies.
  expect(source.match(/ensureNativeAccessToken\([^)]*\)\s*\.catch\(/g) ?? []).toEqual([])
  // The options-accepting coordinator (forceRefresh) is what the helpers get.
  expect(/const ensureNativeAccessToken = nativeAccessTokenCoordinator\.ensure\b/.test(source)).toBe(true)

  const mint = source.match(/async function mintGatewayWsTicket\([^)]*\)[^{]*\{([\s\S]*?)\n\}/)
  expect(mint?.[1]).toMatch(/mintGatewayWsTicketWithRetries\(/)
  expect(mint?.[1]).not.toMatch(/ensureNativeAccessToken\(/)

  // Readiness probe, file download, and fetchJsonForBackend (REST/api/status).
  expect(source.match(/requestWithOauthFallback\(/g)?.length ?? 0).toBeGreaterThanOrEqual(3)
})
