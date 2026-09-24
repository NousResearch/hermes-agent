/**
 * §4 discovery with the desktop bearer: GET {portal}/api/agents with
 * `Authorization: Bearer <portal AT>`, one forced refresh + retry on 401,
 * then a needsCloudLogin error. No cookies, no `?org=` (the bearer pins it).
 */

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { discoverCloudAgentsWithBearer, trimCloudAgents } from './cloud-discovery'

const PORTAL = 'https://portal.example.test'

function jwt(claims: Record<string, unknown>) {
  const enc = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url')

  return `${enc({ alg: 'none' })}.${enc(claims)}.sig`
}

function makeDeps(responses: Array<(bearer: string) => unknown>, tokens: Array<null | string>) {
  const requests: Array<{ url: string; bearer: string; method: string }> = []
  const tokenCalls: any[] = []

  return {
    requests,
    tokenCalls,
    deps: {
      portalBaseUrl: PORTAL,
      getAccessToken: async (options?: any) => {
        tokenCalls.push(options ?? {})

        return tokens.shift() ?? null
      },
      fetchJson: async (url: string, _token: null, options: any) => {
        requests.push({ url, bearer: options.bearer, method: options.method })
        const respond = responses.shift()

        if (!respond) {
          throw new Error('unexpected request')
        }

        return respond(options.bearer)
      }
    }
  }
}

const AGENTS = {
  agents: [
    {
      id: 'agt_1',
      name: 'One',
      status: 'running',
      dashboardUrl: 'https://a1.example.test',
      dashboardGatewayState: 'ready',
      secret: 'x'
    }
  ]
}

test('discovery sends the portal bearer to /api/agents with no org query and trims the rows', async () => {
  const at = jwt({ org_id: 'org_9' })
  const { deps, requests } = makeDeps([() => AGENTS], [at])

  await expect(discoverCloudAgentsWithBearer(deps)).resolves.toEqual({
    agents: [
      {
        id: 'agt_1',
        name: 'One',
        status: 'running',
        dashboardUrl: 'https://a1.example.test',
        dashboardGatewayState: 'ready'
      }
    ],
    org: { id: 'org_9', slug: null, name: 'org_9', isPersonal: false, role: 'MEMBER' }
  })
  expect(requests).toEqual([{ url: `${PORTAL}/api/agents`, bearer: at, method: 'GET' }])
})

test('an org echoed by the portal wins over the token claim', async () => {
  const { deps } = makeDeps(
    [() => ({ ...AGENTS, org: { id: 'org_9', slug: 'acme', name: 'Acme', isPersonal: false, role: 'ADMIN' } })],
    [jwt({ org_id: 'org_9' })]
  )

  await expect(discoverCloudAgentsWithBearer(deps)).resolves.toMatchObject({
    org: { id: 'org_9', slug: 'acme', name: 'Acme', role: 'ADMIN' }
  })
})

test('a 401 forces ONE refresh of the rejected token and retries once', async () => {
  const { deps, requests, tokenCalls } = makeDeps(
    [
      () => {
        throw httpStatusError(401, JSON.stringify({ error: 'invalid_token' }))
      },
      () => AGENTS
    ],
    ['AT-1', 'AT-2']
  )

  await expect(discoverCloudAgentsWithBearer(deps)).resolves.toMatchObject({ agents: [{ id: 'agt_1' }] })
  expect(requests.map(r => r.bearer)).toEqual(['AT-1', 'AT-2'])
  expect(tokenCalls).toEqual([{}, { forceRefresh: true, rejectedAccessToken: 'AT-1' }])
})

test('a second 401 (or a failed forced refresh) surfaces the existing needsCloudLogin "expired" error', async () => {
  const unauthorized = () => {
    throw httpStatusError(401, 'nope')
  }

  const twice = makeDeps([unauthorized, unauthorized], ['AT-1', 'AT-2'])
  const error = await discoverCloudAgentsWithBearer(twice.deps).catch(e => e)
  expect(error).toMatchObject({ needsCloudLogin: true })
  expect(error.message).toBe('Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.')
  expect(twice.requests).toHaveLength(2)

  const refreshFailed = makeDeps([unauthorized], ['AT-1', null])
  await expect(discoverCloudAgentsWithBearer(refreshFailed.deps)).rejects.toMatchObject({ needsCloudLogin: true })
  expect(refreshFailed.requests).toHaveLength(1)
})

test('a forced refresh that hands back the SAME rejected token is not retried: needsCloudLogin', async () => {
  const { deps, requests, tokenCalls } = makeDeps(
    [
      () => {
        throw httpStatusError(401, 'nope')
      },
      () => AGENTS
    ],
    ['AT-1', 'AT-1']
  )

  await expect(discoverCloudAgentsWithBearer(deps)).rejects.toMatchObject({ needsCloudLogin: true })
  expect(requests).toHaveLength(1)
  expect(tokenCalls).toHaveLength(2)
})

test('no portal token at all is the existing "not signed in" needsCloudLogin error, without a request', async () => {
  const { deps, requests } = makeDeps([], [null])
  const error = await discoverCloudAgentsWithBearer(deps).catch(e => e)

  expect(error).toMatchObject({ needsCloudLogin: true })
  expect(error.message).toBe(
    'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.'
  )
  expect(requests).toHaveLength(0)
})

test('non-auth failures keep their own meaning', async () => {
  const { deps } = makeDeps(
    [
      () => {
        throw httpStatusError(503, 'down')
      }
    ],
    ['AT-1']
  )

  await expect(discoverCloudAgentsWithBearer(deps)).rejects.toMatchObject({ statusCode: 503 })
})

test('trimCloudAgents drops malformed rows and defaults missing fields', () => {
  expect(trimCloudAgents({ agents: [null, { name: 'no id' }, { id: 'a' }] })).toEqual([
    { id: 'a', name: 'a', status: 'unknown', dashboardUrl: null, dashboardGatewayState: 'unknown' }
  ])
  expect(trimCloudAgents(null)).toEqual([])
})
