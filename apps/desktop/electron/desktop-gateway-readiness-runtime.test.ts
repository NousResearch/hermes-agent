import { expect, test } from 'vitest'

import { createDesktopGatewayReadinessRuntime } from './desktop-gateway-readiness-runtime'

test('credentialed readiness probes the exact gateway with request headers', async () => {
  const calls: any[] = []

  const runtime = createDesktopGatewayReadinessRuntime({
    fetchJson: async (url: string, token: string | null, options: any) => {
      calls.push({ url, token, options })

      return { ok: true }
    },
    fetchPublicJson: async () => {
      throw new Error('credentialed gateway must not use public probe')
    },
    fetchJsonViaOauthSession: async () => ({ ok: true }),
    ensureNativeAccessToken: async () => 'unused'
  })

  await runtime.waitForHermes('https://gateway.example', 'pin', undefined, 'token', { 'X-Scope': 'fleet' })
  expect(calls).toHaveLength(1)
  expect(calls[0]).toMatchObject({
    url: 'https://gateway.example/api/health',
    token: 'pin',
    options: { headers: { 'X-Scope': 'fleet' } }
  })
})

test('provider metadata is sanitized and cached per gateway', async () => {
  let fetches = 0

  const runtime = createDesktopGatewayReadinessRuntime({
    fetchJson: async () => ({ ok: true }),
    fetchPublicJson: async () => {
      fetches += 1

      return { providers: [null, { name: 'password', supports_password: true }, { name: '' }] }
    },
    fetchJsonViaOauthSession: async () => ({ ok: true }),
    ensureNativeAccessToken: async () => 'unused'
  })

  expect(await runtime.gatewayAuthProviders('https://gateway.example')).toEqual([
    { name: 'password', supportsPassword: true }
  ])
  expect(await runtime.gatewayAuthProviders('https://gateway.example')).toHaveLength(1)
  expect(fetches).toBe(1)
})
