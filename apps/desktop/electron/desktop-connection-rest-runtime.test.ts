import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopConnectionRestRuntime } from './desktop-connection-rest-runtime'

test('descriptor REST preserves native bearer, static token, and OAuth upload rejection', async () => {
  const calls: Array<{ url: string; token: unknown; options: any }> = []

  const runtime = createDesktopConnectionRestRuntime({
    ensureNativeAccessToken: async () => 'bearer-token',
    fetchJson: async (url: string, token: unknown, options: any) => {
      calls.push({ url, token, options })

      return { ok: true }
    },
    fetchJsonViaOauthSession: async () => {
      throw new Error('unexpected OAuth cookie fallback')
    }
  })

  const oauth = { baseUrl: 'https://portal.example', authMode: 'oauth', headers: { 'X-Test': 'yes' } }
  const token = { baseUrl: 'https://gateway.example', authMode: 'token', token: 'static-token', headers: {} }

  await runtime.getJsonForBackend(oauth, '/health')
  await runtime.postJsonForBackend(token, '/run', null)
  await assert.rejects(runtime.fetchJsonForBackend(oauth, '/upload', { upload: {} }), /not supported/)

  assert.deepEqual(calls, [
    {
      url: 'https://portal.example/health',
      token: null,
      options: { method: undefined, body: undefined, timeoutMs: undefined, headers: oauth.headers, bearer: 'bearer-token' }
    },
    {
      url: 'https://gateway.example/run',
      token: 'static-token',
      options: { method: 'POST', body: {}, upload: undefined, timeoutMs: undefined, headers: token.headers }
    }
  ])
})
