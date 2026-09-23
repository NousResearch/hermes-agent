import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopConnectionAuthRuntime } from './desktop-connection-auth-runtime'

test('native token exchange sends an object body through the shared JSON transport', async () => {
  const requests: any[] = []

  const runtime = createDesktopConnectionAuthRuntime({
    app: { getPath: () => '', isReady: () => true },
    BrowserWindow: class {},
    fetchJson: async (_url, _token, options) => {
      requests.push(options)

      return { ok: true }
    },
    fetchJsonViaOauthSession: async () => ({}),
    encryptDesktopSecret: value => ({ encoding: 'plain', value }),
    decryptDesktopSecret: secret => secret?.value || '',
    ensureBackend: async () => ({ baseUrl: '', wsUrl: '', authMode: 'token', headers: {} }),
    getOauthSession: () => null,
    warmOauthCookieStore: async () => {},
    hasOauthSessionCookie: async () => false,
    openOauthLoginWindow: async () => {},
    rememberRemoteWsHeaders: () => {},
    rememberLog: () => {}
  })

  const body = { refresh_token: 'opaque-token', provider: 'nous' }

  await runtime.postJsonNoAuth('https://gateway.example/auth/native/refresh', body)

  assert.equal(requests.length, 1)
  assert.equal(requests[0].method, 'POST')
  assert.equal(requests[0].body, body)
})
