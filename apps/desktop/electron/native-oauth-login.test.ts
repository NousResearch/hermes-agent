/**
 * Tests for electron/native-oauth-login.ts — the loopback-listener
 * orchestration of the RFC 8252 native login, with all I/O injected (fake
 * http server, fake openExternal, fake token POST) so no real socket or
 * browser is needed.
 *
 * Run with: node --test electron/native-oauth-login.test.ts
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { runNativeLogin } from './native-oauth-login'

// A fake http.Server: captures the request handler, lets the test drive a
// synthetic browser callback, and records listen/close lifecycle.
function makeFakeServerFactory(port = 51234) {
  const state: any = { handler: null, listening: false, closed: false, openedUrl: null }

  const createServer: any = (handler: any) => {
    state.handler = handler
    const server: any = new EventEmitter()
    server.listen = (_port: number, _host: string, cb: () => void) => {
      state.listening = true
      cb()
    }
    server.address = () => ({ address: '127.0.0.1', family: 'IPv4', port })
    server.close = () => {
      state.closed = true
    }
    state.server = server

    return server
  }

  // Drive a synthetic browser hit to the loopback callback.
  state.hitCallback = (query: string) => {
    const res: any = { writeHead: () => undefined, end: () => undefined }
    state.handler({ url: `/callback?${query}` }, res)
  }

  return { createServer, state }
}

test('runNativeLogin completes the loopback round trip and returns tokens', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let capturedAuthorizeUrl = ''
  let tokenPostBody: any = null

  const promise = runNativeLogin(
    'https://gw.example.com',
    {
      openExternal: async url => {
        capturedAuthorizeUrl = url
      },
      postJson: async (_url, body) => {
        tokenPostBody = body

        return {
          access_token: 'AT-native',
          refresh_token: 'RT-native',
          token_type: 'Bearer',
          expires_at: 1893456000,
          provider: 'nous',
          user_id: 'u-9'
        }
      },
      createServer,
      timeoutMs: 5_000
    },
    { provider: 'nous' }
  )

  // Give the listen callback a tick to open the browser + capture the URL.
  await new Promise(r => setTimeout(r, 5))

  // The authorize URL must carry OUR challenge + loopback redirect + state.
  const authorize = new URL(capturedAuthorizeUrl)
  assert.equal(authorize.pathname, '/auth/native/authorize')
  const challenge = authorize.searchParams.get('code_challenge')
  const stateParam = authorize.searchParams.get('state')
  assert.ok(challenge && challenge.length > 0)
  assert.match(authorize.searchParams.get('redirect_uri') || '', /^http:\/\/127\.0\.0\.1:\d+\/callback$/)

  // Synthetic browser redirect back with the matching state + a code.
  state.hitCallback(`code=gw-code-1&state=${encodeURIComponent(stateParam!)}`)

  const tokens = await promise
  assert.equal(tokens.accessToken, 'AT-native')
  assert.equal(tokens.refreshToken, 'RT-native')
  assert.equal(tokens.userId, 'u-9')
  // The token POST carried the code + a verifier whose hash is the challenge.
  assert.equal(tokenPostBody.code, 'gw-code-1')
  assert.ok(tokenPostBody.code_verifier && tokenPostBody.code_verifier.length >= 43)
  // Listener was cleaned up.
  assert.equal(state.closed, true)
})

test('runNativeLogin rejects on a state mismatch (CSRF) without redeeming', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let tokenPostCalled = false

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async () => undefined,
    postJson: async () => {
      tokenPostCalled = true

      return {}
    },
    createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  // Wrong state — must not redeem the code.
  state.hitCallback('code=evil&state=not-the-real-state')

  await assert.rejects(promise, /state mismatch/i)
  assert.equal(tokenPostCalled, false)
  assert.equal(state.closed, true)
})

test('runNativeLogin surfaces a gateway error param', async () => {
  const { createServer, state } = makeFakeServerFactory()

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async () => undefined,
    postJson: async () => ({}),
    createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  state.hitCallback('error=access_denied&error_description=user_declined')

  await assert.rejects(promise, /access_denied/i)
})

test('runNativeLogin times out when no callback arrives', async () => {
  const { createServer } = makeFakeServerFactory()

  await assert.rejects(
    runNativeLogin('https://gw.example.com', {
      openExternal: async () => undefined,
      postJson: async () => ({}),
      createServer,
      timeoutMs: 20
    }),
    /timed out/i
  )
})

test('runNativeLogin fails if the browser cannot be opened', async () => {
  const { createServer } = makeFakeServerFactory()

  await assert.rejects(
    runNativeLogin('https://gw.example.com', {
      openExternal: async () => {
        throw new Error('no browser')
      },
      postJson: async () => ({}),
      createServer,
      timeoutMs: 5_000
    }),
    /could not open the system browser/i
  )
})
