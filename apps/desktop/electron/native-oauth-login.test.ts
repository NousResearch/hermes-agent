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

import { NativeLoginCancelledError } from './native-oauth'
import { nativeLoginFailureResult, runLoopbackAuthorization, runNativeLogin } from './native-oauth-login'

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

  // Drive a synthetic browser hit to the loopback callback. Returns what the
  // browser was answered with.
  state.hit = (url: string) => {
    const reply: { status: number; body: string } = { status: 0, body: '' }

    const res: any = {
      writeHead: (status: number) => {
        reply.status = status
      },
      end: (body = '') => {
        reply.body = body
      }
    }

    state.handler({ url }, res)

    return reply
  }

  state.hitCallback = (query: string) => state.hit(`/callback?${query}`)

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

const tick = () => new Promise(r => setTimeout(r, 5))

function startGatewayLogin(extra: Record<string, unknown> = {}) {
  const { createServer, state } = makeFakeServerFactory()
  let opened = ''
  let redeemed = 0

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async url => {
      opened = url
    },
    postJson: async () => {
      redeemed++

      return { access_token: 'AT-ok', refresh_token: 'RT-ok' }
    },
    createServer,
    timeoutMs: 5_000,
    ...extra
  })

  let outcome: 'pending' | 'resolved' | 'rejected' = 'pending'
  promise.then(
    () => (outcome = 'resolved'),
    () => (outcome = 'rejected')
  )

  return {
    promise,
    state,
    realState: () => new URL(opened).searchParams.get('state')!,
    redeemed: () => redeemed,
    outcome: () => outcome
  }
}

test('a callback with a mismatched state is answered 400 and IGNORED: the login keeps waiting (no DoS)', async () => {
  const login = startGatewayLogin()
  await tick()

  const reply = login.state.hitCallback('code=evil&state=not-the-real-state')
  await tick()

  assert.equal(reply.status, 400)
  // The success page is rendered only after the state check.
  assert.doesNotMatch(reply.body, /signed in/i)
  assert.equal(login.redeemed(), 0)
  assert.equal(login.outcome(), 'pending')
  assert.equal(login.state.closed, false)

  // The genuine redirect still completes the sign-in.
  const good = login.state.hitCallback(`code=real&state=${login.realState()}`)
  assert.equal(good.status, 200)
  assert.match(good.body, /signed in/i)
  assert.equal((await login.promise).accessToken, 'AT-ok')
  assert.equal(login.redeemed(), 1)
})

test('stateless and foreign-state error callbacks are ignored, never a cancel or a failure', async () => {
  const login = startGatewayLogin()
  await tick()

  for (const query of ['error=access_denied&error_description=user_declined', 'error=access_denied&state=attacker']) {
    assert.equal(login.state.hitCallback(query).status, 400)
  }

  await tick()
  assert.equal(login.outcome(), 'pending')
  login.state.hitCallback(`code=real&state=${login.realState()}`)
  assert.equal((await login.promise).accessToken, 'AT-ok')
})

test('only the /callback path is honoured; any other path is 404 and ignored', async () => {
  const login = startGatewayLogin()
  await tick()

  assert.equal(login.state.hit(`/favicon.ico`).status, 404)
  assert.equal(login.state.hit(`/other?code=x&state=${login.realState()}`).status, 404)
  await tick()
  assert.equal(login.redeemed(), 0)
  assert.equal(login.outcome(), 'pending')
  login.state.hitCallback(`code=real&state=${login.realState()}`)
  await login.promise
})

test('a Deny (error=access_denied with our state) rejects as a typed cancel with a cancelled page', async () => {
  const login = startGatewayLogin()
  await tick()

  const reply = login.state.hitCallback(`error=access_denied&state=${login.realState()}`)

  const error = await login.promise.catch(e => e)
  assert.ok(error instanceof NativeLoginCancelledError)
  assert.match(error.message, /access_denied/)
  assert.match(reply.body, /sign-in cancelled/i)
  assert.equal(login.state.closed, true)
})

test('any other error with our state is a failure with a neutral "did not complete" page', async () => {
  const login = startGatewayLogin()
  await tick()

  const reply = login.state.hitCallback(`error=server_error&state=${login.realState()}`)

  const error = await login.promise.catch(e => e)
  assert.ok(!(error instanceof NativeLoginCancelledError))
  assert.match(error.message, /server_error/)
  assert.match(reply.body, /did not complete/i)
  assert.doesNotMatch(reply.body, /cancelled/i)
})

test('a callback with our state but neither code nor error gets the "did not complete" page, not success', async () => {
  const login = startGatewayLogin()
  await tick()

  const reply = login.state.hitCallback(`state=${login.realState()}`)

  const error = await login.promise.catch(e => e)
  assert.match(error.message, /missing authorization code/i)
  assert.equal(reply.status, 200)
  assert.match(reply.body, /did not complete/i)
  assert.doesNotMatch(reply.body, /signed in/i)
  assert.equal(login.redeemed(), 0)
})

test('aborting the signal cancels a pending login and tears the listener down', async () => {
  const controller = new AbortController()
  const login = startGatewayLogin({ signal: controller.signal })
  await tick()

  controller.abort()

  const error = await login.promise.catch(e => e)
  assert.ok(error instanceof NativeLoginCancelledError)
  assert.equal(login.state.closed, true)
  assert.equal(login.redeemed(), 0)
})

test('an already-aborted signal never opens the browser', async () => {
  const controller = new AbortController()
  controller.abort()
  let opened = false
  const { createServer } = makeFakeServerFactory()

  const error = await runLoopbackAuthorization(
    {
      openExternal: async () => {
        opened = true
      },
      createServer,
      signal: controller.signal
    },
    { buildAuthorizeUrl: () => 'https://idp.example/authorize', redeem: async () => 'x' }
  ).catch(e => e)

  assert.ok(error instanceof NativeLoginCancelledError)
  assert.equal(opened, false)
})

test('the authorize URL is reported to onAuthorizeUrl so the UI can offer a copy-link fallback', async () => {
  const reported: string[] = []
  const login = startGatewayLogin({ onAuthorizeUrl: (url: string) => reported.push(url) })
  await tick()

  assert.equal(reported.length, 1)
  assert.equal(new URL(reported[0]).searchParams.get('state'), login.realState())
  login.state.hitCallback(`code=real&state=${login.realState()}`)
  await login.promise
})

test('nativeLoginFailureResult maps a browser Deny to a quiet cancel and anything else to an error', () => {
  assert.deepEqual(nativeLoginFailureResult(new NativeLoginCancelledError()), {
    ok: false,
    connected: false,
    cancelled: true
  })
  assert.deepEqual(nativeLoginFailureResult(new Error('boom')), { ok: false, connected: false, error: 'boom' })
})

test('runLoopbackAuthorization hands redeem the exact redirect_uri it authorized with', async () => {
  const { createServer, state } = makeFakeServerFactory(40404)
  let authorizeRedirect = ''
  let authorizeState = ''
  let redeemed: any = null

  const promise = runLoopbackAuthorization(
    { openExternal: async () => undefined, createServer, timeoutMs: 5_000 },
    {
      buildAuthorizeUrl: ({ redirectUri, state: s }) => {
        authorizeRedirect = redirectUri
        authorizeState = s

        return `https://idp.example/authorize?state=${s}`
      },
      redeem: async params => {
        redeemed = params

        return 'ok'
      }
    }
  )

  await new Promise(r => setTimeout(r, 5))
  state.hitCallback(`code=C1&state=${authorizeState}`)
  assert.equal(await promise, 'ok')
  assert.equal(authorizeRedirect, 'http://127.0.0.1:40404/callback')
  assert.equal(redeemed.redirectUri, authorizeRedirect)
  assert.equal(redeemed.code, 'C1')
  assert.ok(redeemed.verifier.length >= 43)
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
