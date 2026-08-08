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

import { test, vi } from 'vitest'

import {
  canUseEmbeddedLoginFallback,
  classifyNativeLoginFailure,
  NativeLoginCoordinator,
  NativeLoginError,
  recoveryActionForNativeLoginFailure,
  runNativeLogin
} from './native-oauth-login'

// A fake http.Server: captures the request handler, lets the test drive a
// synthetic browser callback, and records listen/close lifecycle.
function makeFakeServerFactory(port = 51234) {
  const state: any = { closeCount: 0, handler: null, listening: false, closed: false, openedUrl: null }

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
      state.closeCount += 1
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

test('starting again cancels the prior gateway attempt and rejects its stale callback', async () => {
  const coordinator = new NativeLoginCoordinator()
  const first = makeFakeServerFactory(51234)
  const second = makeFakeServerFactory(51235)
  const openedUrls: string[] = []
  let firstTokenPosts = 0

  const firstPromise = coordinator.start('https://gw.example.com', {
    openExternal: async url => {
      openedUrls.push(url)
    },
    postJson: async () => {
      firstTokenPosts += 1

      return { access_token: 'stale-token' }
    },
    createServer: first.createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  const firstAuthorize = new URL(openedUrls[0])

  const secondPromise = coordinator.start('https://gw.example.com', {
    openExternal: async url => {
      openedUrls.push(url)
    },
    postJson: async () => ({ access_token: 'fresh-token' }),
    createServer: second.createServer,
    timeoutMs: 5_000
  })

  // Replacement is synchronous: the old listener is closed before the new
  // attempt can open a browser or receive a callback.
  assert.equal(first.state.closed, true)
  await assert.rejects(firstPromise, (error: any) => error?.code === 'superseded')

  await new Promise(r => setTimeout(r, 5))
  const secondAuthorize = new URL(openedUrls[1])

  assert.notEqual(secondAuthorize.searchParams.get('state'), firstAuthorize.searchParams.get('state'))
  assert.notEqual(secondAuthorize.searchParams.get('code_challenge'), firstAuthorize.searchParams.get('code_challenge'))
  assert.notEqual(secondAuthorize.searchParams.get('redirect_uri'), firstAuthorize.searchParams.get('redirect_uri'))

  // Even if a fake drives the now-closed old handler, stale state cannot be
  // redeemed or affect the fresh attempt.
  first.state.hitCallback(`code=old-code&state=${encodeURIComponent(firstAuthorize.searchParams.get('state') || '')}`)
  assert.equal(firstTokenPosts, 0)

  second.state.hitCallback(`code=new-code&state=${encodeURIComponent(secondAuthorize.searchParams.get('state') || '')}`)
  assert.equal((await secondPromise).accessToken, 'fresh-token')
  assert.equal(second.state.closed, true)
})

test('replacement before bind prevents the cancelled attempt from opening a tab', async () => {
  const coordinator = new NativeLoginCoordinator()
  const firstState: any = { closeCount: 0, listenCallback: null }

  const firstCreateServer: any = (_handler: any) => {
    const server: any = new EventEmitter()

    server.listen = (_port: number, _host: string, callback: () => void) => {
      firstState.listenCallback = callback
    }

    server.address = () => ({ address: '127.0.0.1', family: 'IPv4', port: 51234 })

    server.close = () => {
      firstState.closeCount += 1
    }

    return server
  }

  let staleTabs = 0

  const firstPromise = coordinator.start('https://gw.example.com', {
    openExternal: async () => {
      staleTabs += 1
    },
    postJson: async () => ({ access_token: 'stale-token' }),
    createServer: firstCreateServer,
    timeoutMs: 5_000
  })

  const second = makeFakeServerFactory(51235)

  const secondPromise = coordinator.start('https://gw.example.com', {
    openExternal: async () => undefined,
    postJson: async () => ({ access_token: 'fresh-token' }),
    createServer: second.createServer,
    timeoutMs: 5_000
  })

  await assert.rejects(firstPromise, (error: any) => error?.code === 'superseded')
  firstState.listenCallback()
  await new Promise(r => setTimeout(r, 0))
  assert.equal(staleTabs, 0)
  assert.equal(firstState.closeCount, 1)

  coordinator.cancel('https://gw.example.com')
  await assert.rejects(secondPromise, (error: any) => error?.code === 'cancelled')
})

test('explicit cancellation closes once and leaves no timeout work behind', async () => {
  vi.useFakeTimers()

  try {
    const coordinator = new NativeLoginCoordinator()
    const { createServer, state } = makeFakeServerFactory()
    let tokenPosts = 0

    const promise = coordinator.start('https://gw.example.com', {
      openExternal: async () => undefined,
      postJson: async () => {
        tokenPosts += 1

        return { access_token: 'should-not-exist' }
      },
      createServer,
      timeoutMs: 5_000
    })

    assert.equal(coordinator.cancel('https://gw.example.com'), true)
    assert.equal(state.closed, true)
    await assert.rejects(promise, (error: any) => error?.code === 'cancelled')

    await vi.advanceTimersByTimeAsync(5_001)
    assert.equal(state.closeCount, 1)
    assert.equal(tokenPosts, 0)
    assert.equal(vi.getTimerCount(), 0)
  } finally {
    vi.useRealTimers()
  }
})

test('replacement aborts an in-flight token exchange and ignores its late result', async () => {
  const coordinator = new NativeLoginCoordinator()
  const first = makeFakeServerFactory(51234)
  const second = makeFakeServerFactory(51235)
  let firstAuthorizeUrl = ''
  let exchangeSignal: AbortSignal | undefined
  let resolveExchange: ((body: any) => void) | undefined

  const firstPromise = coordinator.start('https://gw.example.com', {
    openExternal: async url => {
      firstAuthorizeUrl = url
    },
    postJson: async (_url, _body, opts: any) => {
      exchangeSignal = opts?.signal

      return new Promise(resolve => {
        resolveExchange = resolve
      })
    },
    createServer: first.createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  const firstState = new URL(firstAuthorizeUrl).searchParams.get('state') || ''
  first.state.hitCallback(`code=first-code&state=${encodeURIComponent(firstState)}`)
  await new Promise(r => setTimeout(r, 0))

  const secondPromise = coordinator.start('https://gw.example.com', {
    openExternal: async () => undefined,
    postJson: async () => ({ access_token: 'fresh-token' }),
    createServer: second.createServer,
    timeoutMs: 5_000
  })

  assert.equal(exchangeSignal?.aborted, true)
  await assert.rejects(firstPromise, (error: any) => error?.code === 'superseded')
  resolveExchange?.({ access_token: 'late-stale-token' })

  await new Promise(r => setTimeout(r, 0))
  assert.equal(first.state.closeCount, 1)

  coordinator.cancel('https://gw.example.com')
  await assert.rejects(secondPromise, (error: any) => error?.code === 'cancelled')
})

test('only the gateway stale-code contract is classified as a stale attempt', () => {
  const stale: any = new Error('400: {"detail":"Invalid or expired authorization code."}')
  stale.statusCode = 400

  assert.equal(classifyNativeLoginFailure(stale), 'stale_attempt')
  assert.equal(classifyNativeLoginFailure(new Error('400: invalid_grant from Google OAuth')), null)
  assert.equal(classifyNativeLoginFailure(new Error('Google returned a 400 stale request')), null)
  assert.equal(
    classifyNativeLoginFailure(
      Object.assign(new Error('401: Invalid or expired authorization code.'), { statusCode: 401 })
    ),
    null
  )
})

test('embedded compatibility fallback is limited to local native startup failures', async () => {
  const callbackFailure = new Error('Google OAuth returned an arbitrary 400 response')
  const { createServer, state } = makeFakeServerFactory()
  let authorizeUrl = ''

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async url => {
      authorizeUrl = url
    },
    postJson: async () => {
      throw callbackFailure
    },
    createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  const callbackState = new URL(authorizeUrl).searchParams.get('state') || ''
  state.hitCallback(`code=failed-code&state=${encodeURIComponent(callbackState)}`)

  const error = await promise.catch(reason => reason)

  assert.equal(classifyNativeLoginFailure(error), null)
  assert.equal(canUseEmbeddedLoginFallback(error), false)
  assert.equal(canUseEmbeddedLoginFallback(new Error('Could not bind native login listener')), false)
})

test('a listener failure after browser launch cannot open an embedded second attempt', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let opened = false

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async () => {
      opened = true
    },
    postJson: async () => ({ access_token: 'unused' }),
    createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  assert.equal(opened, true)
  state.server.emit('error', new Error('listener failed'))

  const error = await promise.catch(reason => reason)
  assert.equal(canUseEmbeddedLoginFallback(error), false)
  assert.equal(state.closeCount, 1)
})

test('recovery is user-driven only for bounded stale-attempt failures', () => {
  for (const code of ['cancelled', 'stale_attempt', 'state_mismatch', 'timeout'] as const) {
    assert.equal(recoveryActionForNativeLoginFailure(new NativeLoginError(code, code)), 'restart')
  }

  assert.equal(recoveryActionForNativeLoginFailure(new NativeLoginError('superseded', 'superseded')), 'ignore')
  assert.equal(recoveryActionForNativeLoginFailure(new Error('Google OAuth returned 400')), null)
  assert.equal(recoveryActionForNativeLoginFailure(new Error('network offline')), null)
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

  await assert.rejects(promise, (error: any) => {
    assert.match(error?.message || '', /state mismatch/i)
    assert.equal(error?.code, 'state_mismatch')

    return true
  })
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

  await assert.rejects(promise, (error: any) => {
    assert.match(error?.message || '', /access_denied/i)
    assert.equal(error?.code, 'cancelled')

    return true
  })
  assert.equal(state.closeCount, 1)
})

test('runNativeLogin times out when no callback arrives', async () => {
  const { createServer, state } = makeFakeServerFactory()

  await assert.rejects(
    runNativeLogin('https://gw.example.com', {
      openExternal: async () => undefined,
      postJson: async () => ({}),
      createServer,
      timeoutMs: 20
    }),
    (error: any) => {
      assert.match(error?.message || '', /timed out/i)
      assert.equal(error?.code, 'timeout')

      return true
    }
  )
  assert.equal(state.closeCount, 1)
})

test('runNativeLogin classifies the gateway expired-code contract and cleans up', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let authorizeUrl = ''

  const promise = runNativeLogin('https://gw.example.com', {
    openExternal: async url => {
      authorizeUrl = url
    },
    postJson: async () => {
      const error: any = new Error('400: {"detail":"Invalid or expired authorization code."}')
      error.statusCode = 400
      throw error
    },
    createServer,
    timeoutMs: 5_000
  })

  await new Promise(r => setTimeout(r, 5))
  const callbackState = new URL(authorizeUrl).searchParams.get('state') || ''
  state.hitCallback(`code=expired-code&state=${encodeURIComponent(callbackState)}`)

  await assert.rejects(promise, (error: any) => error?.code === 'stale_attempt')
  assert.equal(state.closeCount, 1)
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
    (error: any) => {
      assert.match(error?.message || '', /could not open the system browser/i)
      assert.equal(canUseEmbeddedLoginFallback(error), true)

      return true
    }
  )
})
