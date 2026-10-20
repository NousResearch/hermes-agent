import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { isGatewayAuthRejection } from './connection-config'
import { NativeAuthChangedError } from './native-access-token'
import {
  canShowInteractiveOauthLogin,
  mintGatewayWsTicket,
  requestWithOauthFallback,
  retryCookie401WithLogin,
  shouldReplayAfterCookie401,
  withoutInteractiveOauthLogin
} from './oauth-rest-request'

const GATE_401 = () =>
  httpStatusError(
    401,
    JSON.stringify({ error: 'unauthenticated', detail: 'Unauthorized', reason: 'no_cookie', login_url: '/login' })
  )

test('a cookie 401 is replayed only for a gate refusal on an idempotent or vouched operation', () => {
  // Pre-auth gate refusal + idempotent method → replay.
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'GET' })).toBe(true)
  expect(shouldReplayAfterCookie401(GATE_401(), {})).toBe(true)
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'HEAD' })).toBe(true)
  expect(
    shouldReplayAfterCookie401(
      httpStatusError(401, JSON.stringify({ error: 'session_expired', reason: 'invalid_or_expired_session' })),
      { method: 'GET' }
    )
  ).toBe(true)

  // Gate refusal on a mutation: only when the caller vouches for the operation.
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'POST' })).toBe(false)
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'POST', replayOn401: true })).toBe(true)
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'DELETE', replayOn401: 'yes' })).toBe(false)

  // An application-level 401 (endpoint/plugin/backend; no gate shape) never replays.
  expect(shouldReplayAfterCookie401(httpStatusError(401, 'rejected'), { method: 'GET' })).toBe(false)
  expect(
    shouldReplayAfterCookie401(httpStatusError(401, JSON.stringify({ error: 'forbidden_tool' })), { method: 'GET' })
  ).toBe(false)
  expect(
    shouldReplayAfterCookie401(httpStatusError(401, JSON.stringify({ error: 'unauthenticated' })), { method: 'GET' })
  ).toBe(false)
  expect(shouldReplayAfterCookie401(GATE_401(), { method: 'POST' })).toBe(false)

  // Not a 401 at all.
  expect(
    shouldReplayAfterCookie401(httpStatusError(403, JSON.stringify({ error: 'unauthenticated', reason: 'x' })), {})
  ).toBe(false)
  expect(shouldReplayAfterCookie401(new Error('socket reset'), {})).toBe(false)
})

test('background roster auth leaves a gate 401 for the source row without clearing cookies or opening login', async () => {
  const refusal = GATE_401()
  const actions: string[] = []

  const recover = () =>
    retryCookie401WithLogin(
      refusal,
      { method: 'GET' },
      {
        clearCookies: () => {
          actions.push('clear')
        },
        login: async () => {
          actions.push('login')
        },
        retry: async () => {
          actions.push('retry')

          return 'ok'
        }
      }
    )

  await expect(
    withoutInteractiveOauthLogin(async () => {
      await Promise.resolve()

      return recover()
    })
  ).rejects.toBe(refusal)
  expect(actions).toEqual([])

  // A separate foreground request retains the existing one-login, one-retry path.
  expect(await recover()).toBe('ok')
  expect(actions).toEqual(['clear', 'login', 'retry'])
})

test('background auth intent stays isolated from a concurrent foreground request', async () => {
  const refusal = GATE_401()

  let releaseBackground: () => void = () => {}

  const backgroundReady = new Promise<void>(resolve => {
    releaseBackground = resolve
  })

  let foregroundLogins = 0

  const background = withoutInteractiveOauthLogin(async () => {
    await backgroundReady
    expect(canShowInteractiveOauthLogin()).toBe(false)

    return retryCookie401WithLogin(
      refusal,
      {},
      {
        clearCookies: () => {
          throw new Error('background cookies cleared')
        },
        login: async () => {
          throw new Error('background login opened')
        },
        retry: async () => 'unexpected'
      }
    )
  })

  const foreground = retryCookie401WithLogin(
    refusal,
    {},
    {
      clearCookies: () => {},
      login: async () => {
        foregroundLogins += 1
      },
      retry: async () => 'foreground recovered'
    }
  )

  releaseBackground()

  await expect(background).rejects.toBe(refusal)
  expect(canShowInteractiveOauthLogin()).toBe(true)
  expect(await foreground).toBe('foreground recovered')
  expect(foregroundLogins).toBe(1)
})

test('failed foreground re-login returns the original gate refusal without retrying', async () => {
  const refusal = GATE_401()
  let retries = 0

  await expect(
    retryCookie401WithLogin(
      refusal,
      {},
      {
        clearCookies: () => {},
        login: async () => {
          throw new Error('login window closed')
        },
        retry: async () => {
          retries += 1

          return 'unexpected'
        }
      }
    )
  ).rejects.toBe(refusal)
  expect(retries).toBe(0)
})

test('ws-ticket minting vouches for replay on a cookie 401', async () => {
  let seen: any

  await mintGatewayWsTicket('https://gw.test', {
    ensureNativeAccessToken: async () => null,
    fetchJson: async () => ({}),
    fetchJsonViaOauthSession: async (_url, options) => {
      seen = options

      return { ticket: 't' }
    }
  })

  expect(seen.method).toBe('POST')
  expect(seen.replayOn401).toBe(true)
})

test('native failures remain transport failures unless an independent cookie session succeeds', async () => {
  for (const nativeError of [new Error('timeout'), httpStatusError(503, 'down'), new Error('malformed response')]) {
    for (const cookieWorks of [true, false]) {
      const run = () =>
        requestWithOauthFallback('https://gw.test', {
          ensureNativeAccessToken: async () => {
            throw nativeError
          },
          requestWithBearer: async () => 'bearer',
          requestWithCookie: async () => {
            if (cookieWorks) {
              return 'cookie'
            }

            throw httpStatusError(401, 'no cookie')
          }
        })

      if (cookieWorks) {
        expect(await run()).toBe('cookie')
      } else {
        await expect(run()).rejects.toBe(nativeError)
        expect(isGatewayAuthRejection(nativeError)).toBe(false)
      }
    }
  }

  let cookieCalls = 0
  await expect(
    requestWithOauthFallback('https://gw.test', {
      ensureNativeAccessToken: async () => {
        throw new NativeAuthChangedError()
      },
      requestWithBearer: async () => 'bearer',
      requestWithCookie: async () => {
        cookieCalls++

        return 'cookie'
      }
    })
  ).rejects.toThrow('Authentication changed')
  expect(cookieCalls).toBe(0)
})

test('ticket mints force one rotation on bearer 401; ordinary requests never replay a mutation', async () => {
  for (const status of [401, 403, 503]) {
    for (const rotated of ['fresh', null, 'transient'] as const) {
      const calls: unknown[] = []
      let refreshes = 0

      const run = () =>
        mintGatewayWsTicket(
          'https://gw.test',
          {
            ensureNativeAccessToken: async (_host, options) => {
              if (!options?.forceRefresh) {
                return 'old'
              }

              expect(options.rejectedAccessToken).toBe('old')
              refreshes++

              if (rotated === 'transient') {
                throw new Error('refresh timeout')
              }

              return rotated
            },
            fetchJson: async (_url, _token, options) => {
              calls.push(options.bearer)
              expect(options.headers).toEqual({ 'x-proxy': 'test' })

              if (options.bearer === 'old') {
                throw httpStatusError(status, 'rejected')
              }

              return { ticket: 'ticket' }
            },
            fetchJsonViaOauthSession: async () => {
              throw httpStatusError(401, 'no cookie')
            }
          },
          { 'x-proxy': 'test' }
        )

      if (status === 401 && rotated === 'fresh') {
        expect(await run()).toBe('ticket')
      } else {
        await expect(run()).rejects.toThrow(status === 401 && rotated === 'transient' ? 'refresh timeout' : 'rejected')
      }

      expect(refreshes).toBe(status === 401 ? 1 : 0)
      expect(calls).toEqual(status === 401 && rotated === 'fresh' ? ['old', 'fresh'] : ['old'])
    }
  }

  for (const error of [httpStatusError(401, 'rejected'), new Error('socket reset after body sent')]) {
    let submissions = 0
    let cookies = 0
    await expect(
      requestWithOauthFallback('https://gw.test', {
        ensureNativeAccessToken: async () => 'live',
        requestWithBearer: async () => {
          submissions++
          throw error
        },
        requestWithCookie: async () => {
          cookies++

          return 'duplicate mutation'
        }
      })
    ).rejects.toBe(error)
    expect(submissions).toBe(1)
    expect(cookies).toBe(0)
  }
})
