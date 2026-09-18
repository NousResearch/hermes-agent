import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { isGatewayAuthRejection } from './connection-config'
import { NativeAuthChangedError } from './native-access-token'
import { mintGatewayWsTicket, requestWithOauthFallback } from './oauth-rest-request'

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
