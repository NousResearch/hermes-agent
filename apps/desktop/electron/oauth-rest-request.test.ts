import assert from 'node:assert/strict'

import { expect, test, vi } from 'vitest'

import { requestWithOauthFallback } from './oauth-rest-request'

test('refresh timeout plus an empty cookie jar rejects with the original transport error', async () => {
  const nativeError = Object.assign(new Error('refresh timed out'), { code: 'ETIMEDOUT' })
  const cookieAuthError = Object.assign(new Error('401: no session cookie'), { statusCode: 401 })
  const requestWithBearer = vi.fn()

  const requestWithCookie = vi.fn(async () => {
    throw cookieAuthError
  })

  await expect(
    requestWithOauthFallback('https://gw.example.com', {
      ensureNativeAccessToken: async () => {
        throw nativeError
      },
      requestWithBearer,
      requestWithCookie
    })
  ).rejects.toBe(nativeError)

  assert.equal(requestWithBearer.mock.calls.length, 0)
  assert.equal(requestWithCookie.mock.calls.length, 1)
})

test('refresh timeout plus a valid cookie session serves the REST request', async () => {
  const nativeError = Object.assign(new Error('refresh timed out'), { code: 'ETIMEDOUT' })
  const cookieResponse = { sessions: [{ id: 'cookie-session' }] }
  const requestWithCookie = vi.fn(async () => cookieResponse)

  const response = await requestWithOauthFallback('https://gw.example.com', {
    ensureNativeAccessToken: async () => {
      throw nativeError
    },
    requestWithBearer: vi.fn(),
    requestWithCookie
  })

  assert.equal(response, cookieResponse)
  assert.equal(requestWithCookie.mock.calls.length, 1)
})

test('a non-auth cookie failure remains the more immediate REST failure', async () => {
  const nativeError = Object.assign(new Error('refresh timed out'), { code: 'ETIMEDOUT' })
  const cookieServerError = Object.assign(new Error('503: gateway unavailable'), { statusCode: 503 })

  await expect(
    requestWithOauthFallback('https://gw.example.com', {
      ensureNativeAccessToken: async () => {
        throw nativeError
      },
      requestWithBearer: vi.fn(),
      requestWithCookie: async () => {
        throw cookieServerError
      }
    })
  ).rejects.toBe(cookieServerError)
})
