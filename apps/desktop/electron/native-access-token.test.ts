import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import {
  isNativeRefreshAuthRejection,
  normalizeRemoteBaseUrl
} from './connection-config'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import { type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'

function tokenSet(
  accessToken: string,
  refreshToken: string,
  expiresAt: number
): NativeTokenSet {
  return {
    accessToken,
    expiresAt,
    provider: 'nous',
    refreshToken,
    userId: 'user-1'
  }
}

function deferred<T>() {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function coordinatorFixture(initialTokens: NativeTokenSet) {
  let storedTokens: NativeTokenSet | null = initialTokens
  const refresh = deferred<NativeTokenSet>()
  const refreshTokens = vi.fn(async () => refresh.promise)

  const storeTokens = vi.fn((_baseUrl: string, tokens: NativeTokenSet) => {
    storedTokens = tokens
  })

  const clearTokens = vi.fn(() => {
    storedTokens = null
  })

  const coordinator = createNativeAccessTokenCoordinator({
    clearTokens,
    isRefreshAuthRejection: isNativeRefreshAuthRejection,
    loadTokens: () => storedTokens,
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    nowSeconds: () => 1_000,
    refreshTokens,
    storeTokens,
    tokenNeedsRefresh
  })

  return {
    clearTokens,
    coordinator,
    refresh,
    refreshTokens,
    setStoredTokens: (tokens: NativeTokenSet | null) => {
      storedTokens = tokens
    },
    storeTokens,
    storedTokens: () => storedTokens
  }
}

test('coalesces concurrent refreshes by normalized base URL and stores one rotation', async () => {
  const oldTokens = tokenSet('expiring-at', 'rotating-rt', 1_000)
  const rotated = tokenSet('rotated-at', 'rotated-rt', 2_000)
  const fixture = coordinatorFixture(oldTokens)

  const first = fixture.coordinator.ensure('https://GW.example.com/')
  const second = fixture.coordinator.ensure('https://gw.example.com')

  assert.equal(fixture.refreshTokens.mock.calls.length, 1)

  fixture.refresh.resolve(rotated)

  assert.deepEqual(await Promise.all([first, second]), ['rotated-at', 'rotated-at'])
  assert.equal(fixture.refreshTokens.mock.calls.length, 1)
  assert.equal(fixture.storeTokens.mock.calls.length, 1)
  assert.equal(fixture.storedTokens(), rotated)
})

test('a stale refresh rejection does not clear an already-rotated refresh token', async () => {
  const oldTokens = tokenSet('expiring-at', 'old-rt', 1_000)
  const rotated = tokenSet('winner-at', 'winner-rt', 2_000)
  const fixture = coordinatorFixture(oldTokens)
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  fixture.setStoredTokens(rotated)
  fixture.refresh.reject(Object.assign(new Error('401: {"error":"session_expired"}'), { statusCode: 401 }))

  assert.equal(await pending, null)
  assert.equal(fixture.clearTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), rotated)
})

test('logout invalidates an in-flight refresh so its result cannot restore tokens', async () => {
  const oldTokens = tokenSet('expiring-at', 'old-rt', 1_000)
  const rotated = tokenSet('stale-at', 'stale-rt', 2_000)
  const fixture = coordinatorFixture(oldTokens)
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  fixture.coordinator.invalidateExplicitAuthChange()
  fixture.setStoredTokens(null)
  fixture.refresh.resolve(rotated)

  assert.equal(await pending, null)
  assert.equal(fixture.storeTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), null)
})

test('a completed explicit login wins over an older in-flight refresh', async () => {
  const oldTokens = tokenSet('expiring-at', 'old-rt', 1_000)
  const staleRotation = tokenSet('stale-at', 'stale-rt', 2_000)
  const loginTokens = tokenSet('login-at', 'login-rt', 3_000)
  const fixture = coordinatorFixture(oldTokens)
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  fixture.coordinator.invalidateExplicitAuthChange()
  fixture.setStoredTokens(loginTokens)
  fixture.refresh.resolve(staleRotation)

  assert.equal(await pending, null)
  assert.equal(fixture.storeTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), loginTokens)
})
