import { expect, test } from 'vitest'

import { normalizeRemoteBaseUrl } from './connection-config'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import { type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'

const tokenSet = (name: string, expiresAt = 2_000): NativeTokenSet => ({
  accessToken: name,
  refreshToken: `${name}-rt`,
  expiresAt,
  provider: 'nous',
  userId: 'user'
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

// Covers forced rotation too: a locally live bearer can be rejected server-side.
test('normalised hosts share a rotation, and late bearer rejections reuse its winner', async () => {
  const store = new Map([['https://gw.test', tokenSet('old')]])
  const pending = deferred<NativeTokenSet>()
  let rotations = 0

  const coordinator = createNativeAccessTokenCoordinator({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    nowSeconds: () => 1_000,
    tokenNeedsRefresh,
    loadTokens: host => store.get(host) ?? null,
    storeTokens: (host, tokens) => {
      store.set(host, tokens)
    },
    clearTokens: host => {
      store.delete(host)
    },
    isRefreshAuthRejection: error => (error as { statusCode?: number })?.statusCode === 401,
    refreshTokens: async () => {
      rotations++

      return pending.promise
    }
  })

  const first = coordinator.ensure('https://GW.test/', { forceRefresh: true, rejectedAccessToken: 'old' })
  const second = coordinator.ensure('https://gw.test', { forceRefresh: true, rejectedAccessToken: 'old' })
  const ordinary = coordinator.ensure('https://gw.test')
  expect(rotations).toBe(1)
  pending.resolve(tokenSet('winner'))
  expect(await Promise.all([first, second, ordinary])).toEqual(['winner', 'winner', 'winner'])
  expect(await coordinator.ensure('https://gw.test', { forceRefresh: true, rejectedAccessToken: 'old' })).toBe('winner')
  expect(rotations).toBe(1)
})

test('a pending or abandoned login preserves the existing refresh flight and its rotation', async () => {
  const host = 'https://gw.test'
  const store = new Map([[host, tokenSet('old', 1_000)]])
  const pending = deferred<NativeTokenSet>()
  const refreshTokens: string[] = []

  const coordinator = createNativeAccessTokenCoordinator({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    nowSeconds: () => 1_000,
    tokenNeedsRefresh,
    loadTokens: key => store.get(key) ?? null,
    storeTokens: (key, tokens) => {
      store.set(key, tokens)
    },
    clearTokens: key => {
      store.delete(key)
    },
    isRefreshAuthRejection: () => false,
    refreshTokens: async (_key, tokens) => {
      refreshTokens.push(tokens.refreshToken!)

      return pending.promise
    }
  })

  const first = coordinator.ensure(host)
  const loginIsCurrent = coordinator.beginLogin('https://GW.test/')
  const parallel = coordinator.ensure(host, { forceRefresh: true, rejectedAccessToken: 'old' })
  const results = Promise.allSettled([first, parallel])
  pending.resolve(tokenSet('rotated'))
  expect(await results).toEqual([
    { status: 'fulfilled', value: 'rotated' },
    { status: 'fulfilled', value: 'rotated' }
  ])
  expect(loginIsCurrent()).toBe(true)
  // No login completion/store: closing the browser must leave the rotation intact.
  expect(store.get(host)?.refreshToken).toBe('rotated-rt')
  expect(await coordinator.ensure(host, { forceRefresh: true, rejectedAccessToken: 'old' })).toBe('rotated')
  expect(refreshTokens).toEqual(['old-rt'])
})

test('explicit token mutations fence stale success and rejection without affecting other hosts', async () => {
  for (const rejection of [false, true]) {
    const host = 'https://gw.test'
    const sibling = 'https://other.test'

    const store = new Map([
      [host, tokenSet('old', 1_000)],
      [sibling, tokenSet('other', 1_000)]
    ])

    const pending = deferred<NativeTokenSet>()
    const other = deferred<NativeTokenSet>()

    const coordinator = createNativeAccessTokenCoordinator({
      normalizeBaseUrl: normalizeRemoteBaseUrl,
      nowSeconds: () => 1_000,
      tokenNeedsRefresh,
      loadTokens: key => store.get(key) ?? null,
      storeTokens: (key, value) => {
        store.set(key, value)
      },
      clearTokens: key => {
        store.delete(key)
      },
      isRefreshAuthRejection: error => (error as { statusCode?: number })?.statusCode === 401,
      refreshTokens: key => (key === host ? pending.promise : other.promise)
    })

    const old = coordinator.ensure(host)
    const stale = expect(old).rejects.toThrow('Authentication changed')
    const otherFlight = coordinator.ensure(sibling)
    const login = coordinator.beginLogin('https://GW.test/')
    coordinator.clearTokens(host) // logout supersedes the pending login before cookie I/O
    expect(login()).toBe(false)
    expect(await coordinator.ensure(host)).toBeNull()

    if (rejection) {
      pending.reject({ statusCode: 401 })
    } else {
      pending.resolve(tokenSet('stale'))
    }

    other.resolve(tokenSet('other-winner'))
    await stale
    expect(await otherFlight).toBe('other-winner')
    expect(store.has(host)).toBe(false)
    expect(await coordinator.ensure(host)).toBeNull()
  }
})

test('newer login intent wins out-of-order completions and fences an older refresh only on store', async () => {
  for (const rejection of [false, true]) {
    const host = 'https://gw.test'
    const store = new Map([[host, tokenSet('old', 1_000)]])
    const pending = deferred<NativeTokenSet>()

    const coordinator = createNativeAccessTokenCoordinator({
      normalizeBaseUrl: normalizeRemoteBaseUrl,
      nowSeconds: () => 1_000,
      tokenNeedsRefresh,
      loadTokens: key => store.get(key) ?? null,
      storeTokens: (key, tokens) => {
        store.set(key, tokens)
      },
      clearTokens: key => {
        store.delete(key)
      },
      isRefreshAuthRejection: () => true,
      refreshTokens: () => pending.promise
    })

    const completeLogin = async (rawHost: string, completion: Promise<NativeTokenSet>) => {
      const isCurrent = coordinator.beginLogin(rawHost)
      const tokens = await completion

      if (!isCurrent()) {
        return false
      }

      coordinator.storeTokens(rawHost, tokens)

      return true
    }

    const refresh = coordinator.ensure(host)
    const stale = expect(refresh).rejects.toThrow('Authentication changed')
    const oldCompletion = deferred<NativeTokenSet>()
    const newerCompletion = deferred<NativeTokenSet>()
    const oldLogin = completeLogin('https://GW.test/', oldCompletion.promise)
    const newerLogin = completeLogin(host, newerCompletion.promise)

    if (!rejection) {
      oldCompletion.resolve(tokenSet('stale-login'))
      expect(await oldLogin).toBe(false) // newer intent wins even before it completes
    }

    newerCompletion.resolve(tokenSet('new-login'))
    expect(await newerLogin).toBe(true)
    expect(await coordinator.ensure(host)).toBe('new-login')

    if (rejection) {
      oldCompletion.resolve(tokenSet('stale-login'))
      expect(await oldLogin).toBe(false)
    }

    if (rejection) {
      pending.reject({ statusCode: 401 })
    } else {
      pending.resolve(tokenSet('stale-refresh'))
    }

    await stale
    expect(store.get(host)?.accessToken).toBe('new-login')

    const logoutCompletion = deferred<NativeTokenSet>()
    const loggedOutLogin = completeLogin(host, logoutCompletion.promise)
    coordinator.clearTokens('https://GW.test/')
    logoutCompletion.resolve(tokenSet('after-logout'))
    expect(await loggedOutLogin).toBe(false)
    expect(await coordinator.ensure(host)).toBeNull()
  }
})
