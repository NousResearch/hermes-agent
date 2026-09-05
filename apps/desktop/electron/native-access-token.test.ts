import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'

import { ScriptTarget, transpileModule } from 'typescript'
import { test, vi } from 'vitest'

import {
  authModeFromStatus,
  isNativeRefreshAuthRejection,
  normalizeRemoteBaseUrl
} from './connection-config'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import {
  type NativeTokenSet,
  resolveLoginStrategy,
  tokenNeedsRefresh
} from './native-oauth'

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

function loadOauthLogoutHandler({
  clearOauthSession,
  clearNativeTokens,
  coordinator
}: {
  clearOauthSession: (baseUrl?: string) => Promise<void>
  clearNativeTokens: (baseUrl: string) => void
  coordinator: ReturnType<typeof createNativeAccessTokenCoordinator>
}) {
  // main.ts cannot be imported under the Electron unit runner without booting
  // the app lifecycle. Execute the registered handler body itself so these tests
  // cannot drift into a hand-written approximation of production ordering.
  const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
  const match = mainSource.match(
    /ipcMain\.handle\('hermes:connection-config:oauth-logout', (async \(_event, rawUrl\) => \{[\s\S]*?\r?\n\})\)\r?\n/
  )

  assert.ok(match, 'oauth-logout IPC handler was not found in main.ts')

  return runInNewContext(`(${match[1]})`, {
    _clearNativeTokens: clearNativeTokens,
    clearOauthSession,
    hasLiveOauthSession: async () => false,
    hasNativeSession: () => false,
    nativeAccessTokenCoordinator: coordinator,
    normalizeRemoteBaseUrl
  }) as (_event: unknown, rawUrl?: string) => Promise<{ connected: boolean; ok: boolean }>
}

function loadOauthLoginHandler({
  coordinator,
  runNativeLogin,
  storeNativeTokens
}: {
  coordinator: ReturnType<typeof createNativeAccessTokenCoordinator>
  runNativeLogin: (baseUrl: string, deps: unknown) => Promise<NativeTokenSet>
  storeNativeTokens: (baseUrl: string, tokens: NativeTokenSet) => void
}) {
  // Execute the registered production handler body, as the logout fixture does.
  // The extracted login handler contains a TypeScript local annotation, so
  // transpile it before passing the otherwise unchanged body to node:vm.
  const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
  const match = mainSource.match(
    /ipcMain\.handle\('hermes:connection-config:oauth-login', (async \(_event, rawUrl\) => \{[\s\S]*?\r?\n\})\)\r?\n/
  )

  assert.ok(match, 'oauth-login IPC handler was not found in main.ts')

  const handlerSource = transpileModule(`const handler = ${match[1]}`, {
    compilerOptions: { target: ScriptTarget.ES2022 }
  }).outputText

  return runInNewContext(`${handlerSource}\nhandler`, {
    _storeNativeTokens: storeNativeTokens,
    // authModeFromStatus is the real connection-config.ts export (imported
    // above), not a stub — upstream 21f34794be added the auth_flows/status
    // gating this handler now does before choosing native vs. embedded login.
    authModeFromStatus,
    fetchPublicJson: async () => ({ auth_flows: ['native_pkce'] }),
    hasOauthSessionCookie: async () => false,
    nativeAccessTokenCoordinator: coordinator,
    normalizeRemoteBaseUrl,
    openOauthLoginWindow: async () => {},
    postJsonNoAuth: async () => ({}),
    rememberLog: () => {},
    remoteReauthFailure: null,
    resolveLoginStrategy,
    runNativeLogin,
    shell: {
      openExternal: async () => {}
    }
  }) as (
    _event: unknown,
    rawUrl: string
  ) => Promise<{ baseUrl: string; connected: boolean; ok: boolean }>
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

test('an auth change for another gateway preserves a successful rotation', async () => {
  const rotated = tokenSet('gw-b-rotated-at', 'gw-b-rotated-rt', 2_000)
  const fixture = coordinatorFixture(tokenSet('gw-b-expiring-at', 'gw-b-old-rt', 1_000))
  const pending = fixture.coordinator.ensure('https://gw-b.example.com')

  fixture.coordinator.invalidateExplicitAuthChange('https://gw-a.example.com')
  fixture.refresh.resolve(rotated)

  assert.equal(await pending, 'gw-b-rotated-at')
  assert.deepEqual(fixture.storeTokens.mock.calls, [['https://gw-b.example.com', rotated]])
})

test('an auth change for another gateway still processes a refresh rejection', async () => {
  const fixture = coordinatorFixture(tokenSet('gw-b-expiring-at', 'gw-b-old-rt', 1_000))
  const pending = fixture.coordinator.ensure('https://gw-b.example.com')

  fixture.coordinator.invalidateExplicitAuthChange('https://gw-a.example.com')
  fixture.refresh.reject(Object.assign(new Error('401: {"error":"session_expired"}'), { statusCode: 401 }))

  assert.equal(await pending, null)
  assert.deepEqual(fixture.clearTokens.mock.calls, [['https://gw-b.example.com']])
})

test('oauth logout without a URL preserves a successful refresh for another gateway', async () => {
  const rotated = tokenSet('gw-b-rotated-at', 'gw-b-rotated-rt', 2_000)
  const fixture = coordinatorFixture(tokenSet('gw-b-expiring-at', 'gw-b-old-rt', 1_000))
  const clearNativeTokens = vi.fn((_baseUrl: string) => {})
  const clearOauthSession = vi.fn(async (_baseUrl?: string) => {})
  const handler = loadOauthLogoutHandler({
    clearNativeTokens,
    clearOauthSession,
    coordinator: fixture.coordinator
  })
  const pending = fixture.coordinator.ensure('https://gw-b.example.com')

  // Upstream 21f34794be dropped the URL-less logout path — the handler now
  // calls normalizeRemoteBaseUrl(rawUrl) unconditionally, which throws
  // before touching any gateway's token state.
  await assert.rejects(handler(undefined, undefined), { message: 'Remote gateway URL is required.' })
  fixture.refresh.resolve(rotated)

  assert.equal(clearOauthSession.mock.calls.length, 0)
  assert.equal(clearNativeTokens.mock.calls.length, 0)
  assert.equal(await pending, 'gw-b-rotated-at')
  assert.deepEqual(fixture.storeTokens.mock.calls, [['https://gw-b.example.com', rotated]])
})

test('oauth logout without a URL still processes a refresh rejection for another gateway', async () => {
  const fixture = coordinatorFixture(tokenSet('gw-b-expiring-at', 'gw-b-old-rt', 1_000))
  const clearNativeTokens = vi.fn((_baseUrl: string) => {})
  const clearOauthSession = vi.fn(async (_baseUrl?: string) => {})
  const handler = loadOauthLogoutHandler({
    clearNativeTokens,
    clearOauthSession,
    coordinator: fixture.coordinator
  })
  const pending = fixture.coordinator.ensure('https://gw-b.example.com')

  // Upstream 21f34794be dropped the URL-less logout path — the handler now
  // calls normalizeRemoteBaseUrl(rawUrl) unconditionally, which throws
  // before touching any gateway's token state.
  await assert.rejects(handler(undefined, undefined), { message: 'Remote gateway URL is required.' })
  fixture.refresh.reject(Object.assign(new Error('401: {"error":"session_expired"}'), { statusCode: 401 }))

  assert.equal(clearOauthSession.mock.calls.length, 0)
  assert.equal(clearNativeTokens.mock.calls.length, 0)
  assert.equal(await pending, null)
  assert.deepEqual(fixture.clearTokens.mock.calls, [['https://gw-b.example.com']])
})

test('oauth logout with a URL invalidates an in-flight refresh for the same gateway', async () => {
  const rotated = tokenSet('stale-at', 'stale-rt', 2_000)
  const fixture = coordinatorFixture(tokenSet('expiring-at', 'old-rt', 1_000))
  const clearNativeTokens = vi.fn((_baseUrl: string) => {
    fixture.setStoredTokens(null)
  })
  const clearOauthSession = vi.fn(async (_baseUrl?: string) => {})
  const handler = loadOauthLogoutHandler({
    clearNativeTokens,
    clearOauthSession,
    coordinator: fixture.coordinator
  })
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  const result = await handler(undefined, 'https://GW.example.com/')
  fixture.refresh.resolve(rotated)

  assert.equal(result.ok, true)
  assert.equal(result.connected, false)
  assert.deepEqual(clearNativeTokens.mock.calls, [['https://gw.example.com']])
  assert.deepEqual(clearOauthSession.mock.calls, [['https://gw.example.com']])
  assert.equal(await pending, null)
  assert.equal(fixture.storeTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), null)
})

test('oauth logout with a URL preserves an in-flight refresh for another gateway', async () => {
  const rotated = tokenSet('gw-b-rotated-at', 'gw-b-rotated-rt', 2_000)
  const fixture = coordinatorFixture(tokenSet('gw-b-expiring-at', 'gw-b-old-rt', 1_000))
  const invalidateExplicitAuthChange = vi.spyOn(
    fixture.coordinator,
    'invalidateExplicitAuthChange'
  )
  const clearNativeTokens = vi.fn((_baseUrl: string) => {})
  const clearOauthSession = vi.fn(async (_baseUrl?: string) => {})
  const handler = loadOauthLogoutHandler({
    clearNativeTokens,
    clearOauthSession,
    coordinator: fixture.coordinator
  })
  const pending = fixture.coordinator.ensure('https://gw-b.example.com')

  const result = await handler(undefined, 'https://GW-A.example.com/')
  fixture.refresh.resolve(rotated)

  assert.equal(result.ok, true)
  assert.equal(result.connected, false)
  assert.deepEqual(invalidateExplicitAuthChange.mock.calls, [['https://gw-a.example.com']])
  assert.deepEqual(clearNativeTokens.mock.calls, [['https://gw-a.example.com']])
  assert.deepEqual(clearOauthSession.mock.calls, [['https://gw-a.example.com']])
  assert.equal(await pending, 'gw-b-rotated-at')
  assert.deepEqual(fixture.storeTokens.mock.calls, [['https://gw-b.example.com', rotated]])
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

test('logout invalidates an in-flight refresh for the same gateway so its result cannot restore tokens', async () => {
  const oldTokens = tokenSet('expiring-at', 'old-rt', 1_000)
  const rotated = tokenSet('stale-at', 'stale-rt', 2_000)
  const fixture = coordinatorFixture(oldTokens)
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  fixture.coordinator.invalidateExplicitAuthChange('https://gw.example.com')
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

  fixture.coordinator.invalidateExplicitAuthChange('https://gw.example.com')
  fixture.setStoredTokens(loginTokens)
  fixture.refresh.resolve(staleRotation)

  assert.equal(await pending, null)
  assert.equal(fixture.storeTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), loginTokens)
})

test('native login invalidates an older in-flight refresh for the same gateway', async () => {
  const staleRotation = tokenSet('stale-at', 'stale-rt', 2_000)
  const loginTokens = tokenSet('login-at', 'login-rt', 3_000)
  const fixture = coordinatorFixture(tokenSet('expiring-at', 'old-rt', 1_000))
  const runNativeLogin = vi.fn(
    async (_baseUrl: string, _deps: unknown) => loginTokens
  )
  const storeNativeTokens = vi.fn(
    (_baseUrl: string, tokens: NativeTokenSet) => {
      fixture.setStoredTokens(tokens)
    }
  )
  const handler = loadOauthLoginHandler({
    coordinator: fixture.coordinator,
    runNativeLogin,
    storeNativeTokens
  })
  const pending = fixture.coordinator.ensure('https://gw.example.com')

  const result = await handler(undefined, 'https://GW.example.com/')
  fixture.refresh.resolve(staleRotation)

  assert.equal(result.ok, true)
  assert.equal(result.connected, true)
  assert.equal(result.baseUrl, 'https://gw.example.com')
  assert.equal(runNativeLogin.mock.calls.length, 1)
  assert.equal(runNativeLogin.mock.calls[0]?.[0], 'https://gw.example.com')
  assert.deepEqual(storeNativeTokens.mock.calls, [['https://gw.example.com', loginTokens]])
  assert.equal(await pending, null)
  assert.equal(fixture.storeTokens.mock.calls.length, 0)
  assert.equal(fixture.storedTokens(), loginTokens)
})
