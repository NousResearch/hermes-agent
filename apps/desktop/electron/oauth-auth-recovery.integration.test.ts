import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'vitest'

import { httpStatusError, readStatusCode } from './api-transport'
import { isReauthRequiredError, waitForHermesReady } from './backend-health'
import { isGatewayAuthRejection, normalizeRemoteBaseUrl, withTransientRetries } from './connection-config'
import { createMediaProtocolHandler } from './media-protocol'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import { nativeRefreshUrl, parseTokenResponse, tokenNeedsRefresh } from './native-oauth'
import { loadNativeTokenSet, type NativeTokenStoreIo, persistNativeTokenSet } from './native-token-store'
import { mintGatewayWsTicket, requestWithOauthFallback } from './oauth-rest-request'

// Real HTTP and encrypted temp-file persistence; no Electron, OS keychain,
// application state, external gateways or real credentials are touched.
async function fixture(beforeRefreshResponse?: () => Promise<void>) {
  const home = mkdtempSync(join(tmpdir(), 'hermes-auth-recovery-'))
  const storeFile = join(home, 'native-tokens.json')
  const key = randomBytes(32)

  const io: NativeTokenStoreIo = {
    encrypt: text => {
      const iv = randomBytes(12)
      const cipher = createCipheriv('aes-256-gcm', key, iv)
      const encrypted = Buffer.concat([cipher.update(text, 'utf8'), cipher.final()])

      return { value: Buffer.concat([iv, cipher.getAuthTag(), encrypted]).toString('base64') }
    },
    decrypt: secret => {
      const bytes = Buffer.from(secret.value, 'base64')
      const cipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12))
      cipher.setAuthTag(bytes.subarray(12, 28))

      return Buffer.concat([cipher.update(bytes.subarray(28)), cipher.final()]).toString('utf8')
    },
    readStoreText: () => readFileSync(storeFile, 'utf8'),
    writeStoreText: text => writeFileSync(storeFile, text, { mode: 0o600 })
  }

  const state = { refreshStatus: 200, cookie: false, refreshes: 0, mutations: 0, tickets: 0, malformed: false }

  const server = createServer(async (req, res) => {
    let body = ''

    for await (const chunk of req) {
      body += chunk
    }

    res.setHeader('content-type', 'application/json')

    if (req.url === '/auth/native/refresh') {
      state.refreshes++
      expect(JSON.parse(body)).toMatchObject({ refresh_token: 'old-rt', provider: 'nous' })
      await beforeRefreshResponse?.()
      res.statusCode = state.refreshStatus
      res.end(
        JSON.stringify(
          state.malformed
            ? {}
            : {
                access_token: 'fresh',
                refresh_token: 'fresh-rt',
                expires_at: 9_000,
                provider: 'nous',
                user_id: 'test'
              }
        )
      )

      return
    }

    if (req.headers.authorization !== 'Bearer fresh' && !(state.cookie && req.headers.cookie === 'test-cookie=live')) {
      res.statusCode = 401
      res.end(JSON.stringify({ error: 'no_cookie' }))

      return
    }

    if (req.url === '/api/auth/ws-ticket') {
      state.tickets++
      res.end(JSON.stringify({ ticket: `one-use-${state.tickets}` }))
    } else if (req.url === '/mutation') {
      state.mutations++
      req.socket.destroy() // server committed but the response was lost
    } else if (req.url?.startsWith('/api/files/stream')) {
      res.statusCode = 206
      res.end('media bytes')
    } else {
      res.end(JSON.stringify({ ready: true }))
    }
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address() as { port: number }
  const baseUrl = `http://127.0.0.1:${address.port}`

  const fetchJson = async (url: string, _token: string | null = null, options: any = {}) => {
    const response = await fetch(url, {
      method: options.method,
      headers: { ...options.headers, ...(options.bearer ? { authorization: `Bearer ${options.bearer}` } : {}) },
      body: options.body === undefined ? undefined : JSON.stringify(options.body)
    })

    const text = await response.text()

    if (!response.ok) {
      throw httpStatusError(response.status, text)
    }

    return JSON.parse(text)
  }

  const fetchCookie = (url: string, options: any = {}) =>
    fetchJson(url, null, {
      ...options,
      headers: { ...options.headers, cookie: 'test-cookie=live' }
    })

  const coordinator = createNativeAccessTokenCoordinator({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    nowSeconds: () => 1_000,
    tokenNeedsRefresh,
    loadTokens: host => loadNativeTokenSet(host, io),
    storeTokens: (host, tokens) => persistNativeTokenSet(host, tokens, io),
    clearTokens: host => persistNativeTokenSet(host, null, io),
    isRefreshAuthRejection: error => readStatusCode(error) === 401,
    refreshTokens: async (host, tokens) =>
      parseTokenResponse(
        await fetchJson(nativeRefreshUrl(host), null, {
          method: 'POST',
          body: { refresh_token: tokens.refreshToken, provider: tokens.provider }
        })
      )
  })

  const seed = (expiresAt = 1_000) =>
    coordinator.storeTokens(baseUrl, {
      accessToken: 'old',
      refreshToken: 'old-rt',
      expiresAt,
      provider: 'nous',
      userId: 'test'
    })

  const mint = () =>
    mintGatewayWsTicket(baseUrl, {
      ensureNativeAccessToken: coordinator.ensure,
      fetchJson,
      fetchJsonViaOauthSession: fetchCookie
    })

  const request = (path: string, method = 'GET') =>
    requestWithOauthFallback(baseUrl, {
      ensureNativeAccessToken: coordinator.ensure,
      requestWithBearer: bearer => fetchJson(`${baseUrl}${path}`, null, { bearer, method }),
      requestWithCookie: () => fetchCookie(`${baseUrl}${path}`, { method })
    })

  return {
    baseUrl,
    coordinator,
    io,
    state,
    seed,
    mint,
    request,
    async close() {
      server.closeAllConnections()
      await new Promise<void>((resolve, reject) => server.close(error => (error ? reject(error) : resolve())))
      rmSync(home, { recursive: true, force: true })
    }
  }
}

test('HTTP rotation survives encrypted restart and concurrent ticket dials without replaying committed mutations', async () => {
  let markRefreshStarted!: () => void
  let releaseRefreshResponse!: () => void

  const refreshStarted = new Promise<void>(resolve => {
    markRefreshStarted = resolve
  })

  const refreshResponse = new Promise<void>(resolve => {
    releaseRefreshResponse = resolve
  })

  const f = await fixture(async () => {
    markRefreshStarted()
    await refreshResponse
  })

  try {
    f.seed(2_000) // locally live, server rejects: the landed forced-refresh path
    const dials = Promise.all(Array.from({ length: 8 }, () => f.mint()))
    await refreshStarted
    const loginIsCurrent = f.coordinator.beginLogin(f.baseUrl)
    const parallel = f.coordinator.ensure(f.baseUrl)
    releaseRefreshResponse()
    const tickets = await dials
    expect(await parallel).toBe('fresh')
    expect(loginIsCurrent()).toBe(true) // abandoned login never stores a replacement
    expect(new Set(tickets).size).toBe(tickets.length)
    expect(f.state.refreshes).toBe(1)
    expect(loadNativeTokenSet(f.baseUrl, f.io)?.refreshToken).toBe('fresh-rt')
    expect(await f.coordinator.ensure(f.baseUrl)).toBe('fresh')
    await expect(f.request('/mutation', 'POST')).rejects.toThrow()
    expect(f.state.mutations).toBe(1)
  } finally {
    await f.close()
  }
})

test('HTTP outage, dead-refresh and cookie coexistence reach the correct ticket/readiness/media verdicts', async () => {
  const f = await fixture()

  try {
    f.seed()
    f.state.refreshStatus = 503
    await expect(withTransientRetries(f.mint, { sleep: async () => {} })).rejects.toMatchObject({ statusCode: 503 })
    expect(f.state.refreshes).toBe(3)
    expect(loadNativeTokenSet(f.baseUrl, f.io)?.refreshToken).toBe('old-rt')
    let clock = 0

    try {
      await waitForHermesReady(f.baseUrl, {
        fetchPublicJson: async () => {
          throw new Error('must not anonymously downgrade')
        },
        fetchJson: () => f.request('/api/status'),
        probeHealth: () => f.request('/api/health'),
        probeIsCredentialed: true,
        now: () => clock,
        sleep: async () => {
          clock++
        },
        timeoutMs: 2
      })
      throw new Error('unexpected ready')
    } catch (error) {
      expect(isReauthRequiredError(error)).toBe(false)
      expect(String(error)).toContain('503:')
    }

    const media = createMediaProtocolHandler({
      ensureRemoteBearer: f.coordinator.ensure,
      resolveRemoteConnection: async () => ({ baseUrl: f.baseUrl, mode: 'remote', authMode: 'oauth' }),
      resolveLocalFile: async path => path,
      fetchLocal: async () => {
        throw new Error('not local')
      },
      fetchRemote: (url, headers, method) => fetch(url, { headers, method }),
      fetchRemoteWithCookies: (url, headers, method) =>
        fetch(url, {
          headers: { ...Object.fromEntries(headers), cookie: 'test-cookie=live' },
          method
        })
    })

    const mediaRequest = { url: 'hermes-media://remote/%2Ftmp%2Fclip.mp4', headers: new Headers(), method: 'GET' }
    expect((await media(mediaRequest)).status).toBe(502)
    f.state.cookie = true
    expect(await f.mint()).toMatch(/^one-use-/)
    expect((await media(mediaRequest)).status).toBe(206)
    f.state.cookie = false
    f.state.refreshStatus = 401
    await expect(f.mint()).rejects.toMatchObject({ statusCode: 401 })
    expect(loadNativeTokenSet(f.baseUrl, f.io)).toBeNull()
    f.seed()
    f.state.refreshStatus = 200
    f.state.malformed = true
    await expect(f.mint()).rejects.toThrow('missing access_token')
    expect(loadNativeTokenSet(f.baseUrl, f.io)).not.toBeNull()
    expect(isGatewayAuthRejection(new Error('401: misleading message'))).toBe(false)
  } finally {
    await f.close()
  }
})
