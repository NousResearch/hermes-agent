/**
 * E2E regression for password-backed remote gateways using native PKCE.
 *
 * The old desktop flow opened password providers in an embedded cookie jar.
 * A renderer reload then exposed the real failure: authenticated REST/WS
 * traffic had no reusable credential and failed with `401 no_cookie`.
 *
 * This test drives the public preload API through a real Electron process.
 * Only the system-browser boundary is replaced: the fake browser follows the
 * gateway redirect into the real loopback listener. The gateway then accepts
 * only the native bearer for its protected WS-ticket route, so the old
 * embedded-cookie implementation fails with the reported error while native
 * PKCE survives both a renderer reload and a full app relaunch.
 */

import { createHash } from 'node:crypto'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import type { Socket } from 'node:net'

import { buildAppEnv, createSandbox, launchDesktop, type Sandbox } from './fixtures'
import { allowErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

const ACCESS_TOKEN = 'e2e-native-password-access-token'
const REFRESH_TOKEN = 'e2e-native-password-refresh-token'
const APP_NAME = 'HermesE2ENativePasswordAuth'

interface FakePasswordGateway {
  url: string
  bearerTicketMints: number
  embeddedLogins: number
  nativeProviders: string[]
  noCookieRejects: number
  close: () => Promise<void>
}

function sendJson(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}

async function readJson(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []

  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk))
  }

  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

async function startFakePasswordGateway(): Promise<FakePasswordGateway> {
  const sockets = new Set<Socket>()
  const state = {
    bearerTicketMints: 0,
    embeddedLogins: 0,
    nativeProviders: [] as string[],
    noCookieRejects: 0
  }

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1')

      if (url.pathname === '/api/status') {
        sendJson(res, 200, {
          auth_flows: ['cookie', 'native_pkce'],
          auth_required: true,
          version: '0.0.0-e2e-password-gateway'
        })

        return
      }

      if (url.pathname === '/api/auth/providers') {
        sendJson(res, 200, {
          providers: [
            {
              display_name: 'Username & Password',
              name: 'basic',
              supports_password: true
            }
          ]
        })

        return
      }

      if (url.pathname === '/login') {
        state.embeddedLogins += 1
        // This makes the legacy embedded window look locally signed in. The
        // protected gateway route below still rejects cookie-only traffic,
        // reproducing the reported post-refresh `no_cookie` failure.
        res.writeHead(200, {
          'content-type': 'text/html; charset=utf-8',
          'set-cookie': 'hermes_session_at=embedded-only; Path=/; HttpOnly; SameSite=Lax'
        })
        res.end('<!doctype html><title>Signed in</title>Signed in')

        return
      }

      if (url.pathname === '/auth/native/authorize') {
        const provider = url.searchParams.get('provider') ?? ''
        const redirectUri = url.searchParams.get('redirect_uri')
        const oauthState = url.searchParams.get('state')

        state.nativeProviders.push(provider)

        if (provider !== 'basic' || !redirectUri || !oauthState) {
          sendJson(res, 400, { detail: 'invalid native authorization request' })

          return
        }

        const callback = new URL(redirectUri)
        callback.searchParams.set('code', 'e2e-native-code')
        callback.searchParams.set('state', oauthState)
        res.writeHead(302, { location: callback.toString() })
        res.end()

        return
      }

      if (url.pathname === '/auth/native/token' && req.method === 'POST') {
        const body = await readJson(req)

        if (body.code !== 'e2e-native-code' || typeof body.code_verifier !== 'string') {
          sendJson(res, 400, { detail: 'invalid native token exchange' })

          return
        }

        sendJson(res, 200, {
          access_token: ACCESS_TOKEN,
          expires_at: Math.floor(Date.now() / 1000) + 3_600,
          provider: 'basic',
          refresh_token: REFRESH_TOKEN,
          token_type: 'Bearer',
          user_id: 'e2e-user'
        })

        return
      }

      if (url.pathname === '/api/auth/ws-ticket' && req.method === 'POST') {
        if (req.headers.authorization !== `Bearer ${ACCESS_TOKEN}`) {
          state.noCookieRejects += 1
          sendJson(res, 401, {
            detail: 'Unauthorized',
            error: 'unauthenticated',
            login_url: '/login',
            reason: 'no_cookie'
          })

          return
        }

        state.bearerTicketMints += 1
        sendJson(res, 200, { ticket: 'e2e-ws-ticket' })

        return
      }

      sendJson(res, 404, { detail: 'not found' })
    } catch (error) {
      sendJson(res, 500, { detail: error instanceof Error ? error.message : String(error) })
    }
  })

  server.on('connection', socket => {
    sockets.add(socket)
    socket.on('close', () => sockets.delete(socket))
  })

  server.on('upgrade', (req, socket) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1')
    const key = req.headers['sec-websocket-key']

    if (url.pathname !== '/api/ws' || url.searchParams.get('ticket') !== 'e2e-ws-ticket' || !key) {
      socket.destroy()

      return
    }

    const accept = createHash('sha1').update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest('base64')

    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
        'Upgrade: websocket\r\n' +
        'Connection: Upgrade\r\n' +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`
    )
    // One unmasked server text frame is enough for the real WS probe to prove
    // the authenticated transport leg is usable.
    socket.write(Buffer.from([0x81, 0x02, 0x6f, 0x6b]))
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  const { port } = server.address() as AddressInfo

  return {
    close: () =>
      new Promise<void>(resolve => {
        for (const socket of sockets) {
          socket.destroy()
        }

        server.closeAllConnections?.()
        server.close(() => resolve())
      }),
    get bearerTicketMints() {
      return state.bearerTicketMints
    },
    get embeddedLogins() {
      return state.embeddedLogins
    },
    get nativeProviders() {
      return state.nativeProviders
    },
    get noCookieRejects() {
      return state.noCookieRejects
    },
    url: `http://127.0.0.1:${port}`
  }
}

async function launchAgainst(sandbox: Sandbox): Promise<{ app: ElectronApplication; page: Page }> {
  const launched = await launchDesktop(
    buildAppEnv(sandbox, {
      HERMES_DESKTOP_APP_NAME: APP_NAME,
      HERMES_DESKTOP_BOOT_FAKE_ERROR: 'E2E native password auth: local backend intentionally not started'
    })
  )

  await launched.page.waitForFunction(
    () =>
      Boolean(
        (window as unknown as { hermesDesktop?: Record<string, unknown> }).hermesDesktop?.oauthLoginConnectionConfig
      ),
    undefined,
    { timeout: 60_000 }
  )

  return launched
}

async function driveSystemBrowser(app: ElectronApplication): Promise<void> {
  await app.evaluate(({ shell }) => {
    shell.openExternal = async (url: string) => {
      const authorize = await fetch(url, { redirect: 'manual' })
      const location = authorize.headers.get('location')

      if (!location) {
        throw new Error(`Native authorize did not redirect (status ${authorize.status}).`)
      }

      const callback = await fetch(new URL(location, url))

      if (!callback.ok) {
        throw new Error(`Native loopback callback failed (status ${callback.status}).`)
      }
    }
  })
}

async function waitForDesktopBridge(page: Page): Promise<void> {
  await page.waitForFunction(
    () =>
      Boolean((window as unknown as { hermesDesktop?: Record<string, unknown> }).hermesDesktop?.testConnectionConfig),
    undefined,
    { timeout: 60_000 }
  )
}

async function testRemoteGateway(page: Page, remoteUrl: string): Promise<{ error: string | null; ok: boolean }> {
  return page.evaluate(async url => {
    const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

    try {
      const result = await desktop.testConnectionConfig({
        mode: 'remote',
        remoteAuthMode: 'oauth',
        remoteUrl: url
      })

      return { error: null, ok: result.ok === true }
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error), ok: false }
    }
  }, remoteUrl)
}

test('password-gateway login survives Cmd+R and a full app restart', async () => {
  test.setTimeout(180_000)
  allowErrorBanners()

  const gateway = await startFakePasswordGateway()
  const sandbox = createSandbox('native-password-auth')
  let app: ElectronApplication | null = null

  try {
    const first = await launchAgainst(sandbox)
    app = first.app
    await driveSystemBrowser(app)

    const login = await first.page.evaluate(async url => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      return desktop.oauthLoginConnectionConfig(url)
    }, gateway.url)

    expect(login.connected, 'the app must hold a reusable credential after login').toBe(true)

    // Cmd+R: a cold renderer has no component/store state from the login.
    await first.page.reload()
    await waitForDesktopBridge(first.page)

    const afterReload = await testRemoteGateway(first.page, gateway.url)

    expect(
      { error: afterReload.error, noCookieRejects: gateway.noCookieRejects, ok: afterReload.ok },
      'Cmd+R must not fall back to the cookie-only `401 no_cookie` path'
    ).toEqual({ error: null, noCookieRejects: 0, ok: true })

    // Full relaunch: only userData survives, proving the token came off disk.
    await app.close()
    app = null

    const second = await launchAgainst(sandbox)
    app = second.app

    const afterRestart = await testRemoteGateway(second.page, gateway.url)

    expect(afterRestart.error, 'the persisted native token must authenticate after relaunch').toBeNull()
    expect(afterRestart.ok).toBe(true)
    expect(gateway.nativeProviders).toEqual(['basic'])
    expect(gateway.embeddedLogins).toBe(0)
    expect(gateway.noCookieRejects).toBe(0)
    expect(gateway.bearerTicketMints).toBe(2)
  } finally {
    await app?.close().catch(() => undefined)
    await gateway.close()
    sandbox.cleanup()
  }
})
