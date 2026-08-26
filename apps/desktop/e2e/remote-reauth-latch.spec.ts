/**
 * E2E regression for issue #95701: a remote OAuth gateway whose stored
 * session has been invalidated server-side must boot into ONE stable
 * recovery overlay with a clickable Sign in action — not alternate between
 * the connecting state and the overlay while the renderer's transient-boot
 * retry loop re-drives a rejection that can never self-heal.
 *
 * The shape under test is the RFC 8252 native-bearer flow: the desktop holds
 * a locally-unexpired access token, but the gateway rejects it (restarted
 * with new signing state), and the refresh token is dead too. The gateway
 * never rotates a native bearer server-side (see
 * hermes_cli/dashboard_auth/middleware.py), so the desktop must try ONE
 * rotation via /auth/native/refresh and then treat the rejection as
 * confirmed: latch reauth, mark the boot non-retryable, hold the overlay.
 *
 * Contract asserted here, end to end through the real Electron main process
 * and renderer:
 *
 *   1. The recovery overlay appears and then stays continuously visible —
 *      no sample over a window longer than the renderer's boot-retry backoff
 *      sees it hidden.
 *   2. The gateway saw exactly one ticket mint and exactly one refresh
 *      attempt: the rejection was confirmed once, not hammered.
 *   3. The boot-progress snapshot the renderer reads is the non-retryable,
 *      structured-401 verdict the reauth latch depends on.
 *   4. The overlay offers the remote sign-in action (dead tokens were dropped,
 *      so Settings no longer reports the session as connected).
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as path from 'node:path'

import { buildAppEnv, createSandbox, launchDesktop, type Sandbox, waitForBootFailure } from './fixtures'
import { allowErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

// Longer than the renderer's first boot-retry delay (2s base, full jitter,
// #82679) by a wide margin, so a retry loop that hides the overlay cannot
// slip between two samples.
const STABILITY_WINDOW_MS = 15_000
const SAMPLE_INTERVAL_MS = 100

interface RecordedRequest {
  method: string
  path: string
  bearer: boolean
}

interface ExpiredSessionGateway {
  url: string
  requests: RecordedRequest[]
  count: (method: string, pathname: string) => number
  close: () => Promise<void>
}

/**
 * A gated gateway whose every authenticated session is dead. Public
 * discovery works (status advertises the native flow, providers are
 * OAuth-style so the desktop keeps the strict guard), but the ticket mint
 * and the native refresh both answer the middleware's structured
 * `session_expired` 401. Every request is recorded so the test can prove the
 * rejection was confirmed exactly once.
 */
async function startExpiredSessionGateway(): Promise<ExpiredSessionGateway> {
  const requests: RecordedRequest[] = []

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1')
    const authorization = req.headers.authorization

    requests.push({
      method: req.method ?? 'GET',
      path: url.pathname,
      bearer: typeof authorization === 'string' && authorization.startsWith('Bearer '),
    })

    const json = (status: number, body: unknown): void => {
      res.writeHead(status, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(body))
    }

    // Drain the body before answering so a POST never sees an early close.
    req.resume()
    req.on('end', () => {
      if (url.pathname === '/api/status') {
        json(200, { auth_required: true, auth_flows: ['native_pkce'], ok: true, version: '0.0.0-e2e-fake' })

        return
      }

      if (url.pathname === '/api/auth/providers') {
        json(200, { providers: [{ name: 'portal', supports_password: false }] })

        return
      }

      if (url.pathname === '/api/auth/ws-ticket' || url.pathname === '/auth/native/refresh') {
        json(401, {
          error: 'session_expired',
          detail: 'Unauthorized',
          reason: 'invalid_or_expired_session',
          login_url: '/login',
        })

        return
      }

      if (url.pathname.startsWith('/api/')) {
        json(401, { error: 'unauthenticated', detail: 'Unauthorized', reason: 'no_cookie', login_url: '/login' })

        return
      }

      json(404, { detail: 'not found' })
    })
  })

  // No WebSocket can ever be authenticated here; refuse the upgrade fast.
  server.on('upgrade', (_req, socket) => {
    socket.destroy()
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  const { port } = server.address() as AddressInfo

  return {
    close: () =>
      new Promise<void>(resolve => {
        server.closeAllConnections?.()
        server.close(() => resolve())
      }),
    count: (method, pathname) => requests.filter(r => r.method === method && r.path === pathname).length,
    requests,
    url: `http://127.0.0.1:${port}`,
  }
}

/**
 * Seed the sandbox the way a previously signed-in desktop leaves it: a saved
 * remote OAuth connection and a stored native token set whose access token
 * has NOT expired locally (so the desktop presents it as-is). The blob uses
 * the plain encoding the desktop's secret reader accepts verbatim — an OS
 * keyring is not available on the CI runner and is irrelevant to the contract.
 */
function seedExpiredNativeSession(sandbox: Sandbox, gatewayUrl: string): void {
  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'connection.json'),
    JSON.stringify({ mode: 'remote', remote: { url: gatewayUrl, authMode: 'oauth' }, profiles: {} }, null, 2),
    { encoding: 'utf8', mode: 0o600 },
  )

  const tokens = {
    accessToken: 'e2e-stale-access-token',
    refreshToken: 'e2e-stale-refresh-token',
    expiresAt: Math.floor(Date.now() / 1000) + 3600,
    provider: 'portal',
    userId: 'e2e-user',
  }

  fs.writeFileSync(
    path.join(sandbox.userDataDir, 'native-oauth-tokens.json'),
    JSON.stringify({ [gatewayUrl]: { encoding: 'plain', value: JSON.stringify(tokens) } }, null, 2),
    { encoding: 'utf8', mode: 0o600 },
  )
}

/**
 * Sample the recovery overlay inside the page at a fixed cadence and report
 * every sample in which it was NOT showing. The overlay is identified by its
 * recovery actions, the same signal waitForBootFailure keys on; while the
 * renderer re-drives a boot the overlay is unmounted (`running: true`) and
 * the connecting surface paints instead.
 */
async function sampleOverlayStability(
  page: Page,
  windowMs: number,
  intervalMs: number,
): Promise<{ samples: number; hidden: number; flips: number }> {
  return page.evaluate(
    async ({ windowMs: w, intervalMs: i }) => {
      const overlayShowing = (): boolean => {
        const text = document.body.textContent ?? ''

        return (
          text.includes('Sign out & sign in') ||
          text.includes('Gateway settings') ||
          text.includes('Use local gateway') ||
          text.includes('Repair install')
        )
      }

      const started = Date.now()
      let samples = 0
      let hidden = 0
      let flips = 0
      let last = overlayShowing()

      while (Date.now() - started < w) {
        await new Promise(resolve => setTimeout(resolve, i))
        const showing = overlayShowing()
        samples += 1

        if (!showing) {
          hidden += 1
        }

        if (showing !== last) {
          flips += 1
          last = showing
        }
      }

      return { samples, hidden, flips }
    },
    { windowMs, intervalMs },
  )
}

let gateway: ExpiredSessionGateway | null = null
let sandbox: Sandbox | null = null
let app: ElectronApplication | null = null

test.afterAll(async () => {
  await app?.close().catch(() => undefined)
  await gateway?.close()
  sandbox?.cleanup()
  app = null
  gateway = null
  sandbox = null
})

test.describe('remote OAuth session rejected at cold boot (#95701)', () => {
  test.beforeEach(() => {
    // The boot deliberately fails; the "Desktop boot failed" toast is expected.
    allowErrorBanners()
  })

  test('the recovery overlay latches once and its Sign in action stays put', async () => {
    gateway = await startExpiredSessionGateway()
    sandbox = createSandbox('reauth-latch')
    seedExpiredNativeSession(sandbox, gateway.url)

    const launched = await launchDesktop(buildAppEnv(sandbox))
    app = launched.app
    const page = launched.page

    await waitForBootFailure(page, 60_000)

    // 1. Stability: once the overlay is up it never blinks out again. A
    //    transient-boot retry (`running: true` re-emitted over the failure)
    //    unmounts it, which is exactly the flicker the issue describes.
    const stability = await sampleOverlayStability(page, STABILITY_WINDOW_MS, SAMPLE_INTERVAL_MS)
    expect(stability.samples).toBeGreaterThan(50)
    expect(stability, 'recovery overlay must stay visible for the whole window').toMatchObject({ hidden: 0, flips: 0 })

    // 2. The rejection was confirmed exactly once: one bearer ticket mint,
    //    one forced rotation attempt, and no further boot re-entry hammering
    //    the dead session for the rest of the window.
    const ticketMints = gateway.requests.filter(r => r.method === 'POST' && r.path === '/api/auth/ws-ticket')
    expect(ticketMints, 'exactly one ws-ticket mint').toHaveLength(1)
    expect(ticketMints[0]?.bearer, 'the mint presented the stored native bearer').toBe(true)
    expect(gateway.count('POST', '/auth/native/refresh'), 'exactly one native refresh attempt').toBe(1)

    // 3. The renderer-visible verdict is the structured, non-retryable one.
    const snapshot = await page.evaluate(() =>
      (window as unknown as { hermesDesktop: { getBootProgress: () => Promise<Record<string, unknown>> } })
        .hermesDesktop.getBootProgress(),
    )
    expect(snapshot).toMatchObject({ running: false, retryable: false, statusCode: 401 })
    // A session that DID exist reads as expired, not as never signed in — the
    // dead tokens are dropped on the way out, so this copy must be chosen
    // from the pre-mint state.
    expect(String(snapshot.error)).toMatch(/session has expired/i)

    // 4. The overlay offers remote sign-in — the dead tokens were dropped, so
    //    the connection no longer reads as "connected" and the reauth branch
    //    renders instead of the local Retry/Repair buttons.
    await expect(page.getByRole('button', { name: /sign out & sign in/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /sign out & sign in/i })).toBeEnabled()
  })
})
