/**
 * native-oauth-login.ts
 *
 * Electron-coupled driver for the RFC 8252 native-app login: it runs the
 * loopback HTTP listener that catches the gateway's browser redirect, opens
 * the system browser, redeems the one-time code for tokens, and hands them
 * back. The PURE logic (PKCE, URL building, callback parsing, token-response
 * normalization) lives in native-oauth.ts and is unit-tested separately; this
 * module is the thin I/O shell around it.
 *
 * Dependencies are INJECTED (openExternal, a JSON-POST fn, an http-server
 * factory, a clock) so the orchestration is testable without booting Electron
 * or opening real sockets — mirroring how connection-config.ts injects
 * `mintTicket`. main.ts supplies the real electron shell.openExternal,
 * electron.net POST, and node:http server.
 *
 * Security posture (see native-oauth.ts for the flow-level rationale):
 *   - the loopback server binds 127.0.0.1 on an EPHEMERAL port and shuts down
 *     the instant it receives the callback (or times out) — no long-lived
 *     local listener;
 *   - the `state` is verified before the code is redeemed (CSRF);
 *   - the PKCE verifier never leaves this process until the token POST, and
 *     the gateway enforces SHA256(verifier)==challenge server-side;
 *   - the browser sees only a minimal "you can close this window" HTML page,
 *     never the tokens.
 */

import http from 'node:http'
import type { AddressInfo } from 'node:net'

import {
  buildNativeAuthorizeUrl,
  generatePkcePair,
  generateState,
  type NativeTokenSet,
  nativeTokenUrl,
  parseLoopbackCallback,
  parseTokenResponse
} from './native-oauth'

// Loopback login must complete inside this window (user opens browser,
// authenticates, gets redirected back). Matches the server-side pending TTL.
const DEFAULT_LOGIN_TIMEOUT_MS = 5 * 60 * 1000

// The minimal page the browser lands on after the gateway redirect. No tokens,
// no secrets — just a close affordance. Served for any loopback request so a
// favicon probe doesn't look like a failure.
const DONE_HTML =
  '<!doctype html><meta charset="utf-8"><title>Signed in</title>' +
  '<body style="font:15px system-ui;margin:3rem;text-align:center">' +
  '<h2>&#10003; Signed in to Hermes</h2>' +
  '<p>You can close this window and return to the app.</p>' +
  '<script>setTimeout(()=>window.close(),800)</script>'

export interface NativeLoginDeps {
  /** Open a URL in the user's system browser (shell.openExternal). */
  openExternal: (url: string) => Promise<void>
  /** POST JSON and resolve the parsed body (electron.net-backed in prod). */
  postJson: (url: string, body: unknown, opts?: { timeoutMs?: number }) => Promise<any>
  /** http.createServer, injectable for tests. */
  createServer?: typeof http.createServer
  /** Clock + timeout, injectable for tests. */
  now?: () => number
  timeoutMs?: number
  /** Optional logger for boot diagnostics. */
  rememberLog?: (line: string) => void
}

/**
 * Drive a full native login against `baseUrl` and return the token set.
 *
 * Steps: bind a loopback listener → open the system browser at the gateway's
 * /auth/native/authorize with our PKCE challenge + loopback redirect_uri →
 * await the ?code= redirect → verify state → POST /auth/native/token with the
 * verifier → return tokens. Rejects on timeout, state mismatch, a gateway
 * error param, or a token-exchange failure. Always tears the listener down.
 */
export async function runNativeLogin(
  baseUrl: string,
  deps: NativeLoginDeps,
  opts: { provider?: string } = {}
): Promise<NativeTokenSet> {
  const createServer = deps.createServer || http.createServer
  const timeoutMs = deps.timeoutMs ?? DEFAULT_LOGIN_TIMEOUT_MS
  const log = deps.rememberLog || (() => undefined)

  const { verifier, challenge } = generatePkcePair()
  const state = generateState()

  return new Promise<NativeTokenSet>((resolve, reject) => {
    let settled = false
    let timer: NodeJS.Timeout | null = null

    const server = createServer((req, res) => {
      // Only the callback path carries the code; any other path (favicon,
      // etc.) still gets the friendly page so the browser tab looks sane.
      const url = req.url || '/'

      // Always answer the browser with the close page — we never surface the
      // outcome to the browser, only to the app.
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(DONE_HTML)

      if (settled) {
        return
      }

      // Ignore non-callback noise (e.g. /favicon.ico) — wait for the ?code=.
      if (!/[?&](code|error)=/.test(url)) {
        return
      }

      try {
        const { code } = parseLoopbackCallback(url, state)
        finishWith(async () => {
          const tokenBody = await deps.postJson(
            nativeTokenUrl(baseUrl),
            { code, code_verifier: verifier },
            { timeoutMs: 15_000 }
          )

          return parseTokenResponse(tokenBody)
        })
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)))
      }
    })

    const cleanup = () => {
      if (timer) {
        clearTimeout(timer)
      }

      try {
        server.close()
      } catch {
        // already closed
      }
    }

    const fail = (error: Error) => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      reject(error)
    }

    const finishWith = (produce: () => Promise<NativeTokenSet>) => {
      if (settled) {
        return
      }

      settled = true
      // Keep the listener up just long enough to have answered the browser,
      // then redeem the code out-of-band.
      produce()
        .then(tokens => {
          cleanup()
          resolve(tokens)
        })
        .catch(error => {
          cleanup()
          reject(error instanceof Error ? error : new Error(String(error)))
        })
    }

    server.on('error', err => fail(err instanceof Error ? err : new Error(String(err))))

    // Bind an ephemeral loopback port, then open the browser.
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address() as AddressInfo | null

      if (!addr || typeof addr === 'string') {
        fail(new Error('Failed to bind loopback listener for native login'))

        return
      }

      const redirectUri = `http://127.0.0.1:${addr.port}/callback`

      const authorizeUrl = buildNativeAuthorizeUrl(baseUrl, {
        challenge,
        redirectUri,
        state,
        provider: opts.provider
      })

      timer = setTimeout(() => {
        fail(
          new Error(
            'Native sign-in timed out. The browser window may not have completed ' +
              'sign-in; open Settings → Gateway and try again.'
          )
        )
      }, timeoutMs)

      log(`[native-oauth] loopback listening on 127.0.0.1:${addr.port}; opening system browser`)

      deps.openExternal(authorizeUrl).catch(error => {
        fail(
          new Error(
            `Could not open the system browser for native sign-in: ${
              error instanceof Error ? error.message : String(error)
            }`
          )
        )
      })
    })
  })
}

export { DEFAULT_LOGIN_TIMEOUT_MS }
