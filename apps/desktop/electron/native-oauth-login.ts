/**
 * native-oauth-login.ts
 *
 * Electron-coupled driver for the RFC 8252 native-app login: it runs the
 * loopback HTTP listener that catches the authorization server's browser
 * redirect, opens the system browser, redeems the one-time code for tokens,
 * and hands them back. The same loopback core drives both a gateway's
 * `/auth/native/*` broker (runNativeLogin) and the portal's `hermes-desktop`
 * client (portal-session.ts). The PURE logic (PKCE, URL building, callback parsing, token-response
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

// The minimal page the browser lands on after the redirect. No tokens, no
// secrets — just a close affordance. Served for any loopback request so a
// favicon probe doesn't look like a failure.
const closePage = (heading: string) =>
  '<!doctype html><meta charset="utf-8"><title>Hermes</title>' +
  '<body style="font:15px system-ui;margin:3rem;text-align:center">' +
  `<h2>${heading}</h2>` +
  '<p>You can close this window and return to the app.</p>' +
  '<script>setTimeout(()=>window.close(),800)</script>'

const DONE_HTML = closePage('&#10003; Signed in to Hermes')
const CANCELLED_HTML = closePage('Sign-in cancelled')

/** The I/O shared by every loopback authorization (gateway or portal). */
export interface LoopbackAuthorizationDeps {
  /** Open a URL in the user's system browser (shell.openExternal). */
  openExternal: (url: string) => Promise<void>
  /** http.createServer, injectable for tests. */
  createServer?: typeof http.createServer
  /** Clock + timeout, injectable for tests. */
  now?: () => number
  timeoutMs?: number
  /** Optional logger for boot diagnostics. */
  rememberLog?: (line: string) => void
}

export interface NativeLoginDeps extends LoopbackAuthorizationDeps {
  /** POST JSON and resolve the parsed body (electron.net-backed in prod). */
  postJson: (url: string, body: unknown, opts?: { timeoutMs?: number }) => Promise<any>
}

/**
 * What differs between authorization servers: where the browser goes, and
 * how the one-time code is redeemed. The loopback redirect URI handed to
 * `redeem` is byte-identical to the one `buildAuthorizeUrl` received, which
 * servers that bind the code to its redirect_uri (the portal, §2) require.
 */
export interface LoopbackAuthorizationFlow<T> {
  buildAuthorizeUrl: (params: { challenge: string; redirectUri: string; state: string }) => string
  redeem: (params: { code: string; verifier: string; redirectUri: string }) => Promise<T>
}

/**
 * The RFC 8252 loopback core: bind 127.0.0.1 on an ephemeral port → open the
 * system browser at the flow's authorize URL with our PKCE challenge + state →
 * await the ?code= (or ?error=) redirect → verify state → redeem. Rejects on
 * timeout, state mismatch, an error param (access_denied rejects with
 * NativeLoginCancelledError), or a redeem failure. Always tears the listener
 * down.
 */
export async function runLoopbackAuthorization<T>(
  deps: LoopbackAuthorizationDeps,
  flow: LoopbackAuthorizationFlow<T>
): Promise<T> {
  const createServer = deps.createServer || http.createServer
  const timeoutMs = deps.timeoutMs ?? DEFAULT_LOGIN_TIMEOUT_MS
  const log = deps.rememberLog || (() => undefined)

  const { verifier, challenge } = generatePkcePair()
  const state = generateState()

  return new Promise<T>((resolve, reject) => {
    let settled = false
    let timer: NodeJS.Timeout | null = null
    let redirectUri = ''

    const server = createServer((req, res) => {
      // Only the callback path carries the code; any other path (favicon,
      // etc.) still gets the friendly page so the browser tab looks sane.
      const url = req.url || '/'

      // Always answer the browser with a close page — we never surface the
      // outcome (let alone tokens) to the browser, only to the app.
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(/[?&]error=/.test(url) ? CANCELLED_HTML : DONE_HTML)

      if (settled) {
        return
      }

      // Ignore non-callback noise (e.g. /favicon.ico) — wait for the ?code=.
      if (!/[?&](code|error)=/.test(url)) {
        return
      }

      try {
        const { code } = parseLoopbackCallback(url, state)
        finishWith(() => flow.redeem({ code, verifier, redirectUri }))
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

    const finishWith = (produce: () => Promise<T>) => {
      if (settled) {
        return
      }

      settled = true
      // Keep the listener up just long enough to have answered the browser,
      // then redeem the code out-of-band.
      produce()
        .then(result => {
          cleanup()
          resolve(result)
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

      redirectUri = `http://127.0.0.1:${addr.port}/callback`

      let authorizeUrl: string

      try {
        authorizeUrl = flow.buildAuthorizeUrl({ challenge, redirectUri, state })
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)))

        return
      }

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

/**
 * Drive a full native login against a GATEWAY's `/auth/native/*` broker and
 * return the token set: authorize at /auth/native/authorize, redeem at
 * /auth/native/token with the verifier.
 */
export async function runNativeLogin(
  baseUrl: string,
  deps: NativeLoginDeps,
  opts: { provider?: string } = {}
): Promise<NativeTokenSet> {
  return runLoopbackAuthorization(deps, {
    buildAuthorizeUrl: ({ challenge, redirectUri, state }) =>
      buildNativeAuthorizeUrl(baseUrl, { challenge, redirectUri, state, provider: opts.provider }),
    redeem: async ({ code, verifier }) =>
      parseTokenResponse(
        await deps.postJson(nativeTokenUrl(baseUrl), { code, code_verifier: verifier }, { timeoutMs: 15_000 })
      )
  })
}

export { DEFAULT_LOGIN_TIMEOUT_MS }
