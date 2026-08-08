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
  postJson: (url: string, body: unknown, opts?: { signal?: AbortSignal; timeoutMs?: number }) => Promise<any>
  /** http.createServer, injectable for tests. */
  createServer?: typeof http.createServer
  /** Clock + timeout, injectable for tests. */
  now?: () => number
  timeoutMs?: number
  /** Abort the attempt and synchronously release its local resources. */
  signal?: AbortSignal
  /** Optional logger for boot diagnostics. */
  rememberLog?: (line: string) => void
}

export type NativeLoginFailureCode = 'cancelled' | 'stale_attempt' | 'state_mismatch' | 'superseded' | 'timeout'

export class NativeLoginError extends Error {
  readonly code: NativeLoginFailureCode

  constructor(code: NativeLoginFailureCode, message: string) {
    super(message)
    this.name = 'NativeLoginError'
    this.code = code
  }
}

class NativeLoginUnavailableError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'NativeLoginUnavailableError'
  }
}

const STALE_AUTHORIZATION_CODE_DETAIL = 'Invalid or expired authorization code.'

/** Recognize only errors that are safe to turn into native-login recovery. */
export function classifyNativeLoginFailure(error: unknown): NativeLoginFailureCode | null {
  if (error instanceof NativeLoginError) {
    return error.code
  }

  if (!error || typeof error !== 'object' || (error as any).statusCode !== 400) {
    return null
  }

  const message = error instanceof Error ? error.message : String(error)
  const body = message.replace(/^400:\s*/, '')

  try {
    const parsed = JSON.parse(body)

    return parsed?.detail === STALE_AUTHORIZATION_CODE_DETAIL ? 'stale_attempt' : null
  } catch {
    return null
  }
}

export function recoveryActionForNativeLoginFailure(error: unknown): 'ignore' | 'restart' | null {
  const failure = classifyNativeLoginFailure(error)

  if (failure === 'superseded') {
    return 'ignore'
  }

  if (failure === 'cancelled' || failure === 'stale_attempt' || failure === 'state_mismatch' || failure === 'timeout') {
    return 'restart'
  }

  return null
}

/** Only local startup failures may use the intentionally supported embedded flow. */
export function canUseEmbeddedLoginFallback(error: unknown): boolean {
  return error instanceof NativeLoginUnavailableError
}

function classifyLoopbackError(error: unknown): Error {
  const message = error instanceof Error ? error.message : String(error)

  if (message.startsWith('Loopback callback state mismatch')) {
    return new NativeLoginError('state_mismatch', message)
  }

  if (message.startsWith('Gateway rejected native login: access_denied')) {
    return new NativeLoginError('cancelled', message)
  }

  return error instanceof Error ? error : new Error(message)
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
    let processingCallback = false
    let browserLaunchStarted = false
    let timer: NodeJS.Timeout | null = null
    let listenerClosed = false

    const server = createServer((req, res) => {
      // Only the callback path carries the code; any other path (favicon,
      // etc.) still gets the friendly page so the browser tab looks sane.
      const url = req.url || '/'

      // Always answer the browser with the close page — we never surface the
      // outcome to the browser, only to the app.
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(DONE_HTML)

      if (settled || processingCallback) {
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
            { signal: deps.signal, timeoutMs: 15_000 }
          )

          return parseTokenResponse(tokenBody)
        })
      } catch (error) {
        fail(classifyLoopbackError(error))
      }
    })

    const closeListener = () => {
      if (timer) {
        clearTimeout(timer)
        timer = null
      }

      if (listenerClosed) {
        return
      }

      listenerClosed = true

      try {
        server.close()
      } catch {
        // already closed
      }
    }

    const cleanup = () => {
      closeListener()
      deps.signal?.removeEventListener('abort', cancel)
    }

    function cancel() {
      const reason = deps.signal?.reason

      fail(
        reason instanceof NativeLoginError
          ? reason
          : new NativeLoginError('superseded', 'Native sign-in was replaced by a newer attempt.')
      )
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
      if (settled || processingCallback) {
        return
      }

      processingCallback = true
      // The browser response is complete. Close the one-shot listener before
      // redeeming so replacement can never leave an old callback port alive.
      closeListener()
      produce()
        .then(tokens => {
          if (settled) {
            return
          }

          settled = true
          cleanup()
          resolve(tokens)
        })
        .catch(error => {
          if (settled) {
            return
          }

          settled = true
          cleanup()
          const failure = classifyNativeLoginFailure(error)
          reject(
            failure === 'stale_attempt'
              ? new NativeLoginError(failure, 'Native sign-in expired before its authorization code could be redeemed.')
              : error instanceof Error
                ? error
                : new Error(String(error))
          )
        })
    }

    server.on('error', err => {
      const message = err instanceof Error ? err.message : String(err)

      fail(
        browserLaunchStarted
          ? err instanceof Error
            ? err
            : new Error(message)
          : new NativeLoginUnavailableError(`Could not start the loopback listener for native sign-in: ${message}`)
      )
    })

    deps.signal?.addEventListener('abort', cancel, { once: true })

    if (deps.signal?.aborted) {
      cancel()

      return
    }

    // Bind an ephemeral loopback port, then open the browser.
    server.listen(0, '127.0.0.1', () => {
      if (settled) {
        return
      }

      const addr = server.address() as AddressInfo | null

      if (!addr || typeof addr === 'string') {
        fail(new NativeLoginUnavailableError('Failed to bind loopback listener for native login'))

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
        fail(new NativeLoginError('timeout', 'Native sign-in timed out. Return to Hermes and select Restart sign-in.'))
      }, timeoutMs)

      log(`[native-oauth] loopback listening on 127.0.0.1:${addr.port}; opening system browser`)

      browserLaunchStarted = true
      deps.openExternal(authorizeUrl).catch(error => {
        fail(
          new NativeLoginUnavailableError(
            `Could not open the system browser for native sign-in: ${
              error instanceof Error ? error.message : String(error)
            }`
          )
        )
      })
    })
  })
}

interface ActiveNativeLogin {
  controller: AbortController
  promise: Promise<NativeTokenSet>
}

/** Owns the single live native-login attempt for each normalized gateway. */
export class NativeLoginCoordinator {
  private readonly active = new Map<string, ActiveNativeLogin>()

  start(baseUrl: string, deps: NativeLoginDeps, opts: { provider?: string } = {}): Promise<NativeTokenSet> {
    this.active
      .get(baseUrl)
      ?.controller.abort(new NativeLoginError('superseded', 'Native sign-in was replaced by a newer attempt.'))

    const controller = new AbortController()
    const promise = runNativeLogin(baseUrl, { ...deps, signal: controller.signal }, opts)
    const attempt = { controller, promise }
    this.active.set(baseUrl, attempt)

    void promise
      .finally(() => {
        if (this.active.get(baseUrl) === attempt) {
          this.active.delete(baseUrl)
        }
      })
      .catch(() => undefined)

    return promise
  }

  cancel(baseUrl: string): boolean {
    const attempt = this.active.get(baseUrl)

    if (!attempt) {
      return false
    }

    this.active.delete(baseUrl)
    attempt.controller.abort(new NativeLoginError('cancelled', 'Native sign-in was cancelled.'))

    return true
  }
}

export { DEFAULT_LOGIN_TIMEOUT_MS }
