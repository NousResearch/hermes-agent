/**
 * apex-loopback.cjs
 *
 * Minimal loopback HTTP server for the ApexNodes Desktop browser-login flows
 * ("用 Google 登录" / "用 APEX 登录", Desktop V0.2). Electron-free (uses only
 * `node:http` + `node:crypto`) so it unit-tests with `node --test`, same pattern
 * as apex-managed.cjs / connection-config.cjs.
 *
 * Why a new server (not the existing OAuth loopback): the desktop's remote-gateway
 * "connect to dashboard" OAuth does NOT run an Electron-side listener — the
 * *backend* binds 127.0.0.1 and the renderer polls a session. The managed-LLM
 * browser flows here have no backend session to poll: the system browser is
 * redirected straight back to the desktop with `?token=<JWT>&state=<s>`, so the
 * desktop itself must catch that redirect. Per the shared contract we add this
 * minimal listener only because none exists to reuse.
 *
 * Security:
 *   - Binds 127.0.0.1 only (never 0.0.0.0) so nothing off-box can hit it.
 *   - A random `state` is generated per flow and validated on the callback
 *     (CSRF). The token is only surfaced on an exact /cb + matching-state hit.
 *   - The server self-closes the moment it resolves (success or first failure),
 *     and a watchdog timeout tears it down if the user never returns.
 */

const crypto = require('node:crypto')
const http = require('node:http')

const { parseLoopbackCallback } = require('./apex-managed.cjs')

// How long to wait for the browser to redirect back before giving up. The user
// has to sign in in a browser tab, so this is generous.
const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000

// Tiny self-contained HTML the browser shows after the redirect. No external
// assets (the desktop has no web server for the loopback to reference).
function resultPage(title, body) {
  return (
    '<!doctype html><html lang="zh"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    `<title>${title}</title>` +
    '<style>html{color-scheme:light dark}body{font-family:-apple-system,BlinkMacSystemFont,' +
    '"Segoe UI",system-ui,sans-serif;display:flex;min-height:100vh;margin:0;align-items:center;' +
    'justify-content:center;text-align:center;padding:2rem}main{max-width:24rem}' +
    'h1{font-size:1.125rem;margin:0 0 .5rem}p{opacity:.7;margin:0;line-height:1.5}</style>' +
    `</head><body><main><h1>${title}</h1><p>${body}</p></main></body></html>`
  )
}

const SUCCESS_HTML = resultPage('登录成功', '已完成登录，请返回 ApexNodes 桌面应用继续。')
const FAILURE_HTML = resultPage('登录失败', '登录未完成，请返回桌面应用重试。')

/**
 * Generate a URL-safe random state token for CSRF protection.
 * @returns {string}
 */
function generateState() {
  return crypto.randomBytes(24).toString('base64url')
}

/**
 * Start a loopback listener and resolve with the token once the browser is
 * redirected back to `/cb?token=<JWT>&state=<state>` (validated). The returned
 * object exposes the bound `redirectUri` (to pass to the start URL builders),
 * the `state`, a `result` promise, and a `close()` to abort early.
 *
 * The `result` promise:
 *   - resolves `{ token }` on a valid callback,
 *   - rejects with an Error (code in `.reason`) on state mismatch / missing
 *     token / explicit ?error= / timeout / socket error.
 *
 * Non-/cb requests (e.g. the browser's automatic /favicon.ico) get a 204 and do
 * NOT resolve or reject — only the real callback settles the flow.
 *
 * @param {{ host?: string, timeoutMs?: number, path?: string }} [options]
 * @returns {Promise<{ redirectUri: string, state: string, port: number,
 *                      result: Promise<{ token: string }>, close: () => void }>}
 */
function startLoopbackLogin(options = {}) {
  const host = options.host || '127.0.0.1'
  const timeoutMs = typeof options.timeoutMs === 'number' ? options.timeoutMs : DEFAULT_TIMEOUT_MS
  const callbackPath = options.path || '/cb'
  const state = generateState()

  return new Promise((resolveStart, rejectStart) => {
    let settled = false
    let resolveResult
    let rejectResult
    const result = new Promise((res, rej) => {
      resolveResult = res
      rejectResult = rej
    })

    let watchdog = null

    const cleanup = () => {
      if (watchdog) {
        clearTimeout(watchdog)
        watchdog = null
      }
      // Force-drop any lingering keep-alive sockets first so close() can release
      // the handle immediately (a browser may hold the connection open after the
      // redirect; without this the listener — and a test runner's event loop —
      // would wait for the idle socket to time out). closeAllConnections is
      // Node 18.2+; guard it for safety. close() is idempotent.
      try {
        if (typeof server.closeAllConnections === 'function') {
          server.closeAllConnections()
        }
      } catch {
        // best effort
      }
      try {
        server.close()
      } catch {
        // already closing / closed
      }
    }

    const fail = reason => {
      if (settled) return
      settled = true
      cleanup()
      const err = new Error(`Loopback login failed: ${reason}`)
      err.reason = reason
      rejectResult(err)
    }

    const succeed = token => {
      if (settled) return
      settled = true
      cleanup()
      resolveResult({ token })
    }

    const server = http.createServer((req, res) => {
      const outcome = parseLoopbackCallback(req.url, state)

      if (outcome.ok) {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
        res.end(SUCCESS_HTML)
        succeed(outcome.token)
        return
      }

      // A failed parse that ISN'T the callback (favicon, stray path) must not
      // settle the flow — answer quietly and keep waiting for the real /cb.
      if (!outcome.isCallback) {
        res.statusCode = 204
        res.end()
        return
      }

      // The /cb callback came back but is invalid (state mismatch, missing
      // token, explicit ?error=) — show the failure page and reject the flow.
      res.writeHead(400, { 'Content-Type': 'text/html; charset=utf-8' })
      res.end(FAILURE_HTML)
      fail(outcome.reason)
    })

    server.on('error', error => {
      if (!settled) {
        // A bind error before we ever started → reject the *start* promise.
        if (!server.listening) {
          rejectStart(error)
          return
        }
        fail(error && error.message ? error.message : String(error))
      }
    })

    // Port 0 → OS assigns a free ephemeral port, returned via address().
    server.listen(0, host, () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      if (!port) {
        try {
          server.close()
        } catch {
          // ignore
        }
        rejectStart(new Error('Loopback server did not bind a port.'))
        return
      }

      watchdog = setTimeout(() => fail('timeout'), timeoutMs)
      // Don't let the watchdog keep the event loop / app alive.
      if (typeof watchdog.unref === 'function') {
        watchdog.unref()
      }

      const redirectUri = `http://${host}:${port}${callbackPath}`
      resolveStart({
        redirectUri,
        state,
        port,
        result,
        close: () => fail('aborted')
      })
    })
  })
}

module.exports = {
  DEFAULT_TIMEOUT_MS,
  generateState,
  startLoopbackLogin
}
