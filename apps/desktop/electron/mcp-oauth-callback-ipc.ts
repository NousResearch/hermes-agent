/**
 * mcp-oauth-callback-ipc.ts
 *
 * Client-side loopback callback listener for MCP OAuth against a REMOTE
 * backend. The gateway's own `mcp.servers.oauth.start` flow binds its
 * callback listener on the BACKEND machine's 127.0.0.1 — unreachable from
 * the user's browser when Desktop connects over SSH/Tailscale, so the
 * provider redirect dies on the user's machine and the flow times out.
 *
 * This module gives the renderer the same primitive the native gateway
 * login uses (native-oauth-login.ts): bind an ephemeral one-shot listener
 * on the USER'S loopback, hand its URL to the gateway as the OAuth
 * redirect_uri (`client_redirect_uri` on oauth.start), and resolve with the
 * redirect's `code`/`state` so the renderer can relay them via
 * `mcp.servers.oauth.callback`.
 *
 * Security posture:
 *   - binds 127.0.0.1 on an ephemeral port; closes on first callback,
 *     cancel, or timeout — no long-lived listener;
 *   - the listener only ever RECEIVES `code`/`state` query params and
 *     forwards them to the renderer; no tokens are exchanged here — the
 *     gateway verifies `state` (constant-time) before redeeming anything;
 *   - the browser sees only a minimal "return to Hermes" page.
 */

import http from 'node:http'
import type { AddressInfo } from 'node:net'

import { ipcMain } from 'electron'

const DEFAULT_WAIT_TIMEOUT_MS = 5 * 60 * 1000
const MAX_PENDING_LISTENERS = 8
// A form-post redirect carries code/state/iss; anything larger than this is not one.
const FORM_POST_MAX_CHARS = 16 * 1024

const DONE_HTML =
  '<!doctype html><meta charset="utf-8"><title>Authorization received</title>' +
  '<body style="font:15px system-ui;margin:3rem;text-align:center">' +
  '<h2>&#10003; Authorization received</h2>' +
  '<p>You can close this window and return to Hermes.</p>' +
  '<script>setTimeout(()=>window.close(),800)</script>'

interface CallbackResult {
  code: null | string
  error: null | string
  // RFC 9207 issuer identifier. Cloudflare and Resend advertise
  // `authorization_response_iss_parameter_supported`, and mcp 2.x rejects an
  // authorization response that omits it — relay it rather than dropping it.
  iss: null | string
  state: null | string
}

interface PendingListener {
  result: CallbackResult | null
  server: http.Server
  settled: boolean
  waiters: Array<(result: CallbackResult) => void>
}

const pending = new Map<string, PendingListener>()
let nextId = 1

function settle(id: string, result: CallbackResult) {
  const entry = pending.get(id)

  if (!entry || entry.settled) {
    return
  }

  entry.settled = true
  entry.result = result

  try {
    entry.server.close()
  } catch {
    // already closed
  }

  for (const waiter of entry.waiters.splice(0)) {
    waiter(result)
  }
}

function dispose(id: string) {
  const entry = pending.get(id)

  if (!entry) {
    return
  }

  if (!entry.settled) {
    settle(id, { code: null, error: 'cancelled', iss: null, state: null })
  }

  pending.delete(id)
}

export function registerMcpOauthCallbackIpc() {
  // Bind a one-shot loopback listener; resolves { id, redirectUri }.
  ipcMain.handle('hermes:mcp-oauth:listen', async () => {
    if (pending.size >= MAX_PENDING_LISTENERS) {
      throw new Error('Too many MCP OAuth listeners are already pending')
    }

    const id = String(nextId++)

    const server = http.createServer((req, res) => {
      const answer = (query: string) => {
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
        res.end(DONE_HTML)

        // Ignore favicon and other noise — wait for the code= / error= hit.
        if (!/(^|&)(code|error)=/.test(query)) {
          return
        }

        let code: null | string = null
        let state: null | string = null
        let error: null | string = null
        let iss: null | string = null

        try {
          const params = new URLSearchParams(query)

          code = params.get('code')
          state = params.get('state')
          error = params.get('error')
          iss = params.get('iss')
        } catch {
          error = 'unparseable callback URL'
        }

        settle(id, { code, error, iss, state })
      }

      if (req.method !== 'POST') {
        answer((req.url || '/').split('?')[1] || '')

        return
      }

      // response_mode=form_post: the redirect's parameters arrive as an urlencoded body.
      const form = /^application\/x-www-form-urlencoded\b/i.test(req.headers['content-type'] || '')
      let body = ''

      req.setEncoding('utf8')
      req.on('data', (chunk: string) => {
        body += chunk

        if (body.length > FORM_POST_MAX_CHARS) {
          req.destroy()
        }
      })
      req.on('end', () => answer(form ? body : ''))
    })

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(0, '127.0.0.1', () => resolve())
    })

    const port = (server.address() as AddressInfo).port

    pending.set(id, { result: null, server, settled: false, waiters: [] })

    return { id, redirectUri: `http://127.0.0.1:${port}/callback` }
  })

  // Resolve when the redirect arrives (or timeout). Safe to call once per id.
  ipcMain.handle('hermes:mcp-oauth:wait', async (_event, id, timeoutMs) => {
    const entry = pending.get(String(id || ''))

    if (!entry) {
      return { code: null, error: 'listener not found', iss: null, state: null }
    }

    if (entry.result) {
      const result = entry.result

      pending.delete(String(id))

      return result
    }

    const timeout = Math.min(Math.max(Number(timeoutMs) || DEFAULT_WAIT_TIMEOUT_MS, 1000), 15 * 60 * 1000)

    const result = await new Promise<CallbackResult>(resolve => {
      const timer = setTimeout(() => {
        settle(String(id), { code: null, error: 'timeout waiting for OAuth callback', iss: null, state: null })
      }, timeout)

      entry.waiters.push(value => {
        clearTimeout(timer)
        resolve(value)
      })
    })

    pending.delete(String(id))

    return result
  })

  // Tear a listener down without waiting (user cancelled, flow errored).
  ipcMain.handle('hermes:mcp-oauth:cancel', (_event, id) => {
    dispose(String(id || ''))

    return true
  })
}
