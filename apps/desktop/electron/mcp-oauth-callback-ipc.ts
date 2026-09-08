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

const DONE_HTML =
  '<!doctype html><meta charset="utf-8"><title>Authorization received</title>' +
  '<body style="font:15px system-ui;margin:3rem;text-align:center">' +
  '<h2>&#10003; Authorization received</h2>' +
  '<p>You can close this window and return to Hermes.</p>' +
  '<script>setTimeout(()=>window.close(),800)</script>'

interface CallbackResult {
  code: null | string
  error: null | string
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
    settle(id, { code: null, error: 'cancelled', state: null })
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
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(DONE_HTML)

      const url = req.url || '/'

      // Ignore favicon and other noise — wait for the ?code= / ?error= hit.
      if (!/[?&](code|error)=/.test(url)) {
        return
      }

      let code: null | string = null
      let state: null | string = null
      let error: null | string = null

      try {
        const parsed = new URL(url, 'http://127.0.0.1')

        code = parsed.searchParams.get('code')
        state = parsed.searchParams.get('state')
        error = parsed.searchParams.get('error')
      } catch {
        error = 'unparseable callback URL'
      }

      settle(id, { code, error, state })
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
      return { code: null, error: 'listener not found', state: null }
    }

    if (entry.result) {
      const result = entry.result

      pending.delete(String(id))

      return result
    }

    const timeout = Math.min(Math.max(Number(timeoutMs) || DEFAULT_WAIT_TIMEOUT_MS, 1000), 15 * 60 * 1000)

    const result = await new Promise<CallbackResult>(resolve => {
      const timer = setTimeout(() => {
        settle(String(id), { code: null, error: 'timeout waiting for OAuth callback', state: null })
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
