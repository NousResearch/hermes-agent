import http from 'node:http'
import type { AddressInfo } from 'node:net'

const CALLBACK_TIMEOUT_MS = 120_000

/**
 * Spin up a temporary HTTP server on a random port to receive the Clerk
 * OAuth redirect from the system browser. This is more reliable than custom
 * protocol handlers (`unitalk://`) which may not be registered everywhere.
 *
 * Returns:
 *   - url: the `http://localhost:<port>/callback` URL (available immediately)
 *   - promise: resolves with the token when received, or rejects on timeout/error
 *   - close: clean up the server (caller must call this on timeout/abort)
 */
export function startLocalhostCallbackServer(): {
  url: string
  promise: Promise<string>
  close: () => void
} {
  let settled = false
  let timeoutId: ReturnType<typeof setTimeout> | null = null
  let resolveToken: (token: string) => void
  let rejectToken: (error: Error) => void

  const tokenPromise = new Promise<string>((resolve, reject) => {
    resolveToken = resolve
    rejectToken = reject
  })

  const server = http.createServer((req, res) => {
    if (settled) {
      res.writeHead(200, { 'Content-Type': 'text/plain' })
      res.end('ok')
      return
    }

    try {
      const parsed = new URL(req.url || '/', 'http://localhost')
      const token = parsed.searchParams.get('token') || ''

      if (parsed.pathname === '/callback' && token) {
        settled = true

        // Respond with a completion page the user can close
        res.writeHead(200, { 'Content-Type': 'text/html' })
        res.end(`
          <!DOCTYPE html>
          <html>
          <head><meta charset="utf-8"><title>Unitalk — Sign-in Complete</title>
          <style>
            body { font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #0a0a0a; color: #e0e0e0; }
            .card { text-align: center; padding: 2.5rem; border-radius: 12px; background: #1a1a1a; max-width: 360px; }
            h1 { font-size: 1.5rem; margin: 0 0 0.5rem; }
            p { font-size: 0.95rem; color: #888; margin: 0; }
          </style>
          </head>
          <body>
            <div class="card">
              <h1>Sign-in Complete</h1>
              <p>You may close this window and return to Unitalk Desktop.</p>
            </div>
          </body>
          </html>
        `)

        clearTimeoutSafely()
        server.close()

        // Persist the token immediately so the renderer state updates
        // even before the caller's caller chains off the promise.
        import('@/store/unitalk-auth').then(({ storeClerkToken }) => {
          storeClerkToken(token)
        }).catch(() => {
          // Dynamic import failure is non-fatal — the token is still
          // delivered through the promise chain.
        })

        resolveToken(token)
        return
      }
    } catch {
      // Malformed URL — treat as missing token
    }

    res.writeHead(400, { 'Content-Type': 'text/plain' })
    res.end('bad request')
  })

  function clearTimeoutSafely() {
    if (timeoutId !== null) {
      clearTimeout(timeoutId)
      timeoutId = null
    }
  }

  let serverUrl = ''

  server.on('listening', () => {
    const addr = server.address() as AddressInfo
    serverUrl = `http://localhost:${addr.port}/callback`

    timeoutId = setTimeout(() => {
      if (settled) return
      settled = true
      server.close()
      rejectToken(new Error('Sign-in timed out after 2 minutes. Please try again.'))
    }, CALLBACK_TIMEOUT_MS)
  })

  server.on('error', err => {
    if (settled) return
    settled = true
    clearTimeoutSafely()
    server.close()
    rejectToken(err)
  })

  // Bind on a random port, localhost only
  server.listen(0, '127.0.0.1')

  // Spin until listening to get the URL, then return
  // The event loop gives us `listening` before we can access serverUrl
  // synchronously, so we use a small polling mechanism:
  const urlPromise = new Promise<string>(resolve => {
    const check = () => {
      if (serverUrl) {
        resolve(serverUrl)
      } else if (server.listening) {
        const addr = server.address() as AddressInfo
        serverUrl = `http://localhost:${addr.port}/callback`
        resolve(serverUrl)
      } else {
        setImmediate(check)
      }
    }
    check()
  })

  return {
    get url() {
      return serverUrl || (
        server.listening
          ? `http://localhost:${(server.address() as AddressInfo).port}/callback`
          : ''
      )
    },
    promise: urlPromise.then(() => tokenPromise),
    close: () => {
      settled = true
      clearTimeoutSafely()
      server.close()
    }
  }
}