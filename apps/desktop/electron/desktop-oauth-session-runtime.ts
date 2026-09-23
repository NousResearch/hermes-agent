import type * as electron from 'electron'

import { cookiesHaveLiveSession, cookiesHaveSession, normalizeRemoteBaseUrl } from './connection-config'
import { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } from './hardening'
import { serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'
import { LEGACY_OAUTH_PARTITION, resolveOauthPartition } from './oauth-partition'
import { wireOauthSessionResponse } from './oauth-session-response'
import { oauthLoginLoadUrlOptions } from './remote-ws-headers'

type Session = electron.Session

export interface DesktopOauthSessionDeps {
  app: typeof electron.app
  BrowserWindow: typeof electron.BrowserWindow
  electronNet: typeof electron.net
  session: typeof electron.session
  readDesktopConnectionsRegistry: () => any
  readDesktopConnectionConfig: () => any
  installRemoteHeaderRulesOnSession: (session: Session) => void
  headersForRemoteRequest: (url: string) => Record<string, string>
  rememberLog: (message: string) => void
  installWindowRendererLifecycle: typeof import('./window-renderer-lifecycle').installWindowRendererLifecycle
}

// The legacy cookie jar remains shared by the primary v1 gateway and portal.
// Non-primary registry gateways retain their own partition and hydration state.
export function createDesktopOauthSessionRuntime(deps: DesktopOauthSessionDeps) {
  const {
    app,
    BrowserWindow,
    electronNet,
    session,
    readDesktopConnectionsRegistry,
    readDesktopConnectionConfig,
    installRemoteHeaderRulesOnSession,
    headersForRemoteRequest,
    rememberLog,
    installWindowRendererLifecycle
  } = deps
  let oauthSession = null

  const OAUTH_SESSION_PARTITION = LEGACY_OAUTH_PARTITION

  function getOauthSession() {
    if (oauthSession || !app.isReady()) {
      return oauthSession
    }

    oauthSession = session.fromPartition(OAUTH_SESSION_PARTITION)
    installRemoteHeaderRulesOnSession(oauthSession)

    return oauthSession
  }

  // Per-connection cookie jars (#92183). A NON-primary v2 registry remote with
  // cookie auth rides its own partition so two registered gateways can never
  // evict — or be handed — each other's session cookies (Chromium jars ignore
  // the port, so two dashboards on one VPN host used to collide in the shared
  // jar above). The primary / v1 remote / cloud / portal flows keep the legacy
  // shared partition; see oauth-partition.ts for the full rules.
  const oauthSessionsByPartition = new Map()

  function resolveOauthPartitionForUrl(url) {
    try {
      return resolveOauthPartition(url, {
        registry: readDesktopConnectionsRegistry(),
        v1RemoteUrl: readDesktopConnectionConfig()?.remote?.url
      })
    } catch {
      // A broken registry read must never take cookie auth down with it.
      return OAUTH_SESSION_PARTITION
    }
  }

  function getOauthSessionForUrl(url) {
    const partition = resolveOauthPartitionForUrl(url)

    if (partition === OAUTH_SESSION_PARTITION) {
      return getOauthSession()
    }

    if (!app.isReady()) {
      return null
    }

    let sess = oauthSessionsByPartition.get(partition)

    if (!sess) {
      sess = session.fromPartition(partition)
      oauthSessionsByPartition.set(partition, sess)
      installRemoteHeaderRulesOnSession(sess)
    }

    return sess
  }

  // Cold-start cookie-jar warm-up. A `persist:` partition materialized via
  // session.fromPartition() loads its on-disk cookie store LAZILY: the very first
  // cookies.get() on a fresh cold start can resolve BEFORE the jar has finished
  // hydrating from disk and return an empty array — even though the user is
  // signed in. That false-negative used to make hasLiveOauthSession() report
  // "not signed in", which on the initial boot path (startHermes → the renderer's
  // single-shot boot() with no retry) surfaced as the "Hermes couldn't start"
  // OAuth overlay that vanishes the instant the user clicks Retry.
  //
  // We force the store to hydrate once, up front: flushStorageData() then a
  // throwaway cookies.get(). The promise is memoized so every caller awaits the
  // same single warm-up. Best-effort — any error resolves so we fall back to the
  // live read (which then does its own bounded re-check).
  // Memoized per PARTITION: per-connection jars (#92183) hydrate independently.
  const oauthCookieWarmups = new Map()

  function warmOauthCookieStore(url?) {
    const partition = resolveOauthPartitionForUrl(url)
    const pending = oauthCookieWarmups.get(partition)

    if (pending) {
      return pending
    }

    const warmup = (async () => {
      const sess = getOauthSessionForUrl(url)

      if (!sess) {
        // App not ready yet — don't memoize a no-op; let a later call retry.
        oauthCookieWarmups.delete(partition)

        return
      }

      try {
        // flushStorageData() forces Chromium to reconcile the in-memory cookie
        // monster with the on-disk SQLite store; the subsequent get() then reads
        // a populated jar rather than racing the lazy first-access load.
        sess.flushStorageData?.()
        await sess.cookies.get({})
      } catch {
        // Best effort; the real read below re-checks with bounded retries.
      }
    })()

    oauthCookieWarmups.set(partition, warmup)

    return warmup
  }

  // Bare + prefixed variants of the session cookies live in
  // connection-config.ts (cookiesHaveSession / cookiesHaveLiveSession). See
  // that module for details.

  async function hasOauthSessionCookie(baseUrl) {
    const sess = getOauthSessionForUrl(baseUrl)

    if (!sess) {
      return false
    }

    const parsed = new URL(baseUrl)

    try {
      // Query by URL so the cookie jar applies Domain/Path/Secure scoping for us.
      const cookies = await sess.cookies.get({ url: baseUrl })

      return cookiesHaveSession(cookies)
    } catch {
      // Fall back to a host match if the URL query path errors.
      try {
        const cookies = await sess.cookies.get({ domain: parsed.hostname })

        return cookiesHaveSession(cookies)
      } catch {
        return false
      }
    }
  }

  // Like hasOauthSessionCookie, but returns true when EITHER a live access-token
  // cookie OR a (longer-lived) refresh-token cookie is present. This is the right
  // "is the user signed in at all?" check: an expired AT with a live RT is still
  // a connectable session because the gateway rotates a fresh AT server-side on
  // the next authenticated request. Gating on the AT alone forces a needless full
  // re-login every ~15 min. Used for the Settings "connected" indicator and as a
  // cheap early-out before attempting a network round-trip in resolveRemoteBackend.
  async function hasLiveOauthSession(baseUrl) {
    const sess = getOauthSessionForUrl(baseUrl)

    if (!sess) {
      return false
    }

    const parsed = new URL(baseUrl)

    const readLive = async () => {
      try {
        const cookies = await sess.cookies.get({ url: baseUrl })

        return cookiesHaveLiveSession(cookies)
      } catch {
        try {
          const cookies = await sess.cookies.get({ domain: parsed.hostname })

          return cookiesHaveLiveSession(cookies)
        } catch {
          return false
        }
      }
    }

    // First read against the (possibly still-hydrating) jar.
    if (await readLive()) {
      return true
    }

    // Cold-start false-negative guard. A `persist:` partition's cookie store
    // loads lazily, so the FIRST read on a fresh boot can come back empty even
    // for a signed-in user — the exact race that produced the transient "Hermes
    // couldn't start / not signed in" overlay that Retry always cleared. Before
    // trusting a negative, force the store to hydrate and re-read a couple of
    // times with a short backoff. A genuinely signed-out user still resolves
    // false quickly (≤ ~180ms); a signed-in user racing the load now wins.
    await warmOauthCookieStore(baseUrl)

    for (const delayMs of [30, 60, 90]) {
      if (await readLive()) {
        return true
      }

      await new Promise(resolve => setTimeout(resolve, delayMs))
    }

    return readLive()
  }

  async function clearOauthSession(baseUrl) {
    const sess = getOauthSessionForUrl(baseUrl)

    if (!sess) {
      return
    }

    try {
      const cookies = await sess.cookies.get(baseUrl ? { url: baseUrl } : {})
      await Promise.all(
        cookies.map(c => {
          const scheme = c.secure ? 'https' : 'http'
          const cookieUrl = `${scheme}://${c.domain.replace(/^\./, '')}${c.path || '/'}`

          return sess.cookies.remove(cookieUrl, c.name).catch(() => undefined)
        })
      )
    } catch {
      // Best effort — a stale cookie self-expires anyway.
    }
  }

  // Open a gateway login window in the OAuth session partition, resolving once
  // the access-token cookie appears (login done) or rejecting if the user closes
  // the window first. The window navigates through the IDP and back to
  // /auth/callback, which sets the session cookies on the partition; we poll the
  // cookie jar rather than try to read the HttpOnly value.
  //
  // `silent` selects the URL the window loads, which decides interactive-vs-silent:
  //   - silent=false (default): load ``/login`` — the public interstitial that
  //     renders the "Log in with X" provider chooser. This is the interactive
  //     remote-gateway login the settings UI drives.
  //   - silent=true: load the PROTECTED root ``/`` instead. ``/login`` is a public
  //     route, so loading it NEVER triggers the gate's auto-SSO and always shows
  //     the chooser. Loading a protected page with no session cookie makes the
  //     gate run ``_auto_sso_response``: single registered provider + a live
  //     portal session in this partition → a silent 302 through
  //     ``/auth/login`` → portal ``/oauth/authorize`` (auto-approves org members)
  //     → ``/auth/callback``, which sets the gateway cookie with NO interactive
  //     prompt. This is the per-agent cloud cascade (decisions.md Q5).
  function openOauthLoginWindow(baseUrl, { silent = false } = {}) {
    return new Promise((resolve, reject) => {
      if (!app.isReady()) {
        reject(new Error('Desktop is not ready to start an OAuth login.'))

        return
      }

      const sess = getOauthSessionForUrl(baseUrl)

      if (!sess) {
        reject(new Error('OAuth session partition is unavailable.'))

        return
      }

      let settled = false
      let win = null
      let pollTimer = null
      let revealTimer = null

      const finish = err => {
        if (settled) {
          return
        }

        settled = true

        if (pollTimer) {
          clearInterval(pollTimer)
        }

        if (revealTimer) {
          clearTimeout(revealTimer)
        }

        try {
          if (win && !win.isDestroyed()) {
            win.destroy()
          }
        } catch {
          // window already torn down
        }

        if (err) {
          reject(err)
        } else {
          resolve({ baseUrl, ok: true })
        }
      }

      const checkCookie = async () => {
        if (settled) {
          return
        }

        if (await hasOauthSessionCookie(baseUrl)) {
          finish(null)
        }
      }

      try {
        win = new BrowserWindow({
          width: 520,
          height: 720,
          title: silent ? 'Connecting to Hermes Cloud agent…' : 'Sign in to Hermes gateway',
          autoHideMenuBar: true,
          // Silent cascade: start HIDDEN. The auto-SSO 302 chain completes in
          // well under a second, so the window normally never needs to show. We
          // only reveal it as a fallback if the cascade DOESN'T complete quickly
          // (e.g. the portal session lapsed and the gate fell through to the
          // interactive chooser) — see the reveal timer below.
          show: !silent,
          webPreferences: {
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true,
            session: sess,
            webSecurity: true
          }
        })
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)))

        return
      }

      // Re-check the cookie jar on every successful navigation (the callback
      // redirect is the moment cookies get set) plus a low-frequency poll as a
      // belt-and-braces fallback for IDPs that finish via in-page JS.
      win.webContents.on('did-navigate', () => void checkCookie())
      win.webContents.on('did-redirect-navigation', () => void checkCookie())
      win.webContents.on('did-frame-navigate', () => void checkCookie())
      // Log-only lifecycle diagnostics: a crashed sign-in renderer is invisible
      // to the window's promise path (it never settles), so without this the
      // failure leaves no trace in desktop.log (#81290 follow-up).
      installWindowRendererLifecycle(win, { kind: 'oauth', callbacks: { log: rememberLog } })
      pollTimer = setInterval(() => void checkCookie(), 750)

      // Silent-mode reveal fallback: if the cascade hasn't settled shortly, the
      // auto-SSO didn't go through silently (no portal session, multi-provider,
      // loop-guard tripped, etc.) and the window is now showing an interactive
      // page. Reveal it so the user can complete sign-in manually rather than
      // staring at nothing. Cleared on finish().
      if (silent && win) {
        revealTimer = setTimeout(() => {
          try {
            if (!settled && win && !win.isDestroyed() && !win.isVisible()) {
              win.show()
            }
          } catch {
            // window torn down
          }
        }, 2500)
      }

      win.on('closed', () => {
        if (!settled) {
          finish(new Error('Login window closed before authentication completed.'))
        }
      })

      // ``next`` is intentionally omitted: the gateway lands on ``/`` after
      // login, which is a valid authenticated page that sets the cookies. We
      // only care that the cookie jar is populated.
      //
      // silent=true loads the protected root so the gate auto-SSOs (no chooser);
      // silent=false loads the public ``/login`` chooser for interactive sign-in.
      const normalizedBase = normalizeRemoteBaseUrl(baseUrl)
      const loginUrl = silent ? `${normalizedBase}/` : `${normalizedBase}/login`
      const loginHeaders = headersForRemoteRequest(loginUrl)
      rememberLog(
        `OAuth login: attaching ${Object.keys(loginHeaders).length} extra gateway header(s) to ${new URL(normalizedBase).host}`
      )
      win.loadURL(loginUrl, oauthLoginLoadUrlOptions(loginHeaders)).catch(error => {
        finish(error instanceof Error ? error : new Error(String(error)))
      })
    })
  }

  // JSON request routed through the OAuth session partition so the HttpOnly
  // session cookie is attached automatically by Electron's net stack. Used for
  // authed REST against a gated gateway, including minting WS tickets.
  function fetchJsonViaOauthSession(url, options: any = {}) {
    return new Promise((resolve, reject) => {
      const sess = getOauthSessionForUrl(url)

      if (!sess) {
        reject(new Error('OAuth session partition is unavailable.'))

        return
      }

      let parsed

      try {
        parsed = new URL(url)
      } catch (error) {
        reject(new Error(`Invalid URL: ${error.message}`))

        return
      }

      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

        return
      }

      const body = serializeJsonBody(options.body)
      const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

      const request = electronNet.request({
        method: options.method || 'GET',
        url,
        session: sess,
        useSessionCookies: true,
        redirect: 'follow'
      } as any)

      setJsonRequestHeaders(request)

      for (const [name, value] of Object.entries({ ...headersForRemoteRequest(url), ...(options.headers || {}) })) {
        request.setHeader(name, String(value))
      }

      let timedOut = false

      const timer = setTimeout(() => {
        timedOut = true

        try {
          request.abort()
        } catch {
          // already finished
        }

        reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
      }, timeoutMs)

      request.on('response', res => {
        wireOauthSessionResponse(res, {
          url,
          isTimedOut: () => timedOut,
          clearTimer: () => clearTimeout(timer),
          resolve,
          reject
        })
      })
      request.on('error', error => {
        if (timedOut) {
          return
        }

        clearTimeout(timer)
        reject(error)
      })

      if (body) {
        request.write(body)
      }

      request.end()
    })
  }

  return {
    getOauthSession,
    getOauthSessionForUrl,
    warmOauthCookieStore,
    hasOauthSessionCookie,
    hasLiveOauthSession,
    clearOauthSession,
    openOauthLoginWindow,
    fetchJsonViaOauthSession
  }
}
