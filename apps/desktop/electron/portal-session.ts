import type { BrowserWindow, BrowserWindowConstructorOptions, Session } from 'electron'

import { cookiesHavePrivyAccessToken, cookiesHavePrivySession } from './connection-config'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'

interface PortalSessionDependencies {
  isReady: () => boolean
  getOauthSession: () => Session | null
  resolvePortalBaseUrl: () => string
  warmOauthCookieStore: () => Promise<unknown>
  createWindow: (options: BrowserWindowConstructorOptions) => BrowserWindow
  rememberLog: (message: string) => void
}

// Portal credentials belong to NAS, independently of the selected gateway.
// Read the jar on every operation so provider changes never latch in Desktop.
export function createPortalSession({
  isReady,
  getOauthSession,
  resolvePortalBaseUrl,
  warmOauthCookieStore,
  createWindow,
  rememberLog
}: PortalSessionDependencies) {
  // Whether the OAuth partition currently holds a live Nous portal session — the
  // credential that powers both discovery and the silent cascade. The portal
  // authenticates via PRIVY, not the Hermes gateway session cookies, so this
  // checks for the `privy-token` cookie on the portal host (NOT
  // hasLiveOauthSession, which looks for hermes_session_at/rt that the portal
  // never sets). See connection-config.ts cookiesHavePrivySession.
  //
  // Mirrors hasLiveOauthSession's cold-start guard (#73495): a `persist:`
  // partition's cookie store hydrates lazily, so the FIRST read on a fresh boot
  // can come back empty even for a signed-in user. The renderer checks Cloud
  // status exactly once on entering cloud mode, so a single false-negative here
  // used to clear the discovered agent list and demand a re-login that a plain
  // retry would have avoided. Warm the store and re-read with a short backoff
  // before trusting a negative.
  async function hasLivePortalSession() {
    const sess = getOauthSession()

    if (!sess) {
      return false
    }

    const portalBaseUrl = resolvePortalBaseUrl()
    const parsed = new URL(portalBaseUrl)

    const readPortal = async () => {
      try {
        const cookies = await sess.cookies.get({ url: portalBaseUrl })

        return cookiesHavePrivySession(cookies)
      } catch {
        try {
          const cookies = await sess.cookies.get({ domain: parsed.hostname })

          return cookiesHavePrivySession(cookies)
        } catch {
          return false
        }
      }
    }

    if (await readPortal()) {
      return true
    }

    await warmOauthCookieStore()

    for (const delayMs of [30, 60, 90]) {
      if (await readPortal()) {
        return true
      }

      await new Promise(resolve => setTimeout(resolve, delayMs))
    }

    return readPortal()
  }

  // Whether the jar holds the short-lived Privy ACCESS token — the exact cookie
  // `/api/agents` validates. hasLivePortalSession() answers "signed in at all?"
  // (renewal material counts); this answers "can discovery succeed right now?".
  async function hasPortalAccessToken() {
    const sess = getOauthSession()

    if (!sess) {
      return false
    }

    const portalBaseUrl = resolvePortalBaseUrl()
    const parsed = new URL(portalBaseUrl)

    try {
      const cookies = await sess.cookies.get({ url: portalBaseUrl })

      return cookiesHavePrivyAccessToken(cookies)
    } catch {
      try {
        const cookies = await sess.cookies.get({ domain: parsed.hostname })

        return cookiesHavePrivyAccessToken(cookies)
      } catch {
        return false
      }
    }
  }

  // Bounded silent renewal of the short-lived Privy access token (#73495).
  //
  // After a Desktop restart the long-lived `privy-session` / `privy-refresh-token`
  // cookies routinely survive while the ~1h `privy-token` access cookie has
  // expired. Discovery then 401s and the only offered recovery used to be a full
  // interactive re-login — even though the persisted refresh material can mint a
  // fresh access token with no user action: loading any portal page runs the
  // Privy client, which rotates a new `privy-token` from the refresh session.
  //
  // This drives exactly that, headlessly: a hidden window on the portal root in
  // the OAuth partition, polled until the access cookie lands, torn down on a
  // bounded timeout. Never shown — if renewal can't complete silently the caller
  // falls back to the interactive needsCloudLogin path. The in-flight promise is
  // shared so concurrent discovery + cascade calls ride one renewal.
  let portalAccessRenewal: Promise<boolean> | null = null

  function renewPortalAccessSilently() {
    if (portalAccessRenewal) {
      return portalAccessRenewal
    }

    portalAccessRenewal = (async () => {
      if (!isReady()) {
        return false
      }

      const sess = getOauthSession()

      if (!sess) {
        return false
      }

      // No renewal material at all → nothing to renew; interactive login is
      // genuinely required.
      if (!(await hasLivePortalSession())) {
        return false
      }

      if (await hasPortalAccessToken()) {
        return true
      }

      const portalBaseUrl = resolvePortalBaseUrl()

      return await new Promise<boolean>(resolve => {
        let settled = false
        let win = null
        let pollTimer = null
        let deadlineTimer = null

        const finish = (ok: boolean) => {
          if (settled) {
            return
          }

          settled = true

          if (pollTimer) {
            clearInterval(pollTimer)
          }

          if (deadlineTimer) {
            clearTimeout(deadlineTimer)
          }

          try {
            if (win && !win.isDestroyed()) {
              win.destroy()
            }
          } catch {
            // window already torn down
          }

          rememberLog(`[cloud] silent portal access renewal ${ok ? 'succeeded' : 'did not complete'}`)
          resolve(ok)
        }

        const checkCookie = async () => {
          if (settled) {
            return
          }

          if (await hasPortalAccessToken()) {
            finish(true)
          }
        }

        try {
          win = createWindow({
            width: 520,
            height: 720,
            show: false,
            title: 'Renewing Hermes Cloud session…',
            autoHideMenuBar: true,
            webPreferences: {
              contextIsolation: true,
              nodeIntegration: false,
              sandbox: true,
              session: sess,
              webSecurity: true
            }
          })
        } catch {
          finish(false)

          return
        }

        win.webContents.on('did-navigate', () => void checkCookie())
        win.webContents.on('did-redirect-navigation', () => void checkCookie())
        win.webContents.on('did-frame-navigate', () => void checkCookie())
        installWindowRendererLifecycle(win, { kind: 'portal-renew', callbacks: { log: rememberLog } })
        pollTimer = setInterval(() => void checkCookie(), 500)
        // Hard deadline: this window is never revealed, so an unrenewable session
        // (revoked refresh token, portal down) must resolve false rather than
        // hang the discovery call behind an invisible window.
        deadlineTimer = setTimeout(() => finish(false), 12_000)

        win.on('closed', () => finish(false))

        win.loadURL(portalBaseUrl).catch(() => finish(false))
      })
    })().finally(() => {
      portalAccessRenewal = null
    }) as Promise<boolean>

    return portalAccessRenewal
  }

  // Drive a one-time interactive portal sign-in in the OAuth partition. Unlike
  // openOauthLoginWindow (which targets a gateway's /login), this lands on the
  // portal itself so the resulting session cookie is portal-scoped — the cookie
  // that authenticates discovery AND is reused for every silent per-agent
  // cascade. Resolves once the portal session cookie appears.
  function openPortalLoginWindow() {
    const portalBaseUrl = resolvePortalBaseUrl()

    return new Promise((resolve, reject) => {
      if (!isReady()) {
        reject(new Error('Desktop is not ready to start a Hermes Cloud sign-in.'))

        return
      }

      const sess = getOauthSession()

      if (!sess) {
        reject(new Error('OAuth session partition is unavailable.'))

        return
      }

      let settled = false
      let win = null
      let pollTimer = null

      const finish = err => {
        if (settled) {
          return
        }

        settled = true

        if (pollTimer) {
          clearInterval(pollTimer)
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
          resolve({ portalBaseUrl, ok: true })
        }
      }

      const checkCookie = async () => {
        if (settled) {
          return
        }

        // A live portal (Privy) session cookie means sign-in completed.
        if (await hasLivePortalSession()) {
          finish(null)
        }
      }

      try {
        win = createWindow({
          width: 520,
          height: 720,
          title: 'Sign in to Hermes Cloud',
          autoHideMenuBar: true,
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

      win.webContents.on('did-navigate', () => void checkCookie())
      win.webContents.on('did-redirect-navigation', () => void checkCookie())
      win.webContents.on('did-frame-navigate', () => void checkCookie())
      // Log-only lifecycle diagnostics, same rationale as the OAuth window:
      // a crashed portal sign-in renderer never settles the promise, so the
      // failure would otherwise leave no trace in desktop.log (#81290
      // follow-up).
      installWindowRendererLifecycle(win, { kind: 'portal', callbacks: { log: rememberLog } })
      pollTimer = setInterval(() => void checkCookie(), 750)

      win.on('closed', () => {
        if (!settled) {
          finish(new Error('Sign-in window closed before authentication completed.'))
        }
      })

      // Land on the portal root; any authenticated portal page sets the session
      // cookie. We only care that the partition cookie jar is populated.
      win.loadURL(portalBaseUrl).catch(error => {
        finish(error instanceof Error ? error : new Error(String(error)))
      })
    })
  }

  return { hasLivePortalSession, hasPortalAccessToken, renewPortalAccessSilently, openPortalLoginWindow }
}
