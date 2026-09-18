import type { BrowserWindow, BrowserWindowConstructorOptions, Session } from 'electron'

import { cookiesHavePortalAccessToken, cookiesHavePortalSession, portalAccessCookies } from './portal-cookies'
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
  // One reader for every portal-cookie question so the failure rungs cannot
  // drift between callers: URL-scoped first, host-scoped when Chromium rejects
  // the URL form, empty when the jar is unreadable.
  async function readPortalCookies() {
    const sess = getOauthSession()

    if (!sess) {
      return []
    }

    const portalBaseUrl = resolvePortalBaseUrl()

    try {
      return await sess.cookies.get({ url: portalBaseUrl })
    } catch {
      try {
        return await sess.cookies.get({ domain: new URL(portalBaseUrl).hostname })
      } catch {
        return []
      }
    }
  }

  // A persisted Chromium jar hydrates lazily; warm and retry before reporting
  // signed-out on a cold start. Both access and refresh credentials count here.
  async function hasLivePortalSession() {
    if (!getOauthSession()) {
      return false
    }

    const readPortal = async () => cookiesHavePortalSession(await readPortalCookies())

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

  async function readAccessCookies() {
    return portalAccessCookies(await readPortalCookies())
  }

  async function hasPortalAccessToken() {
    return cookiesHavePortalAccessToken(await readAccessCookies())
  }

  // Loading the portal lets NAS choose its own refresher: Privy client renewal,
  // or the NAS server-side refresh redirect. Never pin a provider in Desktop.
  // Share a single renewal so concurrent calls cannot race rotating refresh tokens.
  let portalAccessRenewal: Promise<boolean> | null = null

  function renewPortalAccessSilently({ force = false } = {}) {
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

      const previousAccess = await readAccessCookies()

      if (!force && previousAccess.length > 0) {
        return true
      }

      const portalBaseUrl = resolvePortalBaseUrl()

      return await new Promise<boolean>(resolve => {
        let settled = false
        let win: BrowserWindow | null = null
        let pollTimer: ReturnType<typeof setInterval> | null = null
        let deadlineTimer: ReturnType<typeof setTimeout> | null = null

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

          const access = await readAccessCookies()

          // A rejected token still in Chromium's jar is not a successful renewal.
          if (
            access.some(cookie => !previousAccess.some(old => old.name === cookie.name && old.value === cookie.value))
          ) {
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
      let win: BrowserWindow | null = null
      let pollTimer: ReturnType<typeof setInterval> | null = null

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

        // Refresh material alone must not close the window before the portal
        // can replace it with usable access, regardless of the login provider.
        if (await hasPortalAccessToken()) {
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

      // The portal owns provider selection, provisioning and refresh redirects.
      win.loadURL(portalBaseUrl).catch(error => {
        finish(error instanceof Error ? error : new Error(String(error)))
      })
    })
  }

  return { hasLivePortalSession, hasPortalAccessToken, renewPortalAccessSilently, openPortalLoginWindow }
}
