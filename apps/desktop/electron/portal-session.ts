import type { BrowserWindow, BrowserWindowConstructorOptions, Session } from 'electron'

import { cookiesHavePortalSession, portalAccessCookies, type PortalCookie } from './portal-cookies'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'

interface PortalSessionDependencies {
  isReady: () => boolean
  getOauthSession: () => Session | null
  resolvePortalBaseUrl: () => string
  warmOauthCookieStore: () => Promise<unknown>
  createWindow: (options: BrowserWindowConstructorOptions) => BrowserWindow
  rememberLog: (message: string) => void
}

interface CookieWindowOptions {
  kind: string
  title: string
  show: boolean
  pollMs: number
  deadlineMs?: number
}

type CookieWindowOutcome = 'landed' | 'closed' | 'timeout' | Error

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
    return (await readAccessCookies()).length > 0
  }

  // A portal window has done its job only when the jar holds an access cookie
  // it did not hold when the window opened. Presence alone is not enough: a
  // token the server already rejected can sit unexpired in Chromium's jar, and
  // trusting it would close the login window before the user signs in, or
  // report a renewal that never happened.
  async function hasNewAccessCookie(previous: PortalCookie[]) {
    const access = await readAccessCookies()

    return access.some(cookie => !previous.some(old => old.name === cookie.name && old.value === cookie.value))
  }

  // The portal owns provider selection, provisioning and refresh redirects;
  // Desktop only watches the jar for a new access cookie.
  function driveCookieWindow(sess: Session, previous: PortalCookie[], options: CookieWindowOptions) {
    const portalBaseUrl = resolvePortalBaseUrl()

    return new Promise<CookieWindowOutcome>(resolve => {
      let settled = false
      let win: BrowserWindow | null = null
      let pollTimer: ReturnType<typeof setInterval> | null = null
      let deadlineTimer: ReturnType<typeof setTimeout> | null = null

      const finish = (outcome: CookieWindowOutcome) => {
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

        // Settle first: a destroy() that throws must not leave the caller hanging.
        resolve(outcome)

        if (win && !win.isDestroyed()) {
          win.destroy()
        }
      }

      const checkCookie = async () => {
        if (!settled && (await hasNewAccessCookie(previous))) {
          finish('landed')
        }
      }

      try {
        win = createWindow({
          width: 520,
          height: 720,
          show: options.show,
          title: options.title,
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
      // Log-only lifecycle diagnostics: a crashed portal renderer never settles
      // the promise, so the failure would otherwise leave no trace in
      // desktop.log (#81290 follow-up).
      installWindowRendererLifecycle(win, { kind: options.kind, callbacks: { log: rememberLog } })
      pollTimer = setInterval(() => void checkCookie(), options.pollMs)

      if (options.deadlineMs !== undefined) {
        deadlineTimer = setTimeout(() => finish('timeout'), options.deadlineMs)
      }

      win.on('closed', () => finish('closed'))
      win.loadURL(portalBaseUrl).catch(error => finish(error instanceof Error ? error : new Error(String(error))))
    })
  }

  // Loading the portal lets NAS choose its own refresher: Privy client renewal,
  // or the NAS server-side refresh redirect. Never pin a provider in Desktop.
  // Concurrent callers share one hidden window so they cannot race rotating
  // refresh tokens; a `force` caller (discovery just got a 401 with this very
  // cookie) is never satisfied by a short-circuit, only by a real renewal.
  let portalAccessRenewal: Promise<boolean> | null = null

  async function renewPortalAccessSilently({ force = false }: { force?: boolean } = {}) {
    const sess = getOauthSession()

    if (!isReady() || !sess) {
      return false
    }

    // No renewal material at all → nothing to renew; interactive login is
    // genuinely required.
    if (!(await hasLivePortalSession())) {
      return false
    }

    if (!force && (await hasPortalAccessToken())) {
      return true
    }

    portalAccessRenewal ??= (async () => {
      const previous = await readAccessCookies()

      // Hard deadline: this window is never revealed, so an unrenewable session
      // (revoked refresh token, portal down) must resolve false rather than
      // hang the discovery call behind an invisible window.
      const outcome = await driveCookieWindow(sess, previous, {
        kind: 'portal-renew',
        title: 'Renewing Hermes Cloud session…',
        show: false,
        pollMs: 500,
        deadlineMs: 12_000
      })

      const ok = outcome === 'landed'

      rememberLog(`[cloud] silent portal access renewal ${ok ? 'succeeded' : 'did not complete'}`)

      return ok
    })().finally(() => {
      portalAccessRenewal = null
    })

    return portalAccessRenewal
  }

  // Drive a one-time interactive portal sign-in in the OAuth partition. Unlike
  // openOauthLoginWindow (which targets a gateway's /login), this lands on the
  // portal itself so the resulting session cookie is portal-scoped — the cookie
  // that authenticates discovery AND is reused for every silent per-agent
  // cascade. Resolves once a new access cookie appears; refresh material alone
  // must not close the window before the portal can replace it.
  async function openPortalLoginWindow(): Promise<void> {
    if (!isReady()) {
      throw new Error('Desktop is not ready to start a Hermes Cloud sign-in.')
    }

    const sess = getOauthSession()

    if (!sess) {
      throw new Error('OAuth session partition is unavailable.')
    }

    const outcome = await driveCookieWindow(sess, await readAccessCookies(), {
      kind: 'portal',
      title: 'Sign in to Hermes Cloud',
      show: true,
      pollMs: 750
    })

    if (outcome !== 'landed') {
      throw outcome instanceof Error ? outcome : new Error('Sign-in window closed before authentication completed.')
    }
  }

  return { hasLivePortalSession, hasPortalAccessToken, renewPortalAccessSilently, openPortalLoginWindow }
}
