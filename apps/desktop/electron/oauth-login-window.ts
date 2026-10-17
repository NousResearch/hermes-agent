/**
 * The embedded gateway sign-in window (cookie-mode OAuth).
 *
 * The window loads the gateway in the connection's cookie partition, follows
 * the IdP round trip back to `/auth/callback`, and settles as soon as the
 * session cookie lands. We poll the jar because the cookie is HttpOnly.
 *
 * `silent` selects the URL:
 *   - false: `/login`, the public provider chooser the settings UI drives.
 *   - true: the protected root `/`. With no session cookie the gate runs
 *     `_auto_sso_response`: single provider + a live portal session in this
 *     partition → 302 through `/auth/login` → portal `/oauth/authorize` →
 *     `/auth/callback`, with no prompt (decisions.md Q5).
 *
 * `background` means nobody asked for this sign-in. The window stays hidden
 * and gives up after `deadlineMs`; it never reveals itself. Otherwise a silent
 * window reveals itself after `revealAfterMs` when auto-SSO did not finish.
 * `reveal()` promotes a hidden attempt when the user's own click joins it.
 */

import type { LoginAttempt } from './oauth-login-flights'

export interface OauthLoginWindowLike {
  destroy: () => void
  focus: () => void
  isDestroyed: () => boolean
  isVisible: () => boolean
  loadURL: (url: string, options?: unknown) => Promise<unknown>
  on: (event: 'closed', listener: () => void) => unknown
  show: () => void
  webContents: { on: (event: string, listener: (...args: any[]) => void) => unknown }
}

export interface OauthLoginWindowDeps<S> {
  createWindow: (options: { session: S; show: boolean; title: string }) => OauthLoginWindowLike
  hasSessionCookie: () => Promise<boolean>
  isExpectedNavigationAbort: (error: unknown) => boolean
  loadOptionsFor: (loginUrl: string) => unknown
  log: (message: string) => void
  onLanded: () => void
  onWindowCreated?: (win: OauthLoginWindowLike) => void
  pollMs?: number
  revealAfterMs?: number
  deadlineMs?: number
}

export interface OauthLoginWindowRequest<S> {
  baseUrl: string
  session: S
  silent: boolean
  background: boolean
}

export type OauthLoginOutcome = { baseUrl: string; ok: boolean }

const SIGN_IN_REQUIRED = 'Silent sign-in did not complete; sign-in required.'

export function startOauthLoginWindow<S>(
  { baseUrl, session, silent, background }: OauthLoginWindowRequest<S>,
  deps: OauthLoginWindowDeps<S>
): LoginAttempt<OauthLoginOutcome> {
  const { pollMs = 750, revealAfterMs = 2_500, deadlineMs = 12_000 } = deps

  let reveal = (): void => {}

  const result = new Promise<OauthLoginOutcome>((resolve, reject) => {
    let hiddenRecovery = background
    let settled = false
    let win: OauthLoginWindowLike | null = null
    let pollTimer: ReturnType<typeof setInterval> | null = null
    let revealTimer: ReturnType<typeof setTimeout> | null = null
    let deadlineTimer: ReturnType<typeof setTimeout> | null = null

    const finish = (err: Error | null) => {
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

      if (err) {
        reject(err)
      } else {
        deps.onLanded()
        resolve({ baseUrl, ok: true })
      }
    }

    const checkCookie = async () => {
      if (!settled && (await deps.hasSessionCookie())) {
        finish(null)
      }
    }

    try {
      win = deps.createWindow({
        session,
        title: silent ? 'Connecting to Hermes Cloud agent…' : 'Sign in to Hermes gateway',
        // Silent windows start hidden: the auto-SSO chain normally finishes
        // in well under a second and never needs to be seen.
        show: !silent && !hiddenRecovery
      })
    } catch (error) {
      finish(error instanceof Error ? error : new Error(String(error)))

      return
    }

    const live = win

    // Auto-SSO that cannot complete ends on the gateway's own chooser. A
    // hidden attempt stops there instead of waiting out its deadline, so the
    // app's Sign in prompt appears promptly.
    const chooserUrl = `${baseUrl.replace(/\/+$/, '')}/login`

    // Re-check on every navigation (the callback redirect sets the cookies)
    // plus a slow poll for IdPs that finish via in-page JS.
    live.webContents.on('did-navigate', (_event: unknown, url?: string) => {
      if (hiddenRecovery && typeof url === 'string' && url.split(/[?#]/)[0] === chooserUrl) {
        void deps.hasSessionCookie().then(present => finish(present ? null : new Error(SIGN_IN_REQUIRED)))

        return
      }

      void checkCookie()
    })
    live.webContents.on('did-redirect-navigation', () => void checkCookie())
    live.webContents.on('did-frame-navigate', () => void checkCookie())
    deps.onWindowCreated?.(live)
    pollTimer = setInterval(() => void checkCookie(), pollMs)

    // Auto-SSO did not finish (no portal session, several providers, loop
    // guard): the window now shows an interactive page the user asked for.
    if (silent && !hiddenRecovery) {
      revealTimer = setTimeout(() => {
        try {
          if (!settled && !live.isDestroyed() && !live.isVisible()) {
            live.show()
          }
        } catch {
          // window torn down
        }
      }, revealAfterMs)
    }

    if (hiddenRecovery) {
      deadlineTimer = setTimeout(
        () => finish(new Error('Silent sign-in did not complete; sign-in required.')),
        deadlineMs
      )
    }

    // The user's click joined this flow: show this window (a second one would
    // replace the gateway's PKCE state cookie) and drop the unattended
    // deadline so they have time to finish.
    reveal = () => {
      if (settled || live.isDestroyed()) {
        return
      }

      hiddenRecovery = false

      if (deadlineTimer) {
        clearTimeout(deadlineTimer)
        deadlineTimer = null
      }

      try {
        live.show()
        live.focus()
      } catch {
        // window torn down
      }
    }

    live.on('closed', () => finish(new Error('Login window closed before authentication completed.')))

    // `next` is omitted on purpose: the gateway lands on `/` after login,
    // which sets the cookies; only the jar matters.
    const loginUrl = silent ? `${baseUrl.replace(/\/+$/, '')}/` : chooserUrl

    live.loadURL(loginUrl, deps.loadOptionsFor(loginUrl)).catch(error => {
      // Callback navigation can abort the original load after setting cookies.
      // Keep the bounded hidden recovery alive long enough to observe them.
      if (hiddenRecovery && deps.isExpectedNavigationAbort(error)) {
        void checkCookie()

        return
      }

      finish(error instanceof Error ? error : new Error(String(error)))
    })
  })

  return { result, reveal: () => reveal() }
}
