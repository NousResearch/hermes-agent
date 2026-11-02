import type { BrowserWindow, BrowserWindowConstructorOptions, Session } from 'electron'

import { createWindowOpenHandler } from './window-open-policy'

/**
 * The free tier's browser challenge, desktop half.
 *
 * The Nous account service can ask an anonymous install to clear a challenge
 * before it mints a token (`hermes_cli/anon_challenge.py`). The challenge is a
 * portal page; everything it does (bot detection, an in-browser proof of work,
 * an interactive fallback) belongs to the portal and changes without a Desktop
 * release. Desktop's whole job is this file: load that page in a HIDDEN
 * window, reveal the window only if the page asks for the human, and close it
 * when the page says it is finished. The backend polls the account service on
 * its own and re-mints; nothing here holds a credential.
 *
 * The page talks to us through its URL fragment, which we can read with no
 * bridge into the page:
 *
 *   #working      nothing to show; stay hidden
 *   #interactive  needs the human: reveal
 *   #done         passed: close
 *   #failed       ended without a pass: close (the backend's next mint says why)
 *
 * The page is remote content, so the window is hardened the way the link-title
 * window is: sandboxed, isolated, its own partition, no popups, no downloads,
 * no permissions. It is pinned to the portal origin three ways: navigations
 * and SERVER REDIRECTS off-origin are cancelled, a main-frame commit anywhere
 * else ends the run, and a phase is only ever read from a portal URL — a
 * foreign page must never be able to get this window revealed under a Hermes
 * title. Only `<portal>/challenge…` URLs are ever loaded — the URL arrives from the
 * network by way of the backend and the renderer, and "open this URL" must not
 * become "open any URL".
 */

export type ChallengePhase = 'working' | 'interactive' | 'done' | 'failed'

// The contract's ``free_tier.challenge_result`` outcomes minus the renderer-only 'unsupported';
// the renderer reports these through that type, so a drift fails its typecheck.
export type ChallengeOutcome = 'done' | 'failed' | 'closed' | 'timeout' | 'refused' | 'error'

export interface ChallengeRequest {
  url: string
  /** False for a challenge the service is only measuring with: never revealed. */
  required: boolean
  /** Seconds the ticket has left, as the service reported it. */
  expiresIn?: number
  /** A new foreground attempt may reopen a window the user closed. */
  attempt?: number
}

export interface ChallengeWindowDependencies {
  isReady: () => boolean
  getSession: () => Session | null
  resolvePortalBaseUrl: () => string
  createWindow: (options: BrowserWindowConstructorOptions) => BrowserWindow
  rememberLog: (message: string) => void
  now?: () => number
}

export const CHALLENGE_PARTITION = 'persist:hermes-challenge'
export const CHALLENGE_PATH = '/challenge'

// While hidden nobody can see it hang: BotID plus a CPU-fallback proof of work
// fits well inside this. Once revealed, the human has the ticket's own life.
const HIDDEN_DEADLINE_MS = 90_000
const DEFAULT_TICKET_LIFE_MS = 10 * 60_000
// Long enough to read "You're all set" when the window was revealed.
const REVEALED_DONE_LINGER_MS = 1_500
const REVEALED_FAILED_LINGER_MS = 20_000
// `expiresIn` is network input that becomes a timer: a delay past 2^31 ms
// overflows setTimeout and fires at once.
const MIN_TICKET_LIFE_MS = 30_000
const MAX_TICKET_LIFE_MS = 15 * 60_000
// Chromium's ERR_ABORTED: the load was superseded (the page navigated while
// loading), not failed.
const ERR_ABORTED = -3

// How long a settled challenge URL is remembered. A window the user closed,
// or a page that already ruled, must not come back for the same ticket just
// because the backend is still announcing it (tickets live ten minutes).
const SETTLED_MEMORY_MS = 10 * 60_000
/** Outcomes after which the same URL is worth another window. */
const RETRYABLE_OUTCOMES: readonly ChallengeOutcome[] = ['timeout', 'error']

const PHASES: readonly ChallengePhase[] = ['working', 'interactive', 'done', 'failed']

export function challengeUrlAllowed(url: string, portalBaseUrl: string): boolean {
  try {
    const target = new URL(url)
    const portal = new URL(portalBaseUrl)

    return (
      (target.protocol === 'https:' || target.protocol === 'http:') &&
      target.origin === portal.origin &&
      (target.pathname === CHALLENGE_PATH || target.pathname.startsWith(`${CHALLENGE_PATH}/`))
    )
  } catch {
    return false
  }
}

export function challengePhase(url: string): ChallengePhase | null {
  try {
    const fragment = new URL(url).hash.replace(/^#/, '')

    return PHASES.find(phase => phase === fragment) ?? null
  } catch {
    return null
  }
}

export function challengeWindowOptions(session: Session): BrowserWindowConstructorOptions {
  return {
    width: 520,
    height: 720,
    show: false,
    title: 'Hermes — quick check',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      session,
      // An unshown window is not composited; without this its timers are
      // clamped and the page's work can stall while nobody is watching it.
      backgroundThrottling: false
    }
  }
}

/** Remote content in its own jar: no permissions, no downloads. Idempotent per session. */
const guardedSessions = new WeakSet<Session>()

export function guardChallengeSession(session: Session): void {
  if (guardedSessions.has(session)) {
    return
  }

  guardedSessions.add(session)
  session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
  session.setPermissionCheckHandler(() => false)
  session.on('will-download', event => event.preventDefault())
}

export function createChallengeWindows({
  isReady,
  getSession,
  resolvePortalBaseUrl,
  createWindow,
  rememberLog,
  now
}: ChallengeWindowDependencies) {
  // One window per challenge URL: the backend's event and the renderer's
  // status read can both ask for the same one.
  const running = new Map<string, Promise<ChallengeOutcome>>()
  const settledAt = new Map<string, { outcome: ChallengeOutcome; at: number }>()
  const clock = now ?? Date.now

  function drive(request: ChallengeRequest, session: Session): Promise<ChallengeOutcome> {
    const portalOrigin = new URL(resolvePortalBaseUrl()).origin

    return new Promise<ChallengeOutcome>(resolve => {
      let settled = false
      let revealed = false
      let win: BrowserWindow | null = null
      let deadline: ReturnType<typeof setTimeout> | null = null
      let linger: ReturnType<typeof setTimeout> | null = null

      const finish = (outcome: ChallengeOutcome) => {
        if (settled) {
          return
        }

        settled = true

        for (const timer of [deadline, linger]) {
          if (timer) {
            clearTimeout(timer)
          }
        }

        rememberLog(`[free-tier] challenge window ${outcome}${revealed ? ' (revealed)' : ''}`)
        // Settle first: a destroy() that throws must not leave the caller hanging.
        resolve(outcome)

        if (win && !win.isDestroyed()) {
          win.destroy()
        }
      }

      const armDeadline = (ms: number) => {
        if (deadline) {
          clearTimeout(deadline)
        }

        deadline = setTimeout(() => finish('timeout'), ms)
      }

      const finishAfter = (outcome: ChallengeOutcome, ms: number) => {
        if (!settled && !linger) {
          linger = setTimeout(() => finish(outcome), ms)
        }
      }

      const onPortal = (url: string) => {
        try {
          return new URL(url).origin === portalOrigin
        } catch {
          return false
        }
      }

      const onPhase = (url: string) => {
        // Only the portal speaks for the challenge.
        const phase = onPortal(url) ? challengePhase(url) : null

        if (settled || phase === null || phase === 'working') {
          return
        }

        if (phase === 'interactive') {
          // An optional challenge runs where nobody is looking; it has nothing
          // to ask a person for.
          if (!request.required) {
            finish('failed')

            return
          }

          if (!revealed && win && !win.isDestroyed()) {
            revealed = true
            armDeadline(ticketLifeMs(request.expiresIn))
            win.show()
            win.focus()
          }

          return
        }

        if (revealed) {
          finishAfter(phase, phase === 'done' ? REVEALED_DONE_LINGER_MS : REVEALED_FAILED_LINGER_MS)
        } else {
          finish(phase)
        }
      }

      try {
        guardChallengeSession(session)
        win = createWindow(challengeWindowOptions(session))
      } catch (error) {
        rememberLog(`[free-tier] challenge window could not be created: ${String(error)}`)
        finish('error')

        return
      }

      const contents = win.webContents

      contents.setAudioMuted(true)
      contents.setWindowOpenHandler(
        createWindowOpenHandler(origin => rememberLog(`[free-tier] challenge popup denied: ${origin}`))
      )

      // The page may move within the portal (a fragment, a reload); it may not
      // take this window anywhere else — by script (`will-navigate`) or by a
      // server redirect (`will-redirect`, which `will-navigate` never sees).
      const stayOnPortal = (event: { preventDefault: () => void }, url: string) => {
        if (!onPortal(url)) {
          event.preventDefault()
        }
      }

      contents.on('will-navigate', stayOnPortal)
      contents.on('will-redirect', stayOnPortal)
      contents.on('did-navigate-in-page', (_event, url, isMainFrame) => {
        if (isMainFrame !== false) {
          onPhase(url)
        }
      })
      contents.on('did-navigate', (_event, url, httpResponseCode) => {
        // Belt and braces: a main-frame commit off the portal ends the run; an
        // HTTP error page is not going to signal anything, but is worth a retry.
        if (!onPortal(url)) {
          finish('failed')
        } else if (typeof httpResponseCode === 'number' && httpResponseCode >= 400) {
          finish('error')
        } else {
          onPhase(url)
        }
      })
      contents.on('render-process-gone', () => finish('error'))
      win.on('closed', () => finish('closed'))

      armDeadline(HIDDEN_DEADLINE_MS)
      win.loadURL(request.url).catch((error: { errno?: number }) => {
        if (error?.errno !== ERR_ABORTED) {
          finish('error')
        }
      })
    })
  }

  function run(request: ChallengeRequest): Promise<ChallengeOutcome> {
    const session = isReady() ? getSession() : null

    if (!session || !challengeUrlAllowed(request.url, resolvePortalBaseUrl())) {
      rememberLog('[free-tier] challenge window refused (not ready, or URL is not a portal challenge)')

      return Promise.resolve('refused')
    }

    // A required ask for a ticket that ran (or is running) as optional, never revealed, must not
    // join that run or inherit its verdict: both keys carry whether the window may be shown.
    const runningKey = `${request.url}:${request.required}`
    const existing = running.get(runningKey)

    if (existing) {
      return existing
    }

    const attemptKey = `${runningKey}:${request.attempt ?? 0}`
    const before = settledAt.get(attemptKey)

    if (before && clock() - before.at < SETTLED_MEMORY_MS && !RETRYABLE_OUTCOMES.includes(before.outcome)) {
      return Promise.resolve(before.outcome)
    }

    const outcome = drive(request, session)
      .then(result => {
        settledAt.set(attemptKey, { outcome: result, at: clock() })

        for (const [key, entry] of settledAt) {
          if (clock() - entry.at >= SETTLED_MEMORY_MS) {
            settledAt.delete(key)
          }
        }

        return result
      })
      .finally(() => running.delete(runningKey))

    running.set(runningKey, outcome)

    return outcome
  }

  return { run }
}

function ticketLifeMs(expiresInSeconds: number | undefined): number {
  const asked = (expiresInSeconds ?? 0) * 1000 || DEFAULT_TICKET_LIFE_MS

  return Math.min(MAX_TICKET_LIFE_MS, Math.max(MIN_TICKET_LIFE_MS, asked))
}

/** IPC payloads are untrusted: accept only the documented shape. */
export function parseChallengeRequest(value: unknown): ChallengeRequest | null {
  if (typeof value !== 'object' || value === null) {
    return null
  }

  const { url, required, expiresIn, attempt } = value as Record<string, unknown>

  if (typeof url !== 'string' || url.length === 0 || url.length > 2048) {
    return null
  }

  return {
    url,
    required: required !== false,
    expiresIn: typeof expiresIn === 'number' && Number.isFinite(expiresIn) && expiresIn > 0 ? expiresIn : undefined,
    ...(typeof attempt === 'number' && Number.isSafeInteger(attempt) && attempt >= 0 ? { attempt } : {})
  }
}
