/**
 * Recovery for a window whose INITIAL renderer load failed.
 *
 * Why this exists: `loadWindowUrl` caught the `loadURL` rejection, wrote one
 * desktop.log line and stopped. On every window whose renderer-lifecycle
 * policy does not opt into `reloadOnFailedLoad` (secondary session windows,
 * browser popouts, instance windows) a failed load therefore ended in a window
 * that is blank forever, with nothing in the UI and the only explanation in a
 * log file. The primary window is the exception: it owns a bounded reload
 * policy and surfaces the error page when that budget is exhausted.
 *
 * This module supplies the missing end of the ladder for the other windows —
 * bounded reloading is the lifecycle's job, so the rule here is simply that a
 * failed load must END somewhere the user can act on:
 *
 *   destroyed window            -> load nothing (never resurrect a dead window)
 *   ERR_ABORTED (-3)            -> load nothing (a superseded navigation is not
 *                                  a failure; the lifecycle refuses -3 for the
 *                                  same reason)
 *   lifecycle owns recovery     -> load nothing (surfacing a page would race the
 *                                  bounded reload and could cover a UI that healed)
 *   anything else               -> load the visible error page, with the failed
 *                                  URL as the Reload target
 *
 * Pure decision + injected deps so it is unit-testable without booting Electron
 * (same shape as renderer-load-error-page.ts / window-renderer-lifecycle.ts).
 */

export interface FailedWindowLoadDetails {
  /** Window label used in log lines: 'Renderer' | 'Session window' | 'Browser window' | … */
  label: string
  /** The URL that failed to load; also the Reload target. */
  url: string
  /** Chromium error code (-2 = ERR_FAILED, -3 = ERR_ABORTED, 'ERR_FILE_NOT_FOUND', …). */
  errorCode?: number | string | undefined
  /** Human description from Electron's did-fail-load, when one was available. */
  errorDescription?: string
  /** True when the window's renderer-lifecycle policy owns failed-load recovery. */
  recoveryOwnedByLifecycle?: boolean
  /** True when the window was destroyed (or is being torn down) already. */
  isDestroyed?: boolean
}

export type FailedLoadRecoveryReason =
  | 'window-destroyed'
  | 'superseded-navigation'
  | 'lifecycle-owns-recovery'
  | 'no-recovery-owner'

export interface FailedLoadRecoveryDecision {
  showErrorPage: boolean
  reason: FailedLoadRecoveryReason
}

/** Minimal structural surface of BrowserWindow used here: `isDestroyed` for the
 *  decision, `loadURL` because the injected error-page loader loads into it. */
export interface LoadRecoveryWindowLike {
  loadURL: (url: string) => Promise<unknown>
  isDestroyed?: () => boolean
}

export interface FailedLoadRecoveryDeps {
  log: (message: string) => void
  /** Loads the visible error page. May reject; the caller must not. */
  showErrorPage: (
    win: LoadRecoveryWindowLike,
    details: {
      errorCode?: number | string | undefined
      errorDescription?: string
      url: string
      reloadUrl: string
    }
  ) => Promise<unknown> | unknown
}

/** -3 = ERR_ABORTED: the load was superseded by another navigation. Never a failure. */
const SUPERSEDED_NAVIGATION_CODES = new Set<number | string>([-3, '-3', 'ERR_ABORTED'])

/**
 * `ERR_ABORTED (-3) loading 'file:///…'` — the message Electron actually rejects
 * with (the app logs it verbatim: `Renderer failed to load: Error: ERR_ABORTED (-3) …`).
 * The optional `Error: ` prefix tolerates an already-formatted rejection.
 */
const LOAD_FAILURE_MESSAGE_CODE = /^(?:Error:\s*)?([A-Z][A-Z0-9_]*)\s+\((-?\d+)\)/

/**
 * The Chromium code behind a rejected `loadURL`.
 *
 * Electron's typings declare the rejection as a bare `Promise<void>`, so the
 * shape is read defensively: a structured `code`/`errno` first (when the
 * rejection carries one), then the message. Naming the code matters because an
 * ABORTED navigation is not a failure to put in front of the user.
 */
export function loadFailureErrorCode(error: unknown): number | string | undefined {
  if (error === null || typeof error !== 'object') {
    return undefined
  }

  const record = error as { code?: unknown; errno?: unknown; message?: unknown }

  if (typeof record.code === 'number' || typeof record.code === 'string') {
    return record.code
  }

  const message = typeof record.message === 'string' ? record.message.trim() : ''
  const named = LOAD_FAILURE_MESSAGE_CODE.exec(message)

  if (named) {
    return named[1]
  }

  return typeof record.errno === 'number' ? record.errno : undefined
}

export function decideFailedLoadRecovery(details: FailedWindowLoadDetails): FailedLoadRecoveryDecision {
  if (details.isDestroyed === true) {
    return { showErrorPage: false, reason: 'window-destroyed' }
  }

  const errorCode = details.errorCode

  if (errorCode !== undefined && errorCode !== null && SUPERSEDED_NAVIGATION_CODES.has(errorCode)) {
    return { showErrorPage: false, reason: 'superseded-navigation' }
  }

  if (details.recoveryOwnedByLifecycle === true) {
    return { showErrorPage: false, reason: 'lifecycle-owns-recovery' }
  }

  return { showErrorPage: true, reason: 'no-recovery-owner' }
}

/**
 * A window that cannot answer `isDestroyed()` — or throws while answering —
 * must be treated as gone: loading a page into it would either no-op or throw
 * from Electron, and the caller has no way to know it worked.
 */
function isWindowGone(win: LoadRecoveryWindowLike | undefined, details: FailedWindowLoadDetails): boolean {
  if (details.isDestroyed === true) {
    return true
  }

  if (typeof win?.isDestroyed === 'function') {
    try {
      return Boolean(win.isDestroyed())
    } catch {
      return true
    }
  }

  return false
}

/**
 * End a failed initial load in a surface the user can act on. Always resolves:
 * the failure being handled is already a bad state, and turning the recovery
 * attempt into an unhandled rejection would lose the reason it happened.
 */
export async function handleFailedWindowLoad(
  win: LoadRecoveryWindowLike,
  details: FailedWindowLoadDetails,
  deps: FailedLoadRecoveryDeps
): Promise<FailedLoadRecoveryDecision> {
  const decision = decideFailedLoadRecovery({ ...details, isDestroyed: isWindowGone(win, details) })

  if (!decision.showErrorPage) {
    return decision
  }

  const code = details.errorCode === undefined ? '' : ` (${String(details.errorCode)})`

  deps.log(`[renderer-load] ${details.label}: loading visible recovery page after a failed load${code}`)

  try {
    await deps.showErrorPage(win, {
      errorCode: details.errorCode,
      errorDescription: details.errorDescription,
      url: details.url,
      reloadUrl: details.url
    })
  } catch (error) {
    deps.log(
      `[renderer-load] ${details.label}: recovery page failed to load: ` +
        `${error instanceof Error ? error.message : String(error)}`
    )
  }

  return decision
}
