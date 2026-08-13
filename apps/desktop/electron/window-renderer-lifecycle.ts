// Per-window renderer lifecycle diagnostics + crash recovery (#81290).
//
// The desktop app renders one Chromium renderer per window (primary, secondary
// session windows, full instance windows, and the small helper overlays), but
// renderer-lifecycle listeners used to be attached ONLY to the primary window:
// a dead peer renderer produced no log line and no recovery, leaving the user
// with a permanently black window and nothing in desktop.log. This module
// attaches the same lifecycle wiring to every window, keyed by a `kind` label,
// with an injected reload policy so the pure decision logic stays Electron-free
// and unit-testable (mirroring windows-sandbox-fallback.ts / session-windows.ts).
//
// Policy (matches the primary window's previous behavior, generalized):
// - `render-process-gone` with reason `crashed`/`oom` → bounded reload (rolling
//   crash-loop guard shared across ALL windows — one budget per process).
// - `render-process-gone` with any other reason (`killed`, `launch-failed`,
//   `clean-exit`, unknown) → log only. `killed` after an expected close/destroy
//   is normal teardown, and blindly reloading it would loop windows back up
//   after the user closed them.
// - `unresponsive` → log only (no reload; Chromium usually follows with
//   render-process-gone, and forcing a reload while the main thread is wedged
//   can make things worse).
// - `did-fail-load` on the MAIN frame → log only (no blind reload: a repeatable
//   startup failure would boot-loop; the backend startup path already surfaces
//   the actionable error).
//
// Console-message capture is deliberately NOT here: renderer-log.ts owns it
// (per-window labels, boundary-report formatting). Keeping one owner avoids
// double-logging on windows that have both, and keeps third-party pages
// (OAuth/portal windows, which install this helper for process events) from
// spilling their console output — potentially tokens/PII — into desktop.log.

export interface RendererLifecycleDetails {
  reason?: string
  exitCode?: number | string | undefined
  isDestroyed?: boolean
}

export interface RendererLifecycleEvent {
  kind: string
  event: 'render-process-gone' | 'unresponsive' | 'did-fail-load'
  reason?: string
  exitCode?: number | string | undefined
  isDestroyed?: boolean
  /** did-fail-load: only main-frame failures are meaningful (issue point 4). */
  isMainFrame?: boolean
  /** did-fail-load: the Chromium error code (e.g. -3 = ERR_ABORTED). */
  errorCode?: number | string | undefined
  /** did-fail-load: the URL that failed. */
  url?: string
}

export interface ReloadPolicyDecision {
  reload: boolean
  /** Why reload was refused, when it was. */
  suppressedReason?: 'crash-loop' | 'expected-teardown' | 'unrecoverable-reason'
}

export interface WindowRendererLifecycleOptions {
  /** Stable label used in log lines: 'main' | 'secondary' | 'instance' |
   *  'overlay' | 'quick' | 'wake'. */
  kind: string
  callbacks: {
    log: (message: string) => void
    /** Omitted → log-only mode (helper windows never reload). */
    reload?: () => void
    /** Called when the shared crash-loop budget suppresses a reload — the
     *  primary window uses it for the #38216 Windows sandbox relaunch check. */
    onCrashLoopSuppressed?: (details?: RendererLifecycleDetails) => void
  }
  /** Rolling crash-loop window, ms. Defaults to 60_000 (RENDERER_RELOAD_WINDOW_MS). */
  reloadWindowMs?: number
  /** Max reloads per rolling window. Defaults to 3 (RENDERER_RELOAD_MAX). */
  reloadMax?: number
  /** Shared per-process reload budget. Omitted → per-window budget (tests). */
  recentReloadTimesRef?: { current: number[] }
  now?: () => number
}

/** Minimal structural surface of BrowserWindow / webContents used here. */
export interface LifecycleWindowLike {
  isDestroyed: () => boolean
  webContents: {
    on: (event: string, listener: (...args: any[]) => void) => unknown
    reload?: () => void
    removeListener?: (event: string, listener: (...args: any[]) => void) => unknown
  }
}

const DEFAULT_RELOAD_WINDOW_MS = 60_000
const DEFAULT_RELOAD_MAX = 3

const RECOVERABLE_REASONS = new Set(['crashed', 'oom'])

function safeNow(now: (() => number) | undefined): number {
  return typeof now === 'function' ? now() : Date.now()
}

function isWithin(timestamp: number, now: number, windowMs: number): boolean {
  return now - timestamp < windowMs
}

/** Drop reload timestamps outside the rolling window. Mutates + returns. */
export function pruneReloadTimes(times: number[], now: number, windowMs: number): number[] {
  return times.filter(timestamp => isWithin(timestamp, now, windowMs))
}

/** Record a reload attempt timestamp. Mutates + returns. */
export function pushReloadTime(times: number[], now: number): number[] {
  times.push(now)

  return times
}

/**
 * Decide whether a render-process-gone event should reload its window.
 *
 * Reload only for `crashed`/`oom` on a live window, bounded by the shared
 * rolling crash-loop budget. Anything else — expected teardown (`killed` after
 * close/destroy), unrecoverable reasons, unknown reasons — is log-only, exactly
 * like the primary window's previous behavior but now per window kind.
 */
export function shouldReloadAfterRendererGone(details: {
  reason?: string
  isDestroyed?: boolean
  recentReloadTimes: number[]
  reloadWindowMs?: number
  reloadMax?: number
  now?: () => number
}): ReloadPolicyDecision {
  if (details.isDestroyed) {
    return { reload: false, suppressedReason: 'expected-teardown' }
  }

  const reason = String(details.reason || '')

  if (!RECOVERABLE_REASONS.has(reason)) {
    return { reload: false, suppressedReason: 'unrecoverable-reason' }
  }

  const windowMs = details.reloadWindowMs ?? DEFAULT_RELOAD_WINDOW_MS
  const max = details.reloadMax ?? DEFAULT_RELOAD_MAX
  const now = safeNow(details.now)
  const recent = pruneReloadTimes(details.recentReloadTimes, now, windowMs)

  if (recent.length >= max) {
    return { reload: false, suppressedReason: 'crash-loop' }
  }

  return { reload: true }
}

/**
 * One log line per renderer lifecycle event, e.g.
 *   [renderer:secondary] render-process-gone reason=crashed exitCode=3
 * Sanitizes unknown fields and annotates expected teardown so a support bundle
 * reads as a story, not a pile of question marks.
 */
export function describeRendererLifecycleEvent(event: RendererLifecycleEvent): string {
  const kind = String(event.kind || '?')

  if (event.event === 'unresponsive') {
    return `[renderer:${kind}] webContents became unresponsive`
  }

  if (event.event === 'did-fail-load') {
    const code = event.errorCode === undefined ? '?' : String(event.errorCode)
    const url = String(event.url || '?')

    return `[renderer:${kind}] did-fail-load code=${code} url=${url}`
  }

  const reason = String(event.reason || '?')
  const exitCode = event.exitCode === undefined ? '?' : String(event.exitCode)
  const teardown = event.isDestroyed && reason === 'killed' ? ' (expected teardown)' : ''

  return `[renderer:${kind}] render-process-gone reason=${reason} exitCode=${exitCode}${teardown}`
}

/**
 * Attach renderer lifecycle listeners to a window. Returns a dispose() that
 * removes every listener (window recreation must not stack handlers).
 *
 * `reload` is never invoked synchronously inside the event handler: Electron
 * warns about re-entrant webContents calls, and the primary window's previous
 * implementation deferred via setImmediate for the same reason.
 */
export function installWindowRendererLifecycle(
  win: LifecycleWindowLike,
  options: WindowRendererLifecycleOptions
): () => void {
  const kind = options.kind
  const { log, reload, onCrashLoopSuppressed } = options.callbacks
  const reloadWindowMs = options.reloadWindowMs ?? DEFAULT_RELOAD_WINDOW_MS
  const reloadMax = options.reloadMax ?? DEFAULT_RELOAD_MAX
  const now = options.now
  const budgetRef = options.recentReloadTimesRef ?? { current: [] }
  const contents = win.webContents

  const onRendererGone = (_event: unknown, details?: RendererLifecycleDetails) => {
    const destroyed = win.isDestroyed()

    log(describeRendererLifecycleEvent({ kind, event: 'render-process-gone', ...details, isDestroyed: destroyed }))

    const nowMs = safeNow(now)
    const recent = pruneReloadTimes(budgetRef.current, nowMs, reloadWindowMs)

    budgetRef.current.length = 0
    budgetRef.current.push(...recent)

    const decision = shouldReloadAfterRendererGone({
      reason: details?.reason,
      isDestroyed: destroyed,
      recentReloadTimes: budgetRef.current,
      reloadWindowMs,
      reloadMax,
      now: () => nowMs
    })

    if (!decision.reload) {
      if (decision.suppressedReason === 'crash-loop') {
        log(
          `[renderer:${kind}] suppressing reload: ${budgetRef.current.length} crashes within ${reloadWindowMs}ms (likely a crash loop)`
        )
        onCrashLoopSuppressed?.(details)
      }

      return
    }

    if (typeof reload !== 'function') {
      return
    }

    pushReloadTime(budgetRef.current, nowMs)

    // Deferred: never reload from inside the event handler (see above).
    setImmediate(() => {
      if (win.isDestroyed()) {
        return
      }

      try {
        reload()
      } catch (error) {
        log(`[renderer:${kind}] reload after crash failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    })
  }

  const onUnresponsive = () => {
    log(describeRendererLifecycleEvent({ kind, event: 'unresponsive' }))
  }

  const onDidFailLoad = (
    _event: unknown,
    errorCode: unknown,
    _errorDescription: unknown,
    validatedURL: unknown,
    isMainFrame?: unknown
  ) => {
    if (isMainFrame === true) {
      log(
        describeRendererLifecycleEvent({
          kind,
          event: 'did-fail-load',
          errorCode: typeof errorCode === 'number' ? errorCode : String(errorCode ?? ''),
          url: String(validatedURL ?? '')
        })
      )
    }
  }

  contents.on('render-process-gone', onRendererGone)
  contents.on('unresponsive', onUnresponsive)
  contents.on('did-fail-load', onDidFailLoad)

  let disposed = false

  return () => {
    if (disposed) {
      return
    }

    disposed = true

    if (typeof contents.removeListener === 'function') {
      contents.removeListener('render-process-gone', onRendererGone)
      contents.removeListener('unresponsive', onUnresponsive)
      contents.removeListener('did-fail-load', onDidFailLoad)
    }
  }
}
