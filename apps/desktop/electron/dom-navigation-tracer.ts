// DOM navigation tracer for the five-second flash/reset symptom (no issue #).
//
// The reported symptom is a visible ~5s flash/reset of the desktop UI while
// every existing diagnostic is quiet: desktop.log carries zero
// render-process-gone / did-fail-load / reload lines across days (verified
// 2026-08-31 over desktop.log + desktop.log.1), the renderer console hook
// logs only error-level messages, and the backend logs show nothing while the
// flash persists. A stable renderer PID does NOT rule out webContents.reload()
// — a reload re-navigates the SAME webContents and renderer process — and no
// instrument today would ever log one.
//
// This tracer closes that gap. It attaches ONE cheap, navigation-scoped
// listener set to a webContents and writes a single desktop.log line per
// navigation lifecycle event. These events are silent in steady state (they
// only fire on actual navigation/load), so the log cost is zero while nothing
// happens and complete (with exact timestamps) when the symptom fires:
//
//   [domnav:main] did-start-navigation url=... main=true inPlace=false
//   [domnav:main] did-navigate url=...
//   [domnav:main] did-finish-load url=...
//   [domnav:main] did-fail-load code=-3 desc=... url=... main=true
//   [domnav:main] render-process-gone reason=crashed exitCode=1
//
// What each outcome will mean for the diagnosis:
// - a `did-start-navigation` burst with ~5s cadence and a stable renderer PID
//   proves a programmatic reload loop (webContents.reload / location.reload)
//   — next step is finding the caller, no more guessing;
// - navigation events ABSENT while the flash is still seen rules the whole
//   document-reload class out and points at compositor-level repaint
//   (GPU/theme/translucency) or a renderer-internal remount;
// - a render-process-gone line here but nowhere else means the lifecycle
//   wiring missed this window (the tracer must be attached everywhere this
//   helper is, so absence is meaningful).
//
// Deliberately duplicate-free of policy: the lifecycle helper owns reload
// decisions (window-renderer-lifecycle.ts); this module only OBSERVES, so its
// did-fail-load lines intentionally coexist with the lifecycle's own log line.

export interface DomNavigationTracerWindowLike {
  isDestroyed: () => boolean
  webContents: {
    on: (event: string, listener: (...args: unknown[]) => void) => unknown
    removeListener?: (event: string, listener: (...args: unknown[]) => void) => unknown
  }
}

interface NavigationDetails {
  url?: unknown
  isInPlace?: unknown
  isMainFrame?: unknown
}

interface FailedLoadDetails {
  errorCode?: unknown
  errorDescription?: unknown
  validatedURL?: unknown
  isMainFrame?: unknown
}

interface RenderGoneDetails {
  reason?: unknown
  exitCode?: unknown
}

/** One line per event, e.g.
 *  `[domnav:main] did-fail-load code=-3 desc=ERR_ABORTED url=... main=true`.
 *  Unknown values render as `?` so a support bundle reads as a story, not
 *  `undefined undefined undefined`. Exported for tests. */
export function formatDomNavigationLine(label: string, event: string, parts: string[]): string {
  const body = parts.length > 0 ? ` ${parts.join(' ')}` : ''

  return `[domnav:${String(label || '?')}] ${event}${body}`
}

function field(name: string, value: unknown): string {
  return `${name}=${value === undefined || value === null || value === '' ? '?' : String(value)}`
}

export interface DomNavigationTracerOptions {
  log: (line: string) => void
}

/**
 * Attach navigation/telemetry listeners to a renderer-content window. Returns
 * a dispose() that removes every listener (window recreation must not stack
 * handlers). Attach alongside attachRendererConsoleCapture — the SAME window
 * set, so "no domnav lines" is itself evidence.
 */
export function traceWindowDomNavigation(
  win: DomNavigationTracerWindowLike,
  label: string,
  options: DomNavigationTracerOptions
): () => void {
  const { log } = options
  const contents = win.webContents
  const bound: [string, (...args: unknown[]) => void][] = []

  const listen = (event: string, handler: (...args: unknown[]) => void) => {
    contents.on(event, handler)
    bound.push([event, handler])
  }

  listen('did-start-navigation', (_event, url?: unknown, isInPlace?: unknown, isMainFrame?: unknown) => {
    if (win.isDestroyed()) {
      return
    }

    log(
      formatDomNavigationLine(label, 'did-start-navigation', [
        field('url', url),
        field('main', isMainFrame),
        field('inPlace', isInPlace)
      ])
    )
  })

  listen('did-navigate', (_event, url?: unknown) => {
    if (win.isDestroyed()) {
      return
    }

    log(formatDomNavigationLine(label, 'did-navigate', [field('url', url)]))
  })

  listen('did-navigate-in-page', (_event, url?: unknown, isMainFrame?: unknown) => {
    if (win.isDestroyed()) {
      return
    }

    log(formatDomNavigationLine(label, 'did-navigate-in-page', [field('url', url), field('main', isMainFrame)]))
  })

  listen('did-finish-load', () => {
    if (win.isDestroyed()) {
      return
    }

    log(formatDomNavigationLine(label, 'did-finish-load', []))
  })

  listen(
    'did-fail-load',
    (_event, errorCode?: unknown, errorDescription?: unknown, validatedURL?: unknown, isMainFrame?: unknown) => {
      if (win.isDestroyed()) {
        return
      }

      log(
        formatDomNavigationLine(label, 'did-fail-load', [
          field('code', errorCode),
          field('desc', errorDescription),
          field('url', validatedURL),
          field('main', isMainFrame)
        ])
      )
    }
  )

  listen('render-process-gone', (_event, details?: RenderGoneDetails) => {
    if (win.isDestroyed()) {
      return
    }

    log(
      formatDomNavigationLine(label, 'render-process-gone', [
        field('reason', details?.reason),
        field('exitCode', details?.exitCode)
      ])
    )
  })

  return () => {
    for (const [event, handler] of bound) {
      contents.removeListener?.(event, handler)
    }

    bound.length = 0
  }
}
