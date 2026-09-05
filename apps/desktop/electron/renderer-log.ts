/**
 * Renderer console/error capture shared by every renderer-content window.
 *
 * Historically only the primary window (`createWindow()`) attached a
 * `console-message` hook, so a renderer crash in ANY other window — secondary
 * session windows, instance windows, the HUD, quick entry, the pet overlay —
 * evaporated with the window: nothing in desktop.log, nothing to attach to a
 * bug report (#79428 defect B). The React error boundary logs crashes via
 * `console.error`, so windows without the hook also lost every boundary catch.
 *
 * `attachRendererConsoleCapture` is the one owner of that hook. Every window
 * that loads our renderer must go through it; the label says which window the
 * line came from so multi-window reports are diagnosable. Windows that load
 * EXTERNAL content (OAuth portals) must NOT attach — third-party pages can log
 * tokens or PII we never want on disk.
 */

type ConsoleMessageLevel = 'info' | 'warning' | 'error' | 'debug'

interface ConsoleMessageEventLike {
  level?: unknown
  message?: unknown
  sourceId?: unknown
  lineNumber?: unknown
}

interface ConsoleMessageDetails {
  level: ConsoleMessageLevel
  message: string
  sourceId: string
  lineNumber: number
}

interface WebContentsLike {
  on(event: 'console-message', listener: (event: ConsoleMessageEventLike) => void): unknown
}

interface WindowLike {
  webContents: WebContentsLike
}

let didReportConsoleMessageSignatureDrift = false

function isConsoleMessageDetails(value: unknown): value is ConsoleMessageDetails {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const event = value as ConsoleMessageEventLike
  const isKnownLevel =
    event.level === 'info' || event.level === 'warning' || event.level === 'error' || event.level === 'debug'

  return (
    isKnownLevel &&
    typeof event.message === 'string' &&
    typeof event.sourceId === 'string' &&
    typeof event.lineNumber === 'number'
  )
}

/** Format Electron's canonical console-message event object into one line, or
 *  null for non-error or malformed events. Hermes's pinned Electron 40.x line
 *  puts severity and source metadata on the event object itself; accepting one
 *  listener argument also avoids Electron's deprecated positional
 *  `(event, level, message, line, sourceId)` path. */
export function formatRendererConsoleLine(label: string, details: ConsoleMessageEventLike): string | null {
  if (!isConsoleMessageDetails(details) || details.level !== 'error') {
    return null
  }

  return `[renderer console:${label}] ${details.message} (${details.sourceId}:${String(details.lineNumber)})`
}

/** Attach the error-level console hook to a renderer window. `log` is the
 *  desktop.log sink (rememberLog in main.ts). */
export function attachRendererConsoleCapture(win: WindowLike, label: string, log: (line: string) => void): void {
  win.webContents.on('console-message', (event) => {
    if (!isConsoleMessageDetails(event)) {
      if (!didReportConsoleMessageSignatureDrift) {
        didReportConsoleMessageSignatureDrift = true
        log('[renderer console] Electron console-message signature drift detected; renderer errors may not be captured')
      }
      return
    }

    const formatted = formatRendererConsoleLine(label, event)

    if (formatted !== null) {
      log(formatted)
    }
  })
}

/** Format a renderer error-boundary report (hermes:logs:renderer-error IPC)
 *  for desktop.log. Boundary catches carry the component stack — the one piece
 *  of context a minified console line loses — so persist it alongside.
 *  Inputs are renderer-supplied: clamp so a hostile/buggy payload cannot bloat
 *  the log. */
export function formatRendererBoundaryReport(
  label: unknown,
  boundary: unknown,
  message: unknown,
  componentStack: unknown
): string {
  const clamp = (value: unknown, max: number): string => String(value ?? '').slice(0, max)

  const head = `[renderer crash:${clamp(label, 64) || 'unknown'}] [error-boundary:${clamp(boundary, 64) || 'unknown'}] ${
    clamp(message, 2000) || '(no message)'
  }`

  const stack = clamp(componentStack, 4000).trim()

  return stack ? `${head}\n${stack}` : head
}
