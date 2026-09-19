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

/** Shape of the `Event<WebContentsConsoleMessageEventParams>` Electron
 *  delivers as the first argument of the `webContents` `console-message`
 *  event. Electron still appends the legacy positional arguments
 *  `(level, message, line, sourceId)` after the event object, but any
 *  listener that declares them triggers Electron's runtime deprecation
 *  warning, so handlers must read from the event object only. */
interface RendererConsoleMessage {
  /** Severity. Chromium's `ConsoleMessageLevel` arrives stringified:
   *  verbose → `debug`, then `info`, `warning`, `error`. */
  level: 'debug' | 'info' | 'warning' | 'error'
  message: string
  lineNumber: number
  sourceId: string
}

interface WebContentsLike {
  on(event: 'console-message', listener: (event: RendererConsoleMessage) => void): unknown
}

interface WindowLike {
  webContents: WebContentsLike
}

/** Format a renderer console-message event into one desktop.log line, or
 *  null for non-error levels. */
export function formatRendererConsoleLine(label: string, event: RendererConsoleMessage): string | null {
  if (event.level !== 'error') {
    return null
  }

  return `[renderer console:${label}] ${event.message} (${event.sourceId}:${event.lineNumber})`
}

/** Attach the error-level console hook to a renderer window. `log` is the
 *  desktop.log sink (rememberLog in main.ts). */
export function attachRendererConsoleCapture(win: WindowLike, label: string, log: (line: string) => void): void {
  win.webContents.on('console-message', (event) => {
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
