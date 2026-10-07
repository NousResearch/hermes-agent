import type { ErrorInfo } from 'react'

/**
 * React 19 recovers from an error thrown during a concurrent render by re-rendering the
 * whole root synchronously — the "Minified React error #520" line. Recovery involves no
 * error-boundary catch, and the console line carries only React's own message, so the
 * error that actually threw survives solely as `error.cause`: nothing reads it, and every
 * occurrence is unattributable in desktop.log.
 *
 * These helpers turn that recovered error into the report the existing renderer-crash
 * channel (`window.hermesDesktop.reportRendererError`) persists, naming the cause.
 */

export interface RecoverableErrorReport {
  label: string
  boundary: string
  message: string
  componentStack: string
}

/** Distinct from an `error-boundary:` catch: this is React's concurrent-render recovery. */
export const RECOVERABLE_BOUNDARY = 'recoverable-concurrent-render'

const MESSAGE_MAX = 2000
const STACK_MAX = 4000
const MAX_CAUSES = 3
const CAUSE_FRAME_MAX = 160

const clamp = (value: unknown, max: number): string => String(value ?? '').slice(0, max)

/** First frame of a cause's stack — enough to tell which module threw, cheap enough to log. */
const firstFrame = (error: unknown): string => {
  if (!(error instanceof Error) || typeof error.stack !== 'string') {
    return ''
  }

  const frame = error.stack
    .split('\n')
    .slice(1)
    .map(line => line.trim())
    .find(line => line.startsWith('at '))

  return frame ? ` (${clamp(frame, CAUSE_FRAME_MAX)})` : ''
}

const describe = (value: unknown): string => {
  if (value instanceof Error) {
    return `${value.name || 'Error'}: ${value.message || '(no message)'}`
  }

  return clamp(value === undefined ? '(undefined thrown)' : String(value), MESSAGE_MAX)
}

/**
 * "<thrown message> ← caused by: <cause message> (at <frame>)".
 * React's wrapper message comes first because that is what the console shows; each cause
 * follows because that is what threw. The chain is bounded — a cyclic `cause` cannot hang it.
 */
export function describeRecoverableError(error: unknown): string {
  const parts = [describe(error)]
  let cause: unknown = error instanceof Error ? error.cause : undefined

  for (let depth = 0; depth < MAX_CAUSES && cause !== undefined && cause !== null; depth += 1) {
    parts.push(`← caused by: ${describe(cause)}${firstFrame(cause)}`)
    cause = cause instanceof Error ? cause.cause : undefined
  }

  return clamp(parts.join(' '), MESSAGE_MAX)
}

export function buildRecoverableErrorReport(
  error: unknown,
  errorInfo: ErrorInfo | undefined,
  label: string
): RecoverableErrorReport {
  return {
    label: label || 'main',
    boundary: RECOVERABLE_BOUNDARY,
    message: describeRecoverableError(error),
    componentStack: clamp(errorInfo?.componentStack ?? '', STACK_MAX).trim()
  }
}
