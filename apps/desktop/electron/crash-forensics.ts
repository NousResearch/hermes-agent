/**
 * Last-chance forensics for the Electron main process.
 *
 * Electron installs its own `uncaughtException` listener and only warns on
 * unhandled rejections, so the app usually survives — but the reason lands on
 * stderr alone, which is discarded entirely when the app is launched from
 * Finder or the Start menu. Without a record in desktop.log, a main-process
 * fault is invisible in a `hermes debug share` bundle and the user is left
 * describing symptoms instead of showing a stack.
 */

export interface CrashForensicsTarget {
  on: (event: 'uncaughtException' | 'unhandledRejection', listener: (value: unknown) => void) => unknown
}

export interface CrashForensicsOptions {
  flush: () => void
  log: (message: string) => void
  target?: CrashForensicsTarget
}

export interface BrokenPipeStream {
  on: (event: 'error', listener: (value: unknown) => void) => unknown
}

export interface BrokenPipeTarget {
  stdout?: BrokenPipeStream
  stderr?: BrokenPipeStream
}

/** Render a thrown value for the log, preferring a stack over a bare message. */
export function describeCrashReason(reason: unknown): string {
  if (reason instanceof Error) {
    return reason.stack || reason.message || reason.name || 'Error'
  }

  if (typeof reason === 'string') {
    return reason
  }

  try {
    return JSON.stringify(reason) ?? String(reason)
  } catch {
    return String(reason)
  }
}

/**
 * A desktop launched from Explorer normally has no writable parent console.
 * If a launcher or terminal closes while Electron is still flushing a
 * console.log, Node emits EPIPE on stdout/stderr. Without an error listener,
 * that stream error becomes a fatal main-process exception and Electron shows
 * its generic JavaScript error dialog.
 */
export function installBrokenPipeGuards(target: BrokenPipeTarget = process): void {
  for (const stream of [target.stdout, target.stderr]) {
    stream?.on('error', error => {
      if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'EPIPE') {
        return
      }

      // Keep non-EPIPE stream errors from becoming a second uncaught error.
      // The normal main-process forensics handler remains responsible for
      // application failures; this guard only protects the output channels.
    })
  }
}

/**
 * Record main-process faults to desktop.log and flush synchronously, since a
 * fault that does prove fatal leaves no chance for the batched async flush.
 */
export function installCrashForensics({ flush, log, target = process }: CrashForensicsOptions): void {
  const record = (label: string) => (reason: unknown) => {
    log(`[main] ${label}: ${describeCrashReason(reason)}`)
    flush()
  }

  target.on('uncaughtException', record('Uncaught exception'))
  target.on('unhandledRejection', record('Unhandled rejection'))
}
