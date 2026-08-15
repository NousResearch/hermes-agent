export type ParentWatchdogEnv = {
  HERMES_PARENT_PID: string
  HERMES_PARENT_START_MARKER?: string
  HERMES_PARENT_NONCE?: string
}

export interface ParentStartMarkerResolverOptions {
  load: () => Promise<string>
  onError?: (error: unknown) => void
}

/**
 * Build the cross-runtime marker for Electron's own process without spawning
 * an OS helper. Electron reports milliseconds since the Unix epoch; the Python
 * watchdog converts its exact Windows FILETIME to the same representation.
 */
export function electronProcessStartMarker(
  pid: number,
  ownPid: number,
  creationTime: unknown
): string | null {
  if (pid !== ownPid || typeof creationTime !== 'number' || !Number.isFinite(creationTime)) {
    return null
  }

  const milliseconds = Math.trunc(creationTime)

  if (!Number.isSafeInteger(milliseconds) || milliseconds <= 0) {
    return null
  }

  return `winms:${milliseconds}`
}

/** Cache a successful parent marker while allowing a transient failure to retry. */
export function createParentStartMarkerResolver(options: ParentStartMarkerResolverOptions) {
  let cached: Promise<string> | null = null

  return async (): Promise<string | null> => {
    const attempt = cached ?? Promise.resolve().then(options.load)
    cached = attempt

    try {
      return await attempt
    } catch (error) {
      let shouldReport = false

      if (cached === attempt) {
        cached = null
        shouldReport = true
      }

      if (shouldReport) {
        try {
          options.onError?.(error)
        } catch {
          // Diagnostics must not turn an optional identity probe into a boot gate.
        }
      }

      return null
    }
  }
}

/**
 * Keep the watchdog's marker and nonce atomic. A failed marker probe degrades
 * to the legacy PID-only watchdog instead of preventing the backend spawn.
 */
export function parentWatchdogEnv(pid: number, startMarker: string | null, nonce: string): ParentWatchdogEnv {
  if (!Number.isInteger(pid) || pid <= 0) {
    throw new Error('Parent watchdog requires a positive process ID.')
  }

  const env: ParentWatchdogEnv = { HERMES_PARENT_PID: String(pid) }

  if (startMarker === null) {
    return env
  }

  if (!startMarker || !nonce) {
    throw new Error('Parent watchdog marker and nonce must be non-empty.')
  }

  env.HERMES_PARENT_START_MARKER = startMarker
  env.HERMES_PARENT_NONCE = nonce

  return env
}
