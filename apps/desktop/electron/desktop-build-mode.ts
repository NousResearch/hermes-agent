/**
 * Build-time/runtime policy for Desktop distributions that are only clients
 * of an already-running Hermes gateway.
 *
 * Keep this policy independent from Electron and React.  The main process uses
 * it to enforce the boundary at IPC/backend seams, while the renderer uses the
 * capability reported by the main process to hide local recovery affordances.
 */

export interface RemoteConnectionGate {
  hasWaiter: () => boolean
  resume: () => void
  wait: () => Promise<void>
}

export function assertConnectionModeAllowed(mode: unknown, remoteOnly: boolean): void {
  if (remoteOnly && mode !== 'remote' && mode !== 'cloud') {
    throw new Error('This Hermes Desktop build requires a remote Hermes connection.')
  }
}

/** Registry/source analogue of assertConnectionModeAllowed(). */
export function isConnectionKindAllowedForRemoteOnly(kind: unknown, remoteOnly: boolean): boolean {
  return !remoteOnly || kind === 'remote' || kind === 'cloud'
}

// Capability payloads cross the renderer boundary. Keep arbitrary transport
// errors (which may contain a URL query, authorization material, or a server
// response excerpt) in the main-process log only; expose a coarse actionable
// category to the renderer instead.
export function sanitizeRemoteSetupError(error: unknown): string {
  const message = String(error instanceof Error ? error.message : error || '').toLowerCase()

  if (message.includes('token') || message.includes('auth') || message.includes('401') || message.includes('403')) {
    return 'Remote gateway authentication needs attention.'
  }

  if (message.includes('url') || message.includes('http') || message.includes('address')) {
    return 'Remote gateway URL needs attention.'
  }

  return 'Could not connect to the remote Hermes gateway.'
}

/**
 * Build the only Error shape allowed to cross a standalone client's backend
 * boundary.  Keep the small structured fields used by the boot overlay, but
 * never retain the original error as `cause` (transport errors can contain
 * query-string tokens or response bodies).
 */
export function sanitizeRemoteSetupFailure(error: unknown) {
  const safe = new Error(sanitizeRemoteSetupError(error)) as Error & {
    isCloudBackendDown?: boolean
    needsOauthLogin?: boolean
    statusCode?: number
  }

  if (error && typeof error === 'object') {
    if ((error as { isCloudBackendDown?: unknown }).isCloudBackendDown === true) {
      safe.isCloudBackendDown = true
    }

    if ((error as { needsOauthLogin?: unknown }).needsOauthLogin === true) {
      safe.needsOauthLogin = true
    }

    const statusCode = (error as { statusCode?: unknown }).statusCode

    if (Number.isInteger(statusCode)) {
      safe.statusCode = statusCode as number
    }
  }

  return safe
}

/**
 * Format a backend failure for the desktop log without leaking remote
 * transport material. Normal Desktop logs retain their existing diagnostics;
 * the standalone flavor only records the safe category because a network
 * library may include a URL query, authorization value, or response body.
 */
export function formatBackendErrorForLog(error: unknown, remoteOnly: boolean, includeStack = false): string {
  if (remoteOnly) {
    return sanitizeRemoteSetupError(error)
  }

  if (error instanceof Error) {
    return includeStack ? error.stack || error.message : error.message
  }

  return String(error)
}

export function shouldResumeRemoteConnectionGate(
  remoteOnly: boolean,
  profile: null | string,
  waiting: boolean
): boolean {
  return remoteOnly && !profile && waiting
}

export function createRemoteConnectionGate(): RemoteConnectionGate {
  let waiter: { promise: Promise<void>; resolve: () => void } | null = null

  return {
    hasWaiter: () => Boolean(waiter),
    resume: () => {
      const active = waiter

      waiter = null
      active?.resolve()
    },
    wait: () => {
      if (waiter) {
        return waiter.promise
      }

      let resolveWaiter: () => void = () => {}

      const promise = new Promise<void>(resolve => {
        resolveWaiter = resolve
      })

      waiter = { promise, resolve: resolveWaiter }

      return promise
    }
  }
}
