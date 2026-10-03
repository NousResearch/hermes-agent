// Attached-backend liveness policy for the Desktop's multiplex attach-first path.
//
// An attached backend has no child process, so `child.exit` can never drive
// recovery. The monitor polls readiness instead. A single slow probe (a 5s
// health timeout under SQLite load, Windows Defender, cold-start stalls) must
// not tear down a live backend: that invalidate() bumps the connection
// generation and every in-flight API caller fails with "superseded by a newer
// connection attempt", thrashing recovery into repeated reconnect storms.
//
// Policy (one resolver, every caller gets the same answer):
// - PID gone  -> dead immediately (no network I/O).
// - Token drift / credentialed 401/403 -> session invalid, recover immediately.
// - Anything else (timeout, reset, 5xx, refused while PID alive) -> transient.
//   Count consecutive transients; recover only at the threshold. A success
//   resets the streak, so intermittent slowness never accumulates into a teardown.

export const ATTACHED_LIVENESS_POLL_MS = 15_000
export const ATTACHED_LIVENESS_FAILURE_THRESHOLD = 3
export const ATTACHED_LIVENESS_PROBE_TIMEOUT_MS = 10_000

export type AttachedProbeFailureKind = 'transient' | 'hard'

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error ?? '')
}

export function isTimeoutLikeError(error: unknown): boolean {
  const message = messageOf(error).toLowerCase()
  const code = (error as { code?: unknown } | null)?.code

  if (typeof code === 'string') {
    const upper = code.toUpperCase()
    if (upper === 'ETIMEDOUT' || upper === 'ECONNRESET' || upper === 'EPIPE' || upper === 'UND_ERR_CONNECT_TIMEOUT') {
      return true
    }
  }

  return (
    message.includes('timed out') ||
    message.includes('timeout') ||
    message.includes('timeouterror') ||
    message.includes('econnreset') ||
    message.includes('socket hang up') ||
    message.includes('fetch failed')
  )
}

export function isAuthRejectionMessage(message: string): boolean {
  return /^40[13]:/.test(message)
}

export function classifyAttachedProbeError(error: unknown): AttachedProbeFailureKind {
  const message = messageOf(error)

  if (isAuthRejectionMessage(message)) {
    return 'hard'
  }

  // Credentialed probes arrive wrapped: backend-health's makeReauthRequiredError
  // replaces the message with the reauth prompt, the 401/403 surviving only in
  // `.detail` plus the reauth flags. Matching the message alone classifies a
  // rejected session as transient and retries a dead token to the threshold.
  const flagged = error as {
    detail?: unknown
    isReauthRequired?: unknown
    needsOauthLogin?: unknown
  } | null
  if (flagged !== null && typeof flagged === 'object') {
    if (flagged.isReauthRequired === true || flagged.needsOauthLogin === true) {
      return 'hard'
    }
    if (typeof flagged.detail === 'string' && isAuthRejectionMessage(flagged.detail)) {
      return 'hard'
    }
  }

  return 'transient'
}

export interface AttachedLivenessTracker {
  readonly consecutiveTransientFailures: number
  noteSuccess(): void
  noteTransientFailure(): boolean
  noteHardFailure(): boolean
  reset(): void
}

export function createAttachedLivenessTracker(
  failureThreshold: number = ATTACHED_LIVENESS_FAILURE_THRESHOLD
): AttachedLivenessTracker {
  let consecutiveTransientFailures = 0

  return {
    get consecutiveTransientFailures() {
      return consecutiveTransientFailures
    },

    noteSuccess(): void {
      consecutiveTransientFailures = 0
    },

    noteTransientFailure(): boolean {
      consecutiveTransientFailures += 1

      return consecutiveTransientFailures >= failureThreshold
    },

    noteHardFailure(): boolean {
      return true
    },

    reset(): void {
      consecutiveTransientFailures = 0
    }
  }
}

export function isTransientWsProbeReason(reason: unknown): boolean {
  const text = String(reason ?? '').toLowerCase()

  return (
    text.includes('timed out') ||
    text.includes('timeout') ||
    text.includes('econnreset') ||
    text.includes('socket hang up') ||
    text.includes('fetch failed') ||
    text.includes('websocket connection failed')
  )
}
