export const DEFAULT_BACKEND_READY_TIMEOUT_MS = 45_000
export const DEFAULT_BACKEND_READY_POLL_MS = 500
// A cold backend can stall its event loop for tens of seconds while Windows
// scans and byte-compiles the gateway import tree. At the default 15s socket
// timeout only three probes fit in the budget; a short one keeps retrying
// across the stall. Health only — the legacy /api/status fallback is genuinely
// slow to answer and keeps the caller's default timeout.
export const DEFAULT_HEALTH_PROBE_TIMEOUT_MS = 5_000

type FetchPublicJson = (url: string, options?: { timeoutMs?: number }) => Promise<unknown>
type FetchJson = (url: string, token?: string | null, options?: { timeoutMs?: number }) => Promise<unknown>

export interface HermesReadyOptions {
  fetchPublicJson: FetchPublicJson
  fetchJson: FetchJson
  token?: string | null
  signal?: AbortSignal
  timeoutMs?: number
  pollMs?: number
  healthProbeTimeoutMs?: number
  sleep?: (ms: number) => Promise<void>
  now?: () => number
  /**
   * Credentialed health probe. When supplied, readiness is probed with the
   * connection's own credentials instead of anonymously — which is what lets
   * a gated backend answer 404 for a genuinely missing /api/health, and what
   * makes a 401 from this probe mean "session rejected" rather than "route
   * behind a gate". Defaults to the credential-free `fetchPublicJson`.
   */
  probeHealth?: (url: string, options?: { timeoutMs?: number }) => Promise<unknown>
  /**
   * Whether `probeHealth` actually presents credentials. Distinguishes the
   * two very different meanings of a 401 (see `waitForHermesReady`).
   */
  probeIsCredentialed?: boolean
}

export const REMOTE_SESSION_EXPIRED_MESSAGE =
  'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'

export function isMissingHealthEndpointError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return /^404:/.test(message) || message.includes('endpoint is likely missing')
}

/**
 * True for a hard auth rejection (401/403) as opposed to a transient failure.
 * Deliberately shape-based: 429 is a throttle and 5xx is a server fault, and
 * both must keep polling.
 */
export function isAuthRejectionError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return /^40[13]:/.test(message)
}

/**
 * True for an auth rejection carrying the dashboard gate's "no session at all"
 * shape. On a backend that predates `/api/health`, the gate runs ahead of the
 * SPA catch-all, so an unknown `/api/*` path is rejected as unauthenticated
 * instead of 404 — this is the signal that an ANONYMOUS probe cannot reach the
 * route, and the reason a credential-free 401 must fall back to `/api/status`
 * rather than be reported as a boot failure.
 */
export function isGatedMissingHealthError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return isAuthRejectionError(error) && message.includes('no_cookie')
}

/** Tag a terminal reauth failure the main process latches and the overlay keys on. */
export function makeReauthRequiredError(detail?: string): Error {
  const error = new Error(REMOTE_SESSION_EXPIRED_MESSAGE) as any
  error.needsOauthLogin = true
  error.isReauthRequired = true

  if (detail) {
    error.detail = detail
  }

  return error
}

export function isReauthRequiredError(error: unknown): boolean {
  return Boolean((error as any)?.isReauthRequired)
}

function supersededError() {
  const error: any = new Error('SSH bootstrap was superseded by newer connection settings.')
  error.kind = 'superseded'

  return error
}

export async function waitForHermesReady(baseUrl: string, options: HermesReadyOptions): Promise<void> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_BACKEND_READY_TIMEOUT_MS
  const pollMs = options.pollMs ?? DEFAULT_BACKEND_READY_POLL_MS
  const healthProbeTimeoutMs = options.healthProbeTimeoutMs ?? DEFAULT_HEALTH_PROBE_TIMEOUT_MS
  const now = options.now ?? Date.now
  const signal = options.signal

  const sleep =
    options.sleep ??
    (ms =>
      new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, ms)
        signal?.addEventListener(
          'abort',
          () => {
            clearTimeout(timer)
            reject(supersededError())
          },
          { once: true }
        )
      }))

  const base = baseUrl.replace(/\/+$/, '')
  const deadline = now() + timeoutMs
  const probeHealth = options.probeHealth ?? options.fetchPublicJson
  const probeIsCredentialed = Boolean(options.probeIsCredentialed)
  let lastError: unknown = null
  let useStatusFallback = false

  while (now() < deadline) {
    if (signal?.aborted) {
      throw supersededError()
    }

    try {
      if (useStatusFallback) {
        await options.fetchJson(`${base}/api/status`, options.token)
      } else {
        await probeHealth(`${base}/api/health`, { timeoutMs: healthProbeTimeoutMs })
      }

      return
    } catch (error) {
      lastError = error

      // A confirmed 401/403 from a CREDENTIALED probe means the session was
      // rejected, not that the route is missing. Fail fast into a reauth
      // state: falling back to the public /api/status would answer 200 and
      // report a dead session as "ready", deferring the failure to the first
      // real API call. Applies to the /api/status leg too — it is routed
      // through the same credentials.
      if (probeIsCredentialed && isAuthRejectionError(error)) {
        throw makeReauthRequiredError(error instanceof Error ? error.message : String(error))
      }

      // An explicitly missing route means the backend predates /api/health.
      // So does a gate-shaped 401 on an ANONYMOUS probe: the dashboard auth
      // gate runs ahead of the SPA catch-all, so a pre-/api/health backend
      // rejects the unknown path as unauthenticated instead of 404 and a
      // credential-free probe can never observe the 404. Timeouts, 5xx, 429,
      // and non-gate 401s keep polling health.
      if (!useStatusFallback && (isMissingHealthEndpointError(error) || isGatedMissingHealthError(error))) {
        useStatusFallback = true

        continue
      }

      await sleep(pollMs)
    }
  }

  const detail = lastError instanceof Error ? lastError.message : 'timeout'
  throw new Error(`Hermes backend did not become ready: ${detail}`)
}
