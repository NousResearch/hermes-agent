const HERMES_READY_TIMEOUT_MS = 45_000
const HERMES_HEALTH_REQUEST_TIMEOUT_MS = 2_500
const HERMES_READY_RETRY_MS = 500

type ReadinessRequest = (url: string, options: { timeoutMs: number }) => Promise<unknown>

type ReadinessOptions = {
  now?: () => number
  sleep?: (delayMs: number) => Promise<void>
  totalTimeoutMs?: number
  requestTimeoutMs?: number
  retryMs?: number
  signal?: AbortSignal
}

function supersededBootstrapError() {
  const error: any = new Error('SSH bootstrap was superseded by newer connection settings.')
  error.kind = 'superseded'

  return error
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

function isMissingHealthEndpointError(error: unknown) {
  const message = errorMessage(error)

  if (/^404(?:\s*:|$)/.test(message)) {
    return true
  }

  return (
    message.includes('Expected JSON from ') &&
    message.includes(' got HTML ') &&
    message.includes('The endpoint is likely missing on the Hermes backend.')
  )
}

function isReadyHealthResponse(body: unknown) {
  if (!body || typeof body !== 'object') {
    return false
  }

  const health = body as { ok?: unknown; status?: unknown }

  return health.ok === true && health.status === 'ready'
}

async function waitForHermesReadiness(
  baseUrl: string,
  request: ReadinessRequest,
  options: ReadinessOptions = {}
) {
  const now = options.now ?? Date.now

  const sleep =
    options.sleep ??
    ((delayMs: number) => {
      return new Promise<void>(resolve => setTimeout(resolve, delayMs))
    })

  const totalTimeoutMs = options.totalTimeoutMs ?? HERMES_READY_TIMEOUT_MS
  const requestTimeoutMs = options.requestTimeoutMs ?? HERMES_HEALTH_REQUEST_TIMEOUT_MS
  const retryMs = options.retryMs ?? HERMES_READY_RETRY_MS
  const signal = options.signal
  const deadline = now() + totalTimeoutMs

  const throwIfSuperseded = () => {
    if (signal?.aborted) {
      throw supersededBootstrapError()
    }
  }

  const sleepUntilRetry = (delayMs: number) => {
    if (!signal) {
      return sleep(delayMs)
    }

    return new Promise<void>((resolve, reject) => {
      const onAbort = () => reject(supersededBootstrapError())
      signal.addEventListener('abort', onAbort, { once: true })
      sleep(delayMs).then(
        () => {
          signal.removeEventListener('abort', onAbort)
          resolve()
        },
        error => {
          signal.removeEventListener('abort', onAbort)
          reject(error)
        }
      )
    })
  }

  let useLegacyStatus = false
  let lastError: unknown = null

  while (now() < deadline) {
    throwIfSuperseded()
    const path = useLegacyStatus ? '/api/status' : '/api/healthz'
    const remainingMs = deadline - now()

    if (remainingMs <= 0) {
      break
    }

    try {
      const body = await request(`${baseUrl}${path}`, {
        timeoutMs: Math.min(requestTimeoutMs, remainingMs)
      })

      if (useLegacyStatus || isReadyHealthResponse(body)) {
        return
      }

      lastError = new Error('Hermes health endpoint did not report ready')
    } catch (error) {
      lastError = error

      if (!useLegacyStatus && isMissingHealthEndpointError(error)) {
        useLegacyStatus = true

        continue
      }
    }

    const remainingAfterAttemptMs = deadline - now()

    if (remainingAfterAttemptMs > 0) {
      await sleepUntilRetry(Math.min(retryMs, remainingAfterAttemptMs))
    }
  }

  throw new Error(`Hermes backend did not become ready: ${errorMessage(lastError || 'timeout')}`)
}

export {
  HERMES_HEALTH_REQUEST_TIMEOUT_MS,
  HERMES_READY_RETRY_MS,
  HERMES_READY_TIMEOUT_MS,
  isMissingHealthEndpointError,
  isReadyHealthResponse,
  waitForHermesReadiness
}
