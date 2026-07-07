export const DEFAULT_BACKEND_READY_TIMEOUT_MS = 45_000
export const DEFAULT_BACKEND_READY_POLL_MS = 500

type FetchPublicJson = (url: string) => Promise<unknown>
type FetchJson = (url: string, token?: string | null) => Promise<unknown>

export interface HermesReadyOptions {
  fetchPublicJson: FetchPublicJson
  fetchJson: FetchJson
  token?: string | null
  timeoutMs?: number
  pollMs?: number
  sleep?: (ms: number) => Promise<void>
  now?: () => number
}

export function isMissingHealthEndpointError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return /^404:/.test(message) || message.includes('endpoint is likely missing')
}

export async function waitForHermesReady(baseUrl: string, options: HermesReadyOptions): Promise<void> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_BACKEND_READY_TIMEOUT_MS
  const pollMs = options.pollMs ?? DEFAULT_BACKEND_READY_POLL_MS
  const sleep = options.sleep ?? (ms => new Promise(resolve => setTimeout(resolve, ms)))
  const now = options.now ?? Date.now
  const base = baseUrl.replace(/\/+$/, '')
  const deadline = now() + timeoutMs
  let lastError: unknown = null
  let useStatusFallback = false

  while (now() < deadline) {
    try {
      if (useStatusFallback) {
        await options.fetchJson(`${base}/api/status`, options.token)
      } else {
        await options.fetchPublicJson(`${base}/api/health`)
      }

      return
    } catch (error) {
      lastError = error

      if (!useStatusFallback && isMissingHealthEndpointError(error)) {
        useStatusFallback = true

        try {
          await options.fetchJson(`${base}/api/status`, options.token)

          return
        } catch (statusError) {
          lastError = statusError
        }
      }

      await sleep(pollMs)
    }
  }

  const detail = lastError instanceof Error ? lastError.message : 'timeout'
  throw new Error(`Hermes backend did not become ready: ${detail}`)
}
