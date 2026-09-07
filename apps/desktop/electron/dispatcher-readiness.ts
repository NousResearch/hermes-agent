export interface DispatcherReadiness {
  status: 'ready' | 'offline' | 'disabled' | 'unknown'
  ready: boolean
  gateway_pid: number | null
  message: string
}

export class DispatcherReadinessError extends Error {
  readonly code = 'dispatcher-offline'
  readonly blocking = true

  constructor(detail: string) {
    super(`KANBAN_DISPATCHER_OFFLINE: ${detail}`)
    this.name = 'DispatcherReadinessError'
  }
}

type FetchJson = (
  url: string,
  token: string | null,
  options?: { timeoutMs?: number; method?: string }
) => Promise<unknown>

interface DispatcherReadinessOptions {
  attempts?: number
  pollMs?: number
  sleep?: (ms: number) => Promise<void>
}

const DEFAULT_START_ATTEMPTS = 60
const DEFAULT_START_POLL_MS = 250

function parseReadiness(payload: unknown): Partial<DispatcherReadiness> {
  if (!payload || typeof payload !== 'object') {
    throw new DispatcherReadinessError('dispatcher readiness returned an invalid response')
  }

  return payload as Partial<DispatcherReadiness>
}

function readinessDetail(result: Partial<DispatcherReadiness>): string {
  const status = typeof result.status === 'string' ? result.status : 'unknown'
  const detail = typeof result.message === 'string' ? result.message : 'no readiness detail was returned'

  return `${status}: ${detail}`
}

export async function ensureKanbanDispatcherReady(
  baseUrl: string,
  token: string,
  fetchJson: FetchJson,
  options: DispatcherReadinessOptions = {}
): Promise<DispatcherReadiness> {
  const normalizedBaseUrl = baseUrl.replace(/\/+$/, '')
  const url = `${normalizedBaseUrl}/api/plugins/kanban/dispatcher-readiness`
  let payload: unknown

  try {
    payload = await fetchJson(url, token, { timeoutMs: 5_000 })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new DispatcherReadinessError(`dispatcher readiness could not be verified: ${detail}`)
  }

  let result = parseReadiness(payload)

  if (result.status === 'ready' && result.ready === true) {
    return result as DispatcherReadiness
  }

  if (result.status !== 'offline') {
    throw new DispatcherReadinessError(readinessDetail(result))
  }

  try {
    await fetchJson(`${normalizedBaseUrl}/api/gateway/start`, token, { method: 'POST', timeoutMs: 10_000 })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new DispatcherReadinessError(`dispatcher gateway could not start: ${detail}`)
  }

  const attempts = options.attempts ?? DEFAULT_START_ATTEMPTS
  const pollMs = options.pollMs ?? DEFAULT_START_POLL_MS
  const sleep = options.sleep ?? (ms => new Promise(resolve => setTimeout(resolve, ms)))

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    await sleep(pollMs)

    try {
      result = parseReadiness(await fetchJson(url, token, { timeoutMs: 5_000 }))
    } catch (error) {
      if (attempt + 1 >= attempts) {
        const detail = error instanceof Error ? error.message : String(error)
        throw new DispatcherReadinessError(`dispatcher readiness could not be verified after gateway start: ${detail}`)
      }
      continue
    }

    if (result.status === 'ready' && result.ready === true) {
      return result as DispatcherReadiness
    }

    if (result.status !== 'offline') {
      throw new DispatcherReadinessError(readinessDetail(result))
    }
  }

  throw new DispatcherReadinessError(`dispatcher did not become ready after gateway start: ${readinessDetail(result)}`)
}
