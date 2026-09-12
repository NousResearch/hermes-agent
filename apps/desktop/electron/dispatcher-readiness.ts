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

type FetchJson = (url: string, token: string | null, options?: { timeoutMs?: number }) => Promise<unknown>

export async function ensureKanbanDispatcherReady(
  baseUrl: string,
  token: string,
  fetchJson: FetchJson
): Promise<DispatcherReadiness> {
  const url = `${baseUrl.replace(/\/+$/, '')}/api/plugins/kanban/dispatcher-readiness`
  let payload: unknown

  try {
    payload = await fetchJson(url, token, { timeoutMs: 5_000 })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new DispatcherReadinessError(`dispatcher readiness could not be verified: ${detail}`)
  }

  if (!payload || typeof payload !== 'object') {
    throw new DispatcherReadinessError('dispatcher readiness returned an invalid response')
  }

  const result = payload as Partial<DispatcherReadiness>

  if (result.status !== 'ready' || result.ready !== true) {
    const status = typeof result.status === 'string' ? result.status : 'unknown'
    const detail = typeof result.message === 'string' ? result.message : 'no readiness detail was returned'
    throw new DispatcherReadinessError(`${status}: ${detail}`)
  }

  return result as DispatcherReadiness
}
