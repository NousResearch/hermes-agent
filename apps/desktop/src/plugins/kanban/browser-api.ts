import { host, type PluginRestOptions } from '@hermes/plugin-sdk'

import type { BoardsResponse, KanbanBoard, KanbanTaskDetail } from './types'

export interface BoardSource {
  readonly connectionId: string
  readonly profile: string
  readonly label: string
}

export type BoardBrowserRest = <T>(path: string, options?: PluginRestOptions) => Promise<T>

export const browserKeys = {
  inventory: ['kanban-browser', 'sources'] as const,
  boards: (source: BoardSource) => ['kanban-browser', source.connectionId, source.profile, 'boards'] as const,
  board: (source: BoardSource, board: string) =>
    ['kanban-browser', source.connectionId, source.profile, 'board', board] as const,
  task: (source: BoardSource, board: string, task: string) =>
    ['kanban-browser', source.connectionId, source.profile, 'board', board, 'task', task] as const
}

export function browserFailure(error: unknown): 'auth' | 'unsupported' | 'unavailable' {
  const message = error instanceof Error ? error.message : String(error)

  if (/\b(401|403)\b/.test(message)) {
    return 'auth'
  }

  return /\b(404|405|501)\b|no connection registry/i.test(message) ? 'unsupported' : 'unavailable'
}

/** Retry transient reads twice, then stop until manual retry. Never loop on a
 *  denied/missing capability; successful visible queries refresh every 30s. */
export const browserRefresh = {
  retry: (attempt: number, error: unknown) => attempt < 2 && browserFailure(error) === 'unavailable',
  retryDelay: (attempt: number) => Math.min(1_000 * 2 ** attempt, 8_000),
  refetchInterval: (query: { state: { status: string } }) =>
    query.state.status === 'error' ? (false as const) : 30_000,
  refetchIntervalInBackground: false,
  refetchOnWindowFocus: false,
  refetchOnReconnect: false,
  staleTime: 15_000,
  gcTime: 60_000
}

/** One page-owned, bounded read queue. Cancellation discards queued work and
 *  late IPC replies; a hung registry dial cannot hold the renderer indefinitely.
 *  Electron remains the only owner of endpoints and credentials. */
export function createBoardBrowserReads(rest: BoardBrowserRest, connections = host.connections) {
  let active = 0
  const waiting: Array<() => void> = []

  const bounded = <T>(read: () => Promise<T>, signal: AbortSignal): Promise<T> =>
    new Promise((resolve, reject) => {
      let started = false
      let settled = false

      const finish = (error: unknown, value?: T) => {
        if (settled) {
          return
        }

        settled = true
        clearTimeout(timer)
        signal.removeEventListener('abort', abort)
        const queued = waiting.indexOf(start)

        if (queued >= 0) {
          waiting.splice(queued, 1)
        }

        if (started) {
          active -= 1
          waiting.shift()?.()
        }

        if (error) {
          reject(error)
        } else {
          resolve(value as T)
        }
      }

      const abort = () => finish(new DOMException('Read cancelled', 'AbortError'))

      const start = () => {
        if (signal.aborted) {
          abort()

          return
        }

        started = true
        active += 1
        void Promise.resolve()
          .then(read)
          .then(
            value => finish(null, value),
            error => finish(error)
          )
      }

      const timer = setTimeout(() => finish(new Error('Board source request timed out')), 15_000)
      signal.addEventListener('abort', abort, { once: true })

      if (signal.aborted) {
        abort()
      } else if (active < 3) {
        start()
      } else {
        waiting.push(start)
      }
    })

  const get = <T>(source: BoardSource, path: string, signal: AbortSignal) => {
    const pin = Object.freeze({ connectionId: source.connectionId, profile: source.profile })

    return bounded(() => rest<T>(path, { method: 'GET', source: pin, timeoutMs: 10_000 }), signal)
  }

  return {
    sources: (signal: AbortSignal): Promise<BoardSource[]> =>
      bounded(async () => {
        const rows = await connections()

        // A registry source is not proof of a unique physical board. Do not
        // deduplicate by slug, profile, or last-known installId.
        return rows.map(row =>
          Object.freeze({
            connectionId: row.id,
            label: row.label,
            profile: row.kind === 'ssh' && row.remoteProfile ? row.remoteProfile : 'default'
          })
        )
      }, signal),
    boards: async (source: BoardSource, signal: AbortSignal) => {
      const data = await get<BoardsResponse>(source, '/boards', signal)

      if (!Array.isArray(data?.boards) || data.boards.some(board => typeof board.slug !== 'string' || !board.slug)) {
        throw new Error('Invalid board inventory response')
      }

      return data
    },
    board: async (source: BoardSource, board: string, signal: AbortSignal) => {
      const data = await get<KanbanBoard>(source, `/board?${new URLSearchParams({ board })}`, signal)

      if (!Array.isArray(data?.columns) || data.columns.some(column => !Array.isArray(column.tasks))) {
        throw new Error('Invalid board response')
      }

      return data
    },
    task: async (source: BoardSource, board: string, task: string, signal: AbortSignal) => {
      const data = await get<KanbanTaskDetail>(
        source,
        `/tasks/${encodeURIComponent(task)}?${new URLSearchParams({ board })}`,
        signal
      )

      if (data?.task?.id !== task || !Array.isArray(data.runs) || !data.links) {
        throw new Error('Invalid task response')
      }

      return data
    }
  }
}

export type BoardBrowserReads = ReturnType<typeof createBoardBrowserReads>
