/**
 * Kanban data layer. Everything goes through `ctx.rest` — the plugin's own
 * `/api/plugins/kanban/*` FastAPI router (`plugins/kanban/dashboard/plugin_api.py`),
 * reused as-is via the desktop's namespace-scoped REST door. No new backend.
 *
 * Fetching, caching, polling, dedupe, and invalidation are React Query's job
 * (the app's standard, via the SDK). This module owns the query keys, the REST
 * calls, and the selected-board atom — every call passes `?board=<slug>` so the
 * desktop's selection never flips the server-wide current-board pointer.
 */

import { atom, type PluginRestOptions, type PluginStorage, queryClient } from '@hermes/plugin-sdk'

import type {
  BoardMeta,
  BoardsResponse,
  KanbanBoard,
  KanbanProfile,
  KanbanProject,
  KanbanTask,
  KanbanTaskDetail,
  OrchestrationSettings,
  WorkerLog
} from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>
type Socket = (path: string, onMessage: (data: unknown) => void) => () => void

let rest: null | Rest = null

/** Selected board slug ('' = the server's current board). Persisted. */
export const $boardSlug = atom<string>('')

/** Whether the "how this board works" intro was dismissed. Persisted. */
export const $introDismissed = atom<boolean>(false)

/** Sub-group the Running lane by assignee (the dashboard's "lanes by
 *  profile"). Persisted. */
export const $lanesByProfile = atom<boolean>(false)

/** Per-lane collapse OVERRIDES (true=collapsed, false=expanded). Absence means
 *  auto: empty lanes collapse to a rail, occupied lanes expand. Persisted. */
export const $collapsedLanes = atom<Record<string, boolean>>({})

const BOARD_SLUG_KEY = 'boardSlug'
const INTRO_KEY = 'introDismissed'
const LANES_KEY = 'lanesByProfile'
const COLLAPSED_KEY = 'collapsedLanes'

/** One live `task_events` frame → precise cache invalidation: the board, plus
 *  each touched task's detail. The polls (8s board / 4s drawer) stay as the
 *  fallback — the socket just makes the board feel instant. */
function onEventsFrame(slug: string, data: unknown): void {
  const events = (data as { events?: Array<{ task_id?: string }> })?.events

  if (!events?.length) {
    return
  }

  void queryClient.invalidateQueries({ queryKey: ['kanban', 'board'] })
  // Any event can change a board's card count — keep the switcher badge honest.
  void queryClient.invalidateQueries({ queryKey: BOARDS_KEY })

  for (const taskId of new Set(events.map(event => event.task_id).filter(Boolean))) {
    void queryClient.invalidateQueries({ queryKey: taskKey(slug, taskId!) })
  }
}

/** Bind the plugin's doors once, at register time. The events socket is pinned
 *  to a board at handshake, so a board switch closes + reopens it. */
export function bindApi(r: Rest, storage: PluginStorage, socket: Socket): void {
  rest = r
  $boardSlug.set(storage.get(BOARD_SLUG_KEY, ''))
  $boardSlug.listen(slug => storage.set(BOARD_SLUG_KEY, slug))
  $introDismissed.set(storage.get(INTRO_KEY, false))
  $introDismissed.listen(dismissed => storage.set(INTRO_KEY, dismissed))
  $lanesByProfile.set(storage.get(LANES_KEY, false))
  $lanesByProfile.listen(on => storage.set(LANES_KEY, on))
  $collapsedLanes.set(storage.get(COLLAPSED_KEY, {}))
  $collapsedLanes.listen(map => storage.set(COLLAPSED_KEY, map))

  let close: (() => void) | null = null

  const open = (slug: string) => {
    close?.()
    close = socket(slug ? `/events?board=${encodeURIComponent(slug)}` : '/events', data => onEventsFrame(slug, data))
  }

  open($boardSlug.get())
  $boardSlug.listen(open)
}

function call<T>(path: string, opts?: PluginRestOptions): Promise<T> {
  return rest ? rest<T>(path, opts) : Promise.reject(new Error('kanban api not ready'))
}

/** Append the selected board (and other params) to a path. */
function withBoard(path: string, params: Record<string, string> = {}): string {
  const search = new URLSearchParams(params)
  const slug = $boardSlug.get()

  if (slug) {
    search.set('board', slug)
  }

  const qs = search.toString()

  return qs ? `${path}?${qs}` : path
}

// ── query keys (all board-scoped so switching boards is a clean cache miss) ──

export const boardKey = (slug: string, archived: boolean) => ['kanban', 'board', slug, archived] as const
export const taskKey = (slug: string, id: string) => ['kanban', 'task', slug, id] as const
export const logKey = (slug: string, id: string) => ['kanban', 'log', slug, id] as const
export const BOARDS_KEY = ['kanban', 'boards'] as const
export const PROFILES_KEY = ['kanban', 'profiles'] as const
export const PROJECTS_KEY = ['kanban', 'projects'] as const
export const ORCHESTRATION_KEY = ['kanban', 'orchestration'] as const

// ── reads ─────────────────────────────────────────────────────────────────────

export const fetchBoard = (archived: boolean) =>
  call<KanbanBoard>(withBoard('/board', archived ? { include_archived: 'true' } : {}))

export const fetchTask = (id: string) => call<KanbanTaskDetail>(withBoard(`/tasks/${id}`))

/** Worker stdout/stderr tail (last 16 KiB — plenty for the drawer). */
export const fetchLog = (id: string) => call<WorkerLog>(withBoard(`/tasks/${id}/log`, { tail: '16384' }))

export const fetchBoards = () => call<BoardsResponse>('/boards')

export const fetchProfiles = () => call<{ profiles: KanbanProfile[] }>('/profiles')

/** First-class Hermes projects, for scoping a board's default workspace. */
export const fetchProjects = () => call<{ projects: KanbanProject[] }>('/projects')

export const fetchOrchestration = () => call<OrchestrationSettings>('/orchestration')

// ── writes ────────────────────────────────────────────────────────────────────

// Every board edit nudges the dispatcher (debounced, fire-and-forget) so the
// change takes effect NOW instead of on the next 60s tick — create a ready
// task and the worker spawns immediately, no manual "nudge" ritual. The tick
// is lock-guarded and ~1ms when there's nothing to do, so over-nudging is
// free; failures are non-events (the periodic tick still exists).
let nudgeTimer: null | ReturnType<typeof setTimeout> = null

function autoNudge(): void {
  if (nudgeTimer != null) {
    clearTimeout(nudgeTimer)
  }

  nudgeTimer = setTimeout(() => {
    nudgeTimer = null
    nudgeDispatcher().catch(() => undefined)
  }, 400)
}

/** Resolve the write, then kick the dispatcher. Rejections pass through. */
function nudged<T>(write: Promise<T>): Promise<T> {
  return write.then(value => {
    autoNudge()

    return value
  })
}

export const patchTask = (id: string, patch: Record<string, unknown>) =>
  nudged(call(withBoard(`/tasks/${id}`), { method: 'PATCH', body: patch }))

export const createTask = (body: Record<string, unknown>) =>
  nudged(call<{ task: KanbanTask | null; warning?: string }>(withBoard('/tasks'), { method: 'POST', body }))

// Deleting can unblock dependants (a gone parent no longer gates), so it
// nudges too.
export const deleteTask = (id: string) => nudged(call(withBoard(`/tasks/${id}`), { method: 'DELETE' }))

/** One patch, many ids — independent per-id application; returns per-id
 *  outcomes so the UI can toast partial failures. */
export const bulkTasks = (ids: string[], patch: Record<string, unknown>) =>
  nudged(
    call<{ results: Array<{ id: string; ok: boolean; error?: string }> }>(withBoard('/tasks/bulk'), {
      method: 'POST',
      body: { ids, ...patch }
    })
  )

export const addComment = (id: string, body: string) =>
  call(withBoard(`/tasks/${id}/comments`), { method: 'POST', body: { author: 'desktop', body } })

export const reassignTask = (id: string, profile: string) =>
  nudged(call(withBoard(`/tasks/${id}/reassign`), { method: 'POST', body: { profile, reclaim_first: true } }))

export const reclaimTask = (id: string) => nudged(call(withBoard(`/tasks/${id}/reclaim`), { method: 'POST', body: {} }))

export const uploadAttachment = (id: string, upload: { filename: string; contentType?: string; bytes: ArrayBuffer }) =>
  call(withBoard(`/tasks/${id}/attachments`), { method: 'POST', upload })

export const createBoard = (slug: string, name: string, projectId?: string) =>
  call<{ board: { slug: string } }>('/boards', {
    method: 'POST',
    body: { slug, name, ...(projectId ? { project_id: projectId } : {}) }
  })

/** Edit a board's display metadata + default project directory. Pass
 *  `default_workdir: ''` to clear it. Slug is immutable. */
export const updateBoard = (slug: string, patch: Record<string, unknown>) =>
  call<{ board: BoardMeta }>(`/boards/${encodeURIComponent(slug)}`, { method: 'PATCH', body: patch })

export const nudgeDispatcher = () => call<{ spawned?: unknown[] }>(withBoard('/dispatch'), { method: 'POST', body: {} })

export const saveOrchestration = (patch: Record<string, unknown>) =>
  call<OrchestrationSettings>('/orchestration', { method: 'PUT', body: patch })

export const saveProfileDescription = (name: string, description: string) =>
  call(`/profiles/${encodeURIComponent(name)}`, { method: 'PATCH', body: { description } })

export const autoDescribeProfile = (name: string) =>
  call<{ ok: boolean; reason?: null | string; description?: null | string }>(
    `/profiles/${encodeURIComponent(name)}/describe-auto`,
    { method: 'POST', body: { overwrite: true } }
  )
