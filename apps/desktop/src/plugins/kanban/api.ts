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

import {
  atom,
  computed,
  host,
  type PluginOs,
  type PluginRestOptions,
  type PluginStorage,
  type PluginTranslate,
  queryClient
} from '@hermes/plugin-sdk'

// Native completion notification.
import { bindCompletionNotify, type CompletionEvent, onKanbanEventsFrame } from './completion-notify'
import type {
  BoardExportResult,
  BoardImportResult,
  BoardMeta,
  BoardsResponse,
  KanbanBoard,
  KanbanProfile,
  KanbanProject,
  KanbanTask,
  KanbanTaskDetail,
  OrchestrationSettings,
  TaskEstimate,
  WorkerLog
} from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>
type Socket = (path: string, onMessage: (data: unknown) => void) => () => void

let rest: null | Rest = null
let os: null | PluginOs = null

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

/** Connection + profile own separate Kanban databases. Keep that server-side
 *  ownership in every client cache key and live-event subscription too. */
export const $kanbanScope = computed(
  [host.state.connectionId, host.state.profile],
  (connectionId, profile) => `${connectionId ?? 'local'}::${profile || 'default'}`
)

const BOARD_SLUG_KEY = 'boardSlug'
const BOARD_SLUGS_KEY = 'boardSlugsByScope'
const INTRO_KEY = 'introDismissed'
const LANES_KEY = 'lanesByProfile'
const COLLAPSED_KEY = 'collapsedLanes'

/** One live `task_events` frame → precise cache invalidation: the board, plus
 *  each touched task's detail. Slow polls stay as the fallback — the socket
 *  just makes the board feel instant. */
function onEventsFrame(scope: string, slug: string, data: unknown): void {
  const events = (data as { events?: CompletionEvent[] })?.events

  if (!events?.length) {
    return
  }

  void queryClient.invalidateQueries({ queryKey: ['kanban', 'board', scope, slug] })
  // Any event can change a board's card count — keep the switcher badge honest.
  void queryClient.invalidateQueries({ queryKey: boardsKey(scope) })

  for (const taskId of new Set(events.map(event => event.task_id).filter(Boolean))) {
    void queryClient.invalidateQueries({ queryKey: taskKey(slug, taskId!, scope) })
  }

  // Completion notification (after invalidation so notify failure
  // never interferes with cache invalidation).
  if (scope === $kanbanScope.get()) {
    void onKanbanEventsFrame(slug, events, scope).catch(() => undefined)
  }
}

// A persisted, subscribable atom (the structural slice we need — avoids
// importing nanostore's type just to describe one).
interface Persisted<T> {
  get(): T
  set(value: T): void
  listen(cb: (value: T) => void): () => void
}

/** Bind the plugin's doors at register time and return a disposer the host
 *  runs on unload/disable — so nothing (store sync, socket) survives a toggle
 *  or duplicates on re-enable. The events socket is pinned to a board at
 *  handshake, so a board switch closes + reopens it. */
export function bindApi(
  r: Rest,
  storage: PluginStorage,
  socket: Socket,
  notifyDoors?: { os?: PluginOs; t?: PluginTranslate }
): () => void {
  rest = r
  os = notifyDoors?.os ?? null
  bindCompletionNotify(r, notifyDoors?.t, notifyDoors?.os)
  const unsubs: Array<() => void> = []

  // Hydrate an atom from storage and keep storage in sync with it.
  const persist = <T>(atom: Persisted<T>, key: string, fallback: T) => {
    atom.set(storage.get(key, fallback))
    unsubs.push(atom.listen(value => storage.set(key, value)))
  }

  persist($introDismissed, INTRO_KEY, false)
  persist($lanesByProfile, LANES_KEY, false)
  persist($collapsedLanes, COLLAPSED_KEY, {})

  // A profile owns its own boards, so its selected slug cannot be shared with
  // another profile. Seed the active scope from the legacy single value once.
  let boardSlugs = storage.get<Record<string, string>>(BOARD_SLUGS_KEY, {})
  const initialScope = $kanbanScope.get()

  if (!(initialScope in boardSlugs)) {
    boardSlugs = { ...boardSlugs, [initialScope]: storage.get(BOARD_SLUG_KEY, '') }
    storage.set(BOARD_SLUGS_KEY, boardSlugs)
  }

  $boardSlug.set(boardSlugs[initialScope] ?? '')
  unsubs.push(
    $kanbanScope.listen(scope => $boardSlug.set(boardSlugs[scope] ?? '')),
    $boardSlug.listen(slug => {
      const scope = $kanbanScope.get()
      boardSlugs = { ...boardSlugs, [scope]: slug }
      storage.set(BOARD_SLUGS_KEY, boardSlugs)
    })
  )

  let close: (() => void) | null = null
  let activeRoute = ''

  const open = () => {
    const scope = $kanbanScope.get()
    const slug = $boardSlug.get()
    const route = `${scope}\0${slug}`

    if (route === activeRoute) {
      return
    }

    activeRoute = route
    close?.()
    close = socket(slug ? `/events?board=${encodeURIComponent(slug)}` : '/events', data =>
      onEventsFrame(scope, slug, data)
    )
  }

  open()
  unsubs.push($boardSlug.listen(open), $kanbanScope.listen(open))

  return () => {
    unsubs.forEach(unsub => unsub())
    close?.()
    rest = null
    os = null
  }
}

/** The plugin's OS door, for components too deep to be handed `ctx`. Null
 *  before `bindApi` and after unload. */
export const pluginOs = (): null | PluginOs => os

function call<T>(path: string, opts?: PluginRestOptions): Promise<T> {
  return rest ? rest<T>(path, opts) : Promise.reject(new Error('kanban api not ready'))
}

/** A query key captures its server scope during render. Refuse to start or
 *  commit a read after that scope changes, so a request can never populate a
 *  cache entry owned by a different connection/profile. */
async function scopedRead<T>(scope: string, path: string): Promise<T> {
  if (scope !== currentScope()) {
    throw new Error('Kanban server scope changed before the request started')
  }

  const value = await call<T>(path)

  if (scope !== currentScope()) {
    throw new Error('Kanban server scope changed before the response arrived')
  }

  return value
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

// ── query keys (server + board scoped, so either switch is a clean miss) ─────

const currentScope = () => $kanbanScope.get()

export const boardKey = (slug: string, archived: boolean, scope = currentScope()) =>
  ['kanban', 'board', scope, slug, archived] as const
export const taskKey = (slug: string, id: string, scope = currentScope()) =>
  ['kanban', 'task', scope, slug, id] as const
export const logKey = (slug: string, id: string, scope = currentScope()) =>
  ['kanban', 'log', scope, slug, id] as const
export const boardsKey = (scope = currentScope()) => ['kanban', 'boards', scope] as const
export const profilesKey = (scope = currentScope()) => ['kanban', 'profiles', scope] as const
export const projectsKey = (scope = currentScope()) => ['kanban', 'projects', scope] as const
export const orchestrationKey = (scope = currentScope()) => ['kanban', 'orchestration', scope] as const

// ── reads ─────────────────────────────────────────────────────────────────────

export const fetchBoard = (archived: boolean, scope: string) =>
  scopedRead<KanbanBoard>(scope, withBoard('/board', archived ? { include_archived: 'true' } : {}))

export const fetchTask = (id: string, scope: string) => scopedRead<KanbanTaskDetail>(scope, withBoard(`/tasks/${id}`))

/** Worker stdout/stderr tail (last 16 KiB — plenty for the drawer). */
export const fetchLog = (id: string, scope: string) =>
  scopedRead<WorkerLog>(scope, withBoard(`/tasks/${id}/log`, { tail: '16384' }))

export const fetchBoards = (scope: string) => scopedRead<BoardsResponse>(scope, '/boards')

export const fetchProfiles = (scope: string) => scopedRead<{ profiles: KanbanProfile[] }>(scope, '/profiles')

/** First-class Hermes projects, for scoping a board's default workspace. */
export const fetchProjects = (scope: string) => scopedRead<{ projects: KanbanProject[] }>(scope, '/projects')

export const fetchOrchestration = (scope: string) => scopedRead<OrchestrationSettings>(scope, '/orchestration')

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

/** Rough auxiliary-model estimate for a task (tokens + complexity). Makes a
 *  model call — gate behind an explicit user action + disclaimer. */
export const estimateTask = (id: string) =>
  call<TaskEstimate>(withBoard(`/tasks/${id}/estimate`), { method: 'POST', body: {} })

/** Estimate from typed title/body before a task exists (create dialog). */
export const estimateNew = (title: string, body: string) =>
  call<TaskEstimate>('/estimate', { method: 'POST', body: { title, body: body || undefined } })

/** Edit a board's display metadata + default project directory. Pass
 *  `default_workdir: ''` to clear it. Slug is immutable. */
export const updateBoard = (slug: string, patch: Record<string, unknown>) =>
  call<{ board: BoardMeta }>(`/boards/${encodeURIComponent(slug)}`, { method: 'PATCH', body: patch })

/** Archive a board to `boards/_archived/` — recoverable, and the backend
 *  refuses to touch `default`. (`?delete=true` hard-deletes; no caller yet.) */
export const deleteBoard = (slug: string) =>
  call<{ result: { action: string; new_path: string }; current: string }>(`/boards/${encodeURIComponent(slug)}`, {
    method: 'DELETE'
  })

// Board transfer exchanges filesystem paths, not bytes — the picker runs on
// the machine hosting the backend, so the backend reads and writes the file.

export const exportBoard = (slug: string, output: string) =>
  call<BoardExportResult>(`/boards/${encodeURIComponent(slug)}/export`, { method: 'POST', body: { output } })

export const importBoard = (archive: string) =>
  call<BoardImportResult>('/boards/import', { method: 'POST', body: { archive } })

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
