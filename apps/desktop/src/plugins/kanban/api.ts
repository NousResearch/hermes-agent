/**
 * Kanban data layer. Everything goes through `ctx.rest` — the plugin's own
 * `/api/plugins/kanban/*` FastAPI router (`plugins/kanban/dashboard/plugin_api.py`),
 * reused as-is via the desktop's namespace-scoped REST door. No new backend.
 *
 * Fetching, caching, polling, dedupe, and invalidation are React Query's job
 * (the app's standard, via the SDK). This module owns the query keys, the REST
 * calls, and the selected-board atom — every call passes `?board=<slug>` so the
 * desktop's selection never flips the server-wide current-board pointer.
 *
 * Every query key and the persisted board selection are scoped by the ACTIVE
 * CONNECTION (`host.state.connectionId`): a board lives on ONE gateway, so a
 * connection switch must be a clean cache miss (the hermes-bots roster
 * pattern), and each gateway remembers its own selected board instead of
 * pinning a slug the next gateway 404s on.
 */

import {
  atom,
  captureGatewayFileDownload,
  host,
  type PluginOs,
  type PluginRestOptions,
  type PluginStorage,
  type PluginTranslate,
  queryClient,
  useValue
} from '@hermes/plugin-sdk'

// Native completion notification.
import {
  bindCompletionNotify,
  type CompletionEvent,
  onKanbanEventsFrame,
  seedCompletionBaseline
} from './completion-notify'
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

/** Cache scope of the local pool — the SDK atom's own spelling. */
const LOCAL_SCOPE = 'local'

const KANBAN_KEY_ROOT = ['kanban'] as const

const BOARD_SLUG_KEY = 'boardSlug'
const INTRO_KEY = 'introDismissed'
const LANES_KEY = 'lanesByProfile'
const COLLAPSED_KEY = 'collapsedLanes'

/** Cache-scope id for the active connection — the segment every query key
 *  embeds. `'local'` covers the pre-descriptor null; the SDK atom already
 *  reports 'local' for the local pool. For NON-rendering code (mutations,
 *  socket frames); rendering components use `useKanbanScope` so the keys they
 *  build during render recompute when the connection changes. */
export function kanbanConnectionScope(): string {
  return host.state.connectionId.get() ?? LOCAL_SCOPE
}

export function useKanbanScope(): string {
  return useValue(host.state.connectionId) ?? LOCAL_SCOPE
}

/** Where a request issued NOW is routed, as a cache scope. The request tag
 *  moves before the connection descriptor publishes, and React re-keys the
 *  observers later still — so between the two an observer can sit on the
 *  outgoing scope's key while a fetch would land on the incoming backend. */
const routedScope = (): string => host.activeConnectionId() ?? LOCAL_SCOPE

/** `enabled` for every kanban query: only fetch while the key's scope is the
 *  routed one. A switch's app-wide invalidation then leaves the outgoing
 *  observers alone (the incoming keys are already a cache miss) instead of
 *  writing the new gateway's payload under the old connection's key — which
 *  would paint on the way back. Installed as the `['kanban']` query default in
 *  `bindApi`; sites with their own `enabled` compose it. */
export const routedToScope = (query: { queryKey: readonly unknown[] }): boolean => query.queryKey[2] === routedScope()

// Where each board's event stream left off this session. The socket reopens
// on every board switch and reconnect; resuming from the last frame's cursor
// replays only what was missed, never the board's whole history. Session-only
// on purpose: events that land while the app is closed are documented as not
// replayed on the next launch. Keyed by connection too: a slug names a
// different board (and event-id sequence) on every gateway.
const eventCursors = new Map<string, number>()

const cursorKey = (scope: string, slug: string) => `${scope}\n${slug}`

/** The cursor a fresh socket for `slug` on `scope` starts from: the last frame
 *  this session saw, else the cached board snapshot's `latest_event_id` (the
 *  board is already rendered from it), else nothing yet. */
function eventsSince(scope: string, slug: string): number | undefined {
  const seen = eventCursors.get(cursorKey(scope, slug))

  if (typeof seen === 'number') {
    return seen
  }

  for (const archived of [false, true]) {
    const board = queryClient.getQueryData<KanbanBoard>(boardKey(scope, slug, archived))

    if (typeof board?.latest_event_id === 'number') {
      return board.latest_event_id
    }
  }

  return undefined
}

/** One live `task_events` frame → precise cache invalidation: the board, plus
 *  each touched task's detail. The polls (60s board / 30s drawer) stay as the
 *  fallback — the socket just makes the board feel instant. */
function onEventsFrame(streamScope: string, slug: string, data: unknown): void {
  const frame = data as { cursor?: unknown; events?: CompletionEvent[] }
  const events = frame?.events

  if (typeof frame?.cursor === 'number') {
    eventCursors.set(cursorKey(streamScope, slug), frame.cursor)
  }

  if (!events?.length) {
    return
  }

  const scope = kanbanConnectionScope()
  void queryClient.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
  // Any event can change a board's card count — keep the switcher badge honest.
  void queryClient.invalidateQueries({ queryKey: boardsKey(scope) })

  for (const taskId of new Set(events.map(event => event.task_id).filter(Boolean))) {
    void queryClient.invalidateQueries({ queryKey: taskKey(scope, slug, taskId!) })
  }

  // Completion notification (after invalidation so notify failure
  // never interferes with cache invalidation).
  void onKanbanEventsFrame(slug, events).catch(() => undefined)
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

  queryClient.setQueryDefaults(KANBAN_KEY_ROOT, { enabled: routedToScope })
  unsubs.push(() => queryClient.setQueryDefaults(KANBAN_KEY_ROOT, {}))

  // Hydrate an atom from storage and keep storage in sync with it.
  const persist = <T>(atom: Persisted<T>, key: string, fallback: T) => {
    atom.set(storage.get(key, fallback))
    unsubs.push(atom.listen(value => storage.set(key, value)))
  }

  persist($introDismissed, INTRO_KEY, false)
  persist($lanesByProfile, LANES_KEY, false)
  persist($collapsedLanes, COLLAPSED_KEY, {})

  let close: (() => void) | null = null
  let dialGeneration = 0

  const open = (slug: string) => {
    const generation = ++dialGeneration
    close?.()
    close = null
    // The connection this socket dials: a frame arriving after a switch
    // still belongs to the gateway that sent it.
    const scope = routedScope()

    const dial = (since?: number) => {
      if (generation !== dialGeneration) {
        return
      }

      const params = new URLSearchParams()

      if (slug) {
        params.set('board', slug)
      }

      if (since !== undefined) {
        params.set('since', String(since))
        // Notifications baseline where the stream starts. Read from /board when
        // the first frame arrives, the baseline already includes that frame.
        seedCompletionBaseline(slug, since)
      }

      const query = params.toString()
      close = socket(query ? `/events?${query}` : '/events', data => onEventsFrame(scope, slug, data))
    }

    const since = eventsSince(scope, slug)

    if (since !== undefined) {
      dial(since)

      return
    }

    // Nothing known yet: start the stream at the snapshot the board renders
    // from (the same query), so no event can land between the two reads.
    void queryClient.fetchQuery({ queryFn: () => fetchBoard(false), queryKey: boardKey(scope, slug, false) }).then(
      board => dial(typeof board.latest_event_id === 'number' ? board.latest_event_id : undefined),
      () => dial()
    )
  }

  // The local connection keeps the BARE key (the bare-local rule of
  // lib/connection-scoped: byte-identical storage for single-backend users, and
  // the slug picked before per-connection keys existed survives the upgrade).
  // Remotes are suffixed by registry id.
  const slugStorageKey = () => {
    const scope = kanbanConnectionScope()

    return scope === LOCAL_SCOPE ? BOARD_SLUG_KEY : `${BOARD_SLUG_KEY}.${scope}`
  }

  $boardSlug.set(storage.get(slugStorageKey(), ''))
  unsubs.push($boardSlug.listen(slug => storage.set(slugStorageKey(), slug)))
  open($boardSlug.get())
  unsubs.push($boardSlug.listen(open))
  unsubs.push(
    host.state.connectionId.listen((next, prev) => {
      // Query keys embed the scope, so the new connection is already a cache
      // miss; only the LIVE bindings (socket, slug) follow it. The boot-time
      // null → 'local' publish is the same scope, not a switch. A changed slug
      // reopens the socket through the $boardSlug listener above; an unchanged
      // slug still needs a dial because the backend behind it changed.
      if ((next ?? LOCAL_SCOPE) === (prev ?? LOCAL_SCOPE)) {
        return
      }

      const previous = $boardSlug.get()
      $boardSlug.set(storage.get(slugStorageKey(), ''))

      if ($boardSlug.get() === previous) {
        open(previous)
      }
    })
  )

  return () => {
    dialGeneration += 1
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

// ── query keys (connection- and board-scoped; scope is always segment [2]) ────

/** Prefix matching every board query on one connection (all slugs, both
 *  archived views) — the mutation-settled invalidation target. */
export const boardKeyPrefix = (scope: string) => ['kanban', 'board', scope] as const
export const boardKey = (scope: string, slug: string, archived: boolean) =>
  [...boardKeyPrefix(scope), slug, archived] as const
export const taskKey = (scope: string, slug: string, id: string) => ['kanban', 'task', scope, slug, id] as const
export const logKey = (scope: string, slug: string, id: string) => ['kanban', 'log', scope, slug, id] as const
export const boardsKey = (scope: string) => ['kanban', 'boards', scope] as const
export const profilesKey = (scope: string) => ['kanban', 'profiles', scope] as const
export const projectsKey = (scope: string) => ['kanban', 'projects', scope] as const
export const orchestrationKey = (scope: string) => ['kanban', 'orchestration', scope] as const

// ── reads ─────────────────────────────────────────────────────────────────────

export const fetchBoard = (archived: boolean) =>
  call<KanbanBoard>(withBoard('/board', archived ? { include_archived: 'true' } : {}))

export const fetchTask = async (id: string) => {
  const downloadAttachment = captureGatewayFileDownload()
  const detail = await call<KanbanTaskDetail>(withBoard(`/tasks/${id}`))

  return { ...detail, downloadAttachment }
}

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
