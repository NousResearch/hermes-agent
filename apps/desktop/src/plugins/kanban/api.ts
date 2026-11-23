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
import type { QueryCacheNotifyEvent } from '@tanstack/react-query'

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
type Socket = (
  path: string,
  onMessage: (data: unknown) => void,
  onStatus?: (status: 'live' | 'polling') => void
) => () => void

let rest: null | Rest = null
let os: null | PluginOs = null

/** Selected board slug ('' = the server's current board). Persisted. */
export const $boardSlug = atom<string>('')

/** Pinned board slugs, in tab order, for the active connection. Persisted. */
export const $pinnedBoards = atom<string[]>([])

export function setBoardPinned(slug: string, pinned: boolean): void {
  const current = $pinnedBoards.get()

  if (current.includes(slug) !== pinned) {
    $pinnedBoards.set(pinned ? [...current, slug] : current.filter(item => item !== slug))
  }
}

/** Move `slug` into `target`'s position. */
export function movePinnedBoard(slug: string, target: string): void {
  const next = [...$pinnedBoards.get()]
  const from = next.indexOf(slug)
  const to = next.indexOf(target)

  if (from < 0 || to < 0 || from === to) {
    return
  }

  next.splice(from, 1)
  next.splice(to, 0, slug)
  $pinnedBoards.set(next)
}

/** Whether the "how this board works" intro was dismissed. Persisted. */
export const $introDismissed = atom<boolean>(false)

/** Sub-group the Running lane by assignee (the dashboard's "lanes by
 *  profile"). Persisted. */
export const $lanesByProfile = atom<boolean>(false)

/** Per-lane collapse OVERRIDES (true=collapsed, false=expanded). Absence means
 *  auto: empty lanes collapse to a rail, occupied lanes expand. Persisted. */
export const $collapsedLanes = atom<Record<string, boolean>>({})

// ── live transport state ────────────────────────────────────────────────────

/** Chip state for the board header: 'live' = events socket open AND frames
 *  arriving (server keepalive: one frame / 25 s); 'polling' = socket down or
 *  frames stalled — the HTTP fallback covers updates; 'offline' = nothing
 *  reached the backend recently (socket AND polls silent). */
export type KanbanLiveState = 'live' | 'polling' | 'offline'

/** Transport of the event stream, straight from pluginSocket's onStatus. */
export const $liveChannel = atom<'live' | 'polling'>('polling')

/** Derived display state — recomputed by the watchdog in `bindApi`. */
export const $liveState = atom<KanbanLiveState>('polling')

/** Wall clock (ms, null = nothing yet) of the last backend answer — an events
 *  frame OR a successful board read. Drives the "Synced <time>" caption and
 *  the offline watchdog. */
export const $lastSyncAt = atom<number | null>(null)

/** Board poll cadence: slow heartbeat while the event socket carries updates,
 *  fast fallback when it doesn't (a change shows within about 10 s on a
 *  degraded transport). */
export const LIVE_POLL_INTERVAL_MS = 60_000
export const POLL_FALLBACK_INTERVAL_MS = 10_000

/** Board-header poll interval derived from the live state. */
export const boardRefetchInterval = (state: KanbanLiveState): number =>
  state === 'live' ? LIVE_POLL_INTERVAL_MS : POLL_FALLBACK_INTERVAL_MS

/** Whether the board's fallback poll may keep running in a background window.
 *  Only while the event channel is not live, and never for a local gateway the
 *  Desktop itself started: its idle-exit would never trigger while a hidden
 *  window keeps asking it for the board. The browser-hosted Webapp and remote
 *  connections have no such gateway to wake, so they keep polling. */
export const pollInBackground = (state: KanbanLiveState, scope: string, browserHosted: boolean): boolean =>
  state !== 'live' && (browserHosted || scope !== LOCAL_SCOPE)

/** True in the browser-hosted Webapp (the host marker the browser bridge writes on the document). */
export const isBrowserHosted = (): boolean =>
  typeof document !== 'undefined' && document.documentElement.dataset.hermesDesktopHost === 'browser'

/** Server keepalive: one frame per 25 s when idle (`stream_events` in the
 *  plugin's `plugin_api.py`). Tolerate three missed keepalives before demoting 'live'. */
export const LIVE_KEEPALIVE_MS = 25_000
export const LIVE_TOLERATED_KEEPALIVES = 3
/** Offline verdict after this many missed fallback polls plus grace. While
 *  'live', the slow heartbeat (60 s) sets the bar instead of the 10 s one. */
export const OFFLINE_AFTER_POLLS = 3
export const OFFLINE_GRACE_MS = 5_000

/** Watchdog cadence — worst-case detection lag for a stalled frame clock. */
export const LIVE_TICK_MS = 5_000

/** Pure clock the watchdog runs every LIVE_TICK_MS — exported for tests.
 *  Boot (lastSyncAt null) claims nothing; evidence flips the chip. */
export function deriveLiveState(
  input: { channel: 'live' | 'polling'; lastFrameAt: number | null; lastSyncAt: number | null },
  now: number
): KanbanLiveState {
  const { channel, lastFrameAt, lastSyncAt } = input

  if (lastSyncAt === null) {
    return 'polling'
  }

  const offlineAfter =
    channel === 'live'
      ? LIVE_POLL_INTERVAL_MS + OFFLINE_GRACE_MS
      : POLL_FALLBACK_INTERVAL_MS * OFFLINE_AFTER_POLLS + OFFLINE_GRACE_MS

  if (now - lastSyncAt > offlineAfter) {
    return 'offline'
  }

  if (
    channel === 'live' &&
    lastFrameAt !== null &&
    now - lastFrameAt <= LIVE_KEEPALIVE_MS * LIVE_TOLERATED_KEEPALIVES
  ) {
    return 'live'
  }

  return 'polling'
}

/** Cache scope of the local pool — the SDK atom's own spelling. */
const LOCAL_SCOPE = 'local'

const KANBAN_KEY_ROOT = ['kanban'] as const

const BOARD_SLUG_KEY = 'boardSlug'
const PINNED_BOARDS_KEY = 'pinnedBoards'
const INTRO_KEY = 'introDismissed'
const LANES_KEY = 'lanesByProfile'
const COLLAPSED_KEY = 'collapsedLanes'

// Last frame cursor per (connection, board) this plugin bind. The socket
// reopens on every board switch and connection change; resuming from the last
// frame replays only what was missed. Keyed by connection so one gateway's
// cursor cannot resume another's stream. Cleared on bind/unbind — events that
// land while the plugin is unloaded are not replayed.
const eventCursorByBoard = new Map<string, number>()

function cursorKey(scope: string, slug: string): string {
  return `${scope}\0${slug}`
}

function snapshotCursor(scope: string, slug: string): number | undefined {
  for (const archived of [false, true]) {
    const board = queryClient.getQueryData<KanbanBoard>(boardKey(scope, slug, archived))

    if (typeof board?.latest_event_id === 'number') {
      return board.latest_event_id
    }
  }

  return undefined
}

/** Cursor a fresh socket starts from: the last frame this connection saw, else
 *  the cached board snapshot's tail. Undefined means nothing is known yet —
 *  fetch the snapshot before opening, never open at since=0. */
function eventsSince(scope: string, slug: string): number | undefined {
  const seen = eventCursorByBoard.get(cursorKey(scope, slug))

  if (typeof seen === 'number') {
    return seen
  }

  return snapshotCursor(scope, slug)
}

function eventsUrl(slug: string, since: number | undefined): string {
  const params = new URLSearchParams()

  if (slug) {
    params.set('board', slug)
  }

  if (since !== undefined) {
    params.set('since', String(since))
  }

  const query = params.toString()

  return query ? `/events?${query}` : '/events'
}

function boardSnapshotPath(slug: string): string {
  return slug ? `/board?board=${encodeURIComponent(slug)}` : '/board'
}

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

/** One live `task_events` frame → precise cache invalidation: the board, plus
 *  each touched task's detail. The polls (8s board / 4s drawer) stay as the
 *  fallback — the socket just makes the board feel instant. */
function onEventsFrame(scope: string, slug: string, data: unknown, selectedSlug = slug): void {
  const frame = data as { cursor?: unknown; events?: CompletionEvent[] }

  if (typeof frame?.cursor === 'number') {
    eventCursorByBoard.set(cursorKey(scope, slug), frame.cursor)
  }

  const events = frame?.events

  if (!events?.length) {
    return
  }

  void queryClient.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
  // Any event can change a board's card count — keep the switcher badge honest.
  void queryClient.invalidateQueries({ queryKey: boardsKey(scope) })

  for (const taskId of new Set(events.map(event => event.task_id).filter(Boolean))) {
    void queryClient.invalidateQueries({ queryKey: taskKey(scope, slug, taskId!) })

    if (selectedSlug !== slug) {
      void queryClient.invalidateQueries({ queryKey: taskKey(scope, selectedSlug, taskId!) })
    }
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

  // Kanban opts back INTO focus/reconnect refetches even though the app-wide
  // default disables them (lib/query-client.ts): hidden/minimized windows used
  // to pause even the fallback poll and nothing resynced on return.
  // 'always' = refetch even when data is fresh.
  queryClient.setQueryDefaults(KANBAN_KEY_ROOT, {
    enabled: routedToScope,
    refetchOnWindowFocus: 'always',
    refetchOnReconnect: 'always'
  })
  unsubs.push(() => queryClient.setQueryDefaults(KANBAN_KEY_ROOT, {}))

  // Hydrate an atom from storage and keep storage in sync with it.
  const persist = <T>(atom: Persisted<T>, key: string, fallback: T) => {
    atom.set(storage.get(key, fallback))
    unsubs.push(atom.listen(value => storage.set(key, value)))
  }

  persist($introDismissed, INTRO_KEY, false)
  persist($lanesByProfile, LANES_KEY, false)
  persist($collapsedLanes, COLLAPSED_KEY, {})

  eventCursorByBoard.clear()

  // ── live-status bookkeeping ────────────────────────────────────────────────
  // Freshness clock: every events frame (server keepalive 25 s) and every
  // successful board/boards cache answer proves the backend is reachable; the
  // watchdog derives the header chip state from channel + clock.
  let lastFrameAt: number | null = null

  const derive = () => {
    $liveState.set(
      deriveLiveState(
        { channel: $liveChannel.get(), lastFrameAt, lastSyncAt: $lastSyncAt.get() },
        Date.now()
      )
    )
  }

  const liveTimer = setInterval(derive, LIVE_TICK_MS)
  unsubs.push(() => clearInterval(liveTimer))

  const markFrame = () => {
    lastFrameAt = Date.now()
    $lastSyncAt.set(lastFrameAt)
    derive()
  }

  // Board/boards cache success — a poll answer, focus refetch or settled
  // mutation — proves reachability even when no socket frame arrived.
  unsubs.push(
    queryClient.getQueryCache().subscribe(
      (event: QueryCacheNotifyEvent) => {
        if (event.type !== 'updated' || event.action.type !== 'success') {
          return
        }

        const key = event.query.queryKey

        if (
          key[0] === 'kanban' &&
          (key[1] === 'board' || key[1] === 'boards') &&
          key[2] === kanbanConnectionScope()
        ) {
          $lastSyncAt.set(Date.now())
          derive()
        }
      }
    )
  )

  // Catch-up on window return / focus / network return. Kanban queries are
  // configured 'always' for focus/reconnect, but a covered window fires
  // visibilitychange without focus and a resumed machine can wake without
  // onlineManager noticing — an explicit invalidate is
  // idempotent and cheap; missed events ride the since-cursor on the next
  // socket frame.
  const resync = () => {
    const scope = kanbanConnectionScope()

    void queryClient.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
    void queryClient.invalidateQueries({ queryKey: boardsKey(scope) })
  }

  const visibilityHandler = () => {
    if (document.visibilityState === 'visible') {
      resync()
    }
  }

  document.addEventListener('visibilitychange', visibilityHandler)
  window.addEventListener('focus', resync)
  window.addEventListener('online', resync)
  unsubs.push(() => {
    document.removeEventListener('visibilitychange', visibilityHandler)
    window.removeEventListener('focus', resync)
    window.removeEventListener('online', resync)
  })

  let close: (() => void) | null = null
  let socketGeneration = 0

  const dial = (scope: string, slug: string, since: number | undefined) => {
    const generation = socketGeneration
    const selectedSlug = $boardSlug.get()

    return socket(
      eventsUrl(slug, since),
      data => {
        if (generation === socketGeneration) {
          onEventsFrame(scope, slug, data, selectedSlug)
          markFrame()
        }
      },
      status => {
        // Stale generations stay silent — only the current dial speaks for
        // the chip; open() resets to 'polling' across reconnects.
        if (generation === socketGeneration) {
          $liveChannel.set(status)

          // An open socket is itself proof of freshness: the server answered
          // the handshake. Counting only frames left the chip on 'polling' for
          // up to one keepalive (25 s) after the socket (re)opened.
          if (status === 'live') {
            lastFrameAt = Date.now()
            $lastSyncAt.set(lastFrameAt)
          }

          derive()
        }
      }
    )
  }

  const open = (selectedSlug: string) => {
    const generation = ++socketGeneration
    const scope = kanbanConnectionScope()

    close?.()
    close = null

    // The previous dial's status callbacks are generation-silenced from here
    // on — reset the chip so it cannot keep claiming 'live' for a channel the
    // new dial has to re-prove (board switch, connection switch, rebind).
    $liveChannel.set('polling')
    lastFrameAt = null
    derive()

    const openResolved = (slug: string) => {
      if (generation !== socketGeneration) {
        return
      }

      const since = eventsSince(scope, slug)

      if (since !== undefined) {
        close = dial(scope, slug, since)

        return
      }

      // No cached tail yet. Wait for the snapshot and open at its
      // latest_event_id. A board switch or unload bumps the generation so a
      // late snapshot cannot open a stale socket. A failed fetch still opens
      // with no since — the server starts at the tail rather than replaying.
      void queryClient
        .fetchQuery({
          queryFn: () => r<KanbanBoard>(boardSnapshotPath(slug)),
          queryKey: boardKey(scope, slug, false)
        })
        .then(board => {
          if (generation !== socketGeneration) {
            return
          }

          const tail = typeof board?.latest_event_id === 'number' ? board.latest_event_id : undefined

          close = dial(scope, slug, tail)
        })
        .catch(() => {
          if (generation !== socketGeneration) {
            return
          }

          close = dial(scope, slug, undefined)
        })
    }

    if (selectedSlug) {
      openResolved(selectedSlug)

      return
    }

    // Resolve the alias BEFORE the handshake: /events is pinned to its board
    // when opened. Resolving on each frame could classify an old socket's
    // events against a different server-current board's cursor.
    void r<BoardsResponse>('/boards')
      .then(boards => openResolved(typeof boards.current === 'string' ? boards.current : ''))
      .catch(() => openResolved('')) // Keep live cache invalidation; notifications fail closed.
  }

  // The local connection keeps the BARE key (the bare-local rule of
  // lib/connection-scoped: byte-identical storage for single-backend users, and
  // the slug picked before per-connection keys existed survives the upgrade).
  // Remotes are suffixed by registry id. Pins follow the same rule: boards
  // belong to a gateway, so each connection keeps its own tabs.
  const scopedKey = (key: string) => {
    const scope = kanbanConnectionScope()

    return scope === LOCAL_SCOPE ? key : `${key}.${scope}`
  }

  const slugStorageKey = () => scopedKey(BOARD_SLUG_KEY)

  const restorePins = () => {
    const stored = storage.get<unknown>(scopedKey(PINNED_BOARDS_KEY), [])
    const slugs = Array.isArray(stored) ? stored.filter((s): s is string => typeof s === 'string' && !!s) : []

    $pinnedBoards.set([...new Set(slugs)])
  }

  restorePins()
  unsubs.push($pinnedBoards.listen(pins => storage.set(scopedKey(PINNED_BOARDS_KEY), pins)))
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

      restorePins()
      const previous = $boardSlug.get()
      $boardSlug.set(storage.get(slugStorageKey(), ''))

      if ($boardSlug.get() === previous) {
        open(previous)
      }
    })
  )

  return () => {
    socketGeneration += 1
    eventCursorByBoard.clear()
    $liveChannel.set('polling')
    $liveState.set('polling')
    $lastSyncAt.set(null)
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
