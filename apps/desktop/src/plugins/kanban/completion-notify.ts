/**
 * Native kanban terminal-event notification (completion, blocker, failure).
 *
 * No maintained exact-fit OSS exists and the SDK
 * has no kanban event door, so this module rides the kanban plugin's EXISTING
 * /events socket (api.ts onEventsFrame). No new WebSocket, no new process,
 * no DB, no auth, no persistence — cursor is an in-memory per-board high-water
 * mark. Notifies on the same terminal kinds the gateway watcher pings
 * (gateway/kanban_watchers.py): 'completed' (kanban_db.complete_task —
 * payload: summary + artifacts), 'blocked' (payload: reason), 'gave_up'
 * (payload: error), 'crashed', 'timed_out', and 'block_loop_detected'
 * (payload: reason — the routed-to-triage orchestration handoff).
 *
 * Two delivery doors, complementary by design:
 *  - `host.notify` — the in-app toast, covers the foreground case;
 *  - `ctx.os.notify` (when bound) — the native OS notification, which the
 *    desktop shell fires only while the user is AWAY from Hermes. This is the
 *    door that covers "walked away and the worker hit a blocker".
 *
 * Cursor contract: first observation of a board baselines
 * seen[board] = GET /board latest_event_id (MAX task_events.id for that
 * board). Events id <= seen are historical/replay — never notified, no
 * cursor change. id > seen advances cursor for EVERY kind; only terminal
 * kinds emit. Reconnect replays from 0; cursor filters. Board switch never
 * mixes cursors; returning reuses prior cursor (never reset to current MAX).
 * Fail-closed: while a board's baseline is unknown, no event can be
 * classified so none is notified. Empty slug ('') suppressed.
 *
 * Every map is keyed by (connection scope, board slug) — `cursorKey`, the
 * same key api.ts's socket cursor uses — so two gateways that share a slug
 * never classify each other's events.
 *
 * Alerts mode (#123596, ./alerts-mode): an emitted event is delivered per
 * `MODE_DELIVERY` (toast = today, quiet = bottom-right + chime, badge =
 * silent), and in EVERY mode counts toward `$unseenByBoard` unless the user
 * is looking at that board (page mounted AND window visible). Counting sits
 * behind the cursor, so replayed/historical events never count.
 */

import {
  atom,
  computed,
  host,
  playCompletionSound,
  type PluginOs,
  type PluginRestOptions,
  type PluginTranslate
} from '@hermes/plugin-sdk'

import { $alertsMode, type KanbanAlertsMode } from './alerts-mode'
import { en } from './i18n'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

export interface CompletionEvent {
  id?: unknown
  task_id?: string
  kind?: string
  payload?: Record<string, unknown> | null
}

type ToastKind = 'error' | 'success' | 'warning'

/** Terminal kinds → toast severity + i18n title key. Mirrors the gateway
 *  watcher's ping set (gateway/kanban_watchers.py) minus the intentionally
 *  silent kinds (status/archived/unblocked, which only advance the cursor). */
const TERMINAL_NOTIFY = new Map<string, { titleKey: string; toast: ToastKind }>([
  ['blocked', { titleKey: 'notify.blockedTitle', toast: 'warning' }],
  ['block_loop_detected', { titleKey: 'notify.blockLoopTitle', toast: 'warning' }],
  ['completed', { titleKey: 'notify.completedTitle', toast: 'success' }],
  ['crashed', { titleKey: 'notify.crashedTitle', toast: 'error' }],
  ['gave_up', { titleKey: 'notify.gaveUpTitle', toast: 'error' }],
  ['timed_out', { titleKey: 'notify.timedOutTitle', toast: 'warning' }]
])

const seenEventIdByBoard = new Map<string, number>()
const baselinePending = new Set<string>()

let rest: Rest | null = null
let translate: PluginTranslate | null = null
let osDoor: PluginOs | null = null

/** (connection scope, board slug) → one map key. api.ts keys its socket
 *  cursor with the same function. */
export function cursorKey(scope: string, slug: string): string {
  return `${scope}\0${slug}`
}

/** Unseen emitted terminal events per `cursorKey`. In memory only. */
export const $unseenByBoard = atom<Record<string, number>>({})

/** The nav-row total: unseen events on the ACTIVE connection's boards. */
export const $kanbanUnseen = computed([$unseenByBoard, host.state.connectionId], (unseen, connectionId) => {
  const prefix = cursorKey(connectionId ?? 'local', '')
  let total = 0

  for (const [key, n] of Object.entries(unseen)) {
    if (key.startsWith(prefix)) {
      total += n
    }
  }

  return total
})

/** Live board-page mounts per `cursorKey`. A count, not a slot: the same
 *  board can be mounted twice (main route + split tile). */
const viewing = new Map<string, number>()
/** Bumped by `resetCompletionNotify` so a frame awaiting its baseline across a
 *  plugin unload cannot write counts, cursors or toasts afterwards. */
let generation = 0
let detachVisibility: (() => void) | null = null

const documentVisible = () => typeof document !== 'undefined' && document.visibilityState === 'visible'

/** The user is looking at this board: its page is mounted AND the window is visible. */
const looking = (key: string) => viewing.has(key) && documentVisible()

function clearUnseen(key: string): void {
  const current = $unseenByBoard.get()

  if (current[key]) {
    const { [key]: _cleared, ...remaining } = current
    $unseenByBoard.set(remaining)
  }
}

/** Called by the board page while mounted: clears that board's count (if the
 *  window is visible) and suppresses counting while the user looks at it.
 *  Returns the unmark, which releases only this mark and only once. */
export function markBoardViewing(scope: string, slug: string): () => void {
  const key = cursorKey(scope, slug)
  let released = false
  viewing.set(key, (viewing.get(key) ?? 0) + 1)

  if (documentVisible()) {
    clearUnseen(key)
  }

  return () => {
    if (released) {
      return
    }

    released = true
    const left = (viewing.get(key) ?? 1) - 1

    if (left > 0) {
      viewing.set(key, left)
    } else {
      viewing.delete(key)
    }
  }
}

/** Plugin unload: forget every cursor, count and viewing mark. */
export function resetCompletionNotify(): void {
  generation += 1
  seenEventIdByBoard.clear()
  baselinePending.clear()
  $unseenByBoard.set({})
  viewing.clear()
  detachVisibility?.()
  detachVisibility = null
}

/** Resolve a dot-path against the plugin's own English bundle — the same
 *  last-rung fallback the plugin i18n registry applies, usable before (or
 *  without) a bound translator. */
function fallbackT(key: string, ...args: unknown[]): string {
  let node: unknown = en

  for (const part of key.split('.')) {
    node = (node as Record<string, unknown> | undefined)?.[part]
  }

  if (typeof node === 'function') {
    return (node as (...a: unknown[]) => string)(...args)
  }

  return typeof node === 'string' ? node : key
}

function t(key: string, ...args: unknown[]): string {
  const translated = translate?.(key, ...args)

  // The registry returns the raw key when the bundle isn't registered yet.
  return translated && translated !== key ? translated : fallbackT(key, ...args)
}

export function bindCompletionNotify(r: Rest, pluginTranslate?: PluginTranslate, os?: PluginOs): void {
  rest = r
  translate = pluginTranslate ?? null
  osDoor = os ?? null

  // Coming back to a window left on a board page = looking at it again.
  detachVisibility?.()
  detachVisibility = null

  if (typeof document !== 'undefined') {
    const onVisibility = () => {
      if (documentVisible()) {
        viewing.forEach((_mounts, key) => clearUnseen(key))
      }
    }

    document.addEventListener('visibilitychange', onVisibility)
    detachVisibility = () => document.removeEventListener('visibilitychange', onVisibility)
  }
}

async function ensureBaseline(key: string, slug: string): Promise<void> {
  if (seenEventIdByBoard.has(key) || baselinePending.has(key)) {
    return
  }

  const bound = generation
  baselinePending.add(key)

  try {
    const board = (await rest!<{ latest_event_id?: unknown }>(`/board?board=${encodeURIComponent(slug)}`)) as {
      latest_event_id?: unknown
    }

    if (bound === generation) {
      seenEventIdByBoard.set(key, typeof board.latest_event_id === 'number' ? board.latest_event_id : 0)
    }
  } catch {
    // Fail-closed: unknown baseline → notifications stay suppressed.
  } finally {
    if (bound === generation) {
      baselinePending.delete(key)
    }
  }
}

function trimmed(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/** The human handoff carried in the event payload, per kind (mirrors the
 *  payload contract the gateway watcher reads). `gave_up` deliberately has no
 *  payload body: its `error` is raw worker text, which belongs in the toast
 *  `detail` (see rawErrorFor), and the body is the plain-words i18n hint. */
function bodyFor(kind: string, ev: CompletionEvent): string {
  const payload = ev.payload

  if (kind === 'completed') {
    return trimmed(payload?.summary)
  }

  if (kind === 'blocked' || kind === 'block_loop_detected') {
    return trimmed(payload?.reason)
  }

  if (kind === 'gave_up') {
    return t('notify.gaveUpBody')
  }

  return ''
}

/** Raw machine text that must never be the toast body — surfaced muted in `detail`. */
function rawErrorFor(kind: string, ev: CompletionEvent): string {
  return kind === 'gave_up' ? trimmed(ev.payload?.error) : ''
}

type Spec = { titleKey: string; toast: ToastKind }

/** `quiet` mode's toast: the quiet corner, and never sticky (warning/error
 *  default to sticky; an explicit duration overrides that). */
const QUIET_TOAST = { placement: 'bottom-right', durationMs: 5000 } as const

function notifyOne(
  kind: string,
  spec: Spec,
  ev: CompletionEvent,
  toastOverrides: Partial<typeof QUIET_TOAST> = {}
): void {
  const taskId = (ev.task_id ?? '').trim()
  const body = bodyFor(kind, ev)

  const artifacts =
    kind === 'completed' && Array.isArray(ev.payload?.artifacts)
      ? (ev.payload!.artifacts as unknown[])
          .filter((a): a is string => typeof a === 'string' && a.trim().length > 0)
          .map(a => a.trim())
      : []

  const artifactText =
    artifacts.length === 1
      ? artifacts[0].split(/[\\/]/).pop() || artifacts[0]
      : artifacts.length > 1
        ? t('notify.artifacts', artifacts.length)
        : ''

  const detail = [taskId, artifactText, rawErrorFor(kind, ev)].filter(Boolean).join(' · ')
  const title = t(spec.titleKey)
  const message = body || taskId || title
  host.notify({
    kind: spec.toast,
    title,
    message,
    ...(detail ? { detail } : {}),
    action: { label: t('notify.openKanban'), onClick: () => host.navigate('/kanban') },
    ...toastOverrides
  })

  // Native OS notification — the desktop shell fires it only while the user
  // is away from Hermes (the toast above covers the foreground case). Isolated:
  // a missing/broken shell must not mark the toast as unfired.
  try {
    osDoor?.notify({ title, body: [message, detail].filter(Boolean).join('\n') })
  } catch {
    /* swallowed */
  }
}

/** Per-event delivery by alerts mode. `toast` is today's call unchanged; the
 *  quiet chime is once per FRAME, so it lives after the loop, not here. */
const MODE_DELIVERY: Record<KanbanAlertsMode, (kind: string, spec: Spec, ev: CompletionEvent) => void> = {
  toast: (kind, spec, ev) => notifyOne(kind, spec, ev),
  quiet: (kind, spec, ev) => notifyOne(kind, spec, ev, QUIET_TOAST),
  badge: () => undefined
}

/** Consume one /events frame for a board on connection `scope`. Returns true
 *  when a terminal event was emitted (delivered per the alerts mode — in
 *  `badge` mode that is the count alone). Never throws: notification failure
 *  cannot interfere with api.ts cache invalidation. */
export async function onKanbanEventsFrame(slug: string, events?: CompletionEvent[], scope = 'local'): Promise<boolean> {
  if (!events?.length || slug === '' || !rest) {
    return false
  }

  const key = cursorKey(scope, slug)
  const bound = generation
  await ensureBaseline(key, slug)
  const seen = seenEventIdByBoard.get(key)

  if (seen === undefined || bound !== generation) {
    return false
  } // fail-closed

  const mode = $alertsMode.get()
  let emitted = false
  let lastEmittedId = 0
  let cursor = seen

  for (const ev of events) {
    if (typeof ev.id !== 'number' || ev.id <= cursor) {
      continue
    }

    cursor = ev.id
    seenEventIdByBoard.set(key, cursor)
    const spec = TERMINAL_NOTIFY.get(ev.kind ?? '')

    if (spec) {
      if (!looking(key)) {
        const unseen = $unseenByBoard.get()
        $unseenByBoard.set({ ...unseen, [key]: (unseen[key] ?? 0) + 1 })
      }

      try {
        MODE_DELIVERY[mode](ev.kind!, spec, ev)
        emitted = true
        lastEmittedId = ev.id
      } catch {
        /* swallowed */
      }
    }
  }

  if (mode === 'quiet' && emitted && !looking(key)) {
    try {
      playCompletionSound(`kanban:${scope}:${slug}:${lastEmittedId}`)
    } catch {
      /* swallowed */
    }
  }

  return emitted
}
