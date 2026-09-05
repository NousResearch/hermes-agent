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
 * (payload: reason — the routed-to-triage human handoff).
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
 */

import { host, type PluginOs, type PluginRestOptions, type PluginTranslate } from '@hermes/plugin-sdk'

import { en } from './i18n'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

export interface CompletionEvent {
  id?: unknown
  task_id?: string
  kind?: string
  payload?: Record<string, unknown> | null
  created_at?: unknown
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
}

async function ensureBaseline(slug: string): Promise<void> {
  if (seenEventIdByBoard.has(slug) || baselinePending.has(slug)) {
    return
  }

  baselinePending.add(slug)

  try {
    const board = (await rest!<{ latest_event_id?: unknown }>(`/board?board=${encodeURIComponent(slug)}`)) as {
      latest_event_id?: unknown
    }

    seenEventIdByBoard.set(slug, typeof board.latest_event_id === 'number' ? board.latest_event_id : 0)
  } catch {
    // Fail-closed: unknown baseline → notifications stay suppressed.
  } finally {
    baselinePending.delete(slug)
  }
}

function trimmed(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/** The human handoff carried in the event payload, per kind (mirrors the
 *  payload contract the gateway watcher reads). */
function bodyFor(kind: string, ev: CompletionEvent): string {
  const payload = ev.payload

  if (kind === 'completed') {
    return trimmed(payload?.summary)
  }

  if (kind === 'blocked' || kind === 'block_loop_detected') {
    return trimmed(payload?.reason)
  }

  if (kind === 'gave_up') {
    return trimmed(payload?.error)
  }

  return ''
}

function notifyOne(kind: string, spec: { titleKey: string; toast: ToastKind }, ev: CompletionEvent): void {
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

  const detail = [taskId, artifactText].filter(Boolean).join(' · ')
  const title = t(spec.titleKey)
  const message = body || taskId || title
  host.notify({
    kind: spec.toast,
    title,
    message,
    ...(detail ? { detail } : {}),
    action: { label: t('notify.openKanban'), onClick: () => host.navigate('/kanban') }
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

/** Consume one /events frame for a board. Returns true when a terminal-event
 *  notification was fired. Never throws: notification failure cannot
 *  interfere with api.ts cache invalidation. */
export async function onKanbanEventsFrame(slug: string, events?: CompletionEvent[]): Promise<boolean> {
  if (!events?.length || slug === '' || !rest) {
    return false
  }

  await ensureBaseline(slug)
  const seen = seenEventIdByBoard.get(slug)

  if (seen === undefined) {
    return false
  } // fail-closed

  let fired = false
  let cursor = seen

  for (const ev of events) {
    if (typeof ev.id !== 'number' || ev.id <= cursor) {
      continue
    }

    cursor = ev.id
    seenEventIdByBoard.set(slug, cursor)
    const spec = TERMINAL_NOTIFY.get(ev.kind ?? '')

    if (spec) {
      try {
        notifyOne(ev.kind!, spec, ev)
        fired = true
      } catch {
        /* swallowed */
      }
    }
  }

  return fired
}
