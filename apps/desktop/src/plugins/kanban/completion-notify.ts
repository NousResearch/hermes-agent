/**
 * Native kanban task-completion notification.
 *
 * No maintained exact-fit OSS exists and the SDK
 * has no kanban event door, so this module rides the kanban plugin's EXISTING
 * /events socket (api.ts onEventsFrame). No new WebSocket, no new process,
 * no DB, no auth, no persistence — cursor is an in-memory per-board high-water
 * mark. Reuses native completion finality: kind == 'completed' events written
 * by kanban_db.complete_task (payload: summary + artifacts).
 *
 * Cursor contract: first observation of a board baselines
 * seen[board] = GET /board latest_event_id (MAX task_events.id for that
 * board). Events id <= seen are historical/replay — never notified, no
 * cursor change. id > seen advances cursor for EVERY kind; only 'completed'
 * emits. Reconnect replays from 0; cursor filters. Board switch never mixes
 * cursors; returning reuses prior cursor (never reset to current MAX).
 * Fail-closed: while a board's baseline is unknown, no event can be
 * classified so none is notified. Empty slug ('') suppressed.
 */

import { host, type PluginRestOptions } from '@hermes/plugin-sdk'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

export interface CompletionEvent {
  id?: unknown
  task_id?: string
  kind?: string
  payload?: Record<string, unknown> | null
}

const seenEventIdByBoard = new Map<string, number>()
const baselinePending = new Set<string>()

let rest: Rest | null = null

export function bindCompletionNotify(r: Rest): void {
  rest = r
}

async function ensureBaseline(slug: string): Promise<void> {
  if (seenEventIdByBoard.has(slug) || baselinePending.has(slug)) { return }
  baselinePending.add(slug)

  try {
    const board = (await rest!<{ latest_event_id?: unknown }>(`/board?board=${encodeURIComponent(slug)}`)) as { latest_event_id?: unknown }
    seenEventIdByBoard.set(slug, typeof board.latest_event_id === 'number' ? board.latest_event_id : 0)
  } catch {
    // Fail-closed: unknown baseline → notifications stay suppressed.
  } finally {
    baselinePending.delete(slug)
  }
}

function notifyOne(ev: CompletionEvent): void {
  const taskId = (ev.task_id ?? '').trim()
  const summary = typeof ev.payload?.summary === 'string' ? ev.payload.summary.trim() : ''

  const artifacts = Array.isArray(ev.payload?.artifacts)
    ? (ev.payload!.artifacts as unknown[]).filter((a): a is string => typeof a === 'string' && a.trim().length > 0).map(a => a.trim())
    : []

  const artifactText = artifacts.length === 1 ? (artifacts[0].split(/[\\/]/).pop() || artifacts[0]) : artifacts.length > 1 ? `${artifacts.length} artifacts` : ''
  const detail = [taskId, artifactText].filter(Boolean).join(' · ')
  host.notify({ kind: 'success', title: 'Task completed', message: summary || taskId || 'Task completed', ...(detail ? { detail } : {}), action: { label: 'Open Kanban', onClick: () => host.navigate('/kanban') } })
}

/** Consume one /events frame for a board. Returns true when a completion
 *  notification was fired. Never throws: notification failure cannot
 *  interfere with api.ts cache invalidation. */
export async function onKanbanEventsFrame(slug: string, events?: CompletionEvent[]): Promise<boolean> {
  if (!events?.length || slug === '' || !rest) { return false }
  await ensureBaseline(slug)
  const seen = seenEventIdByBoard.get(slug)

  if (seen === undefined) { return false } // fail-closed
  let fired = false
  let cursor = seen

  for (const ev of events) {
    if (typeof ev.id !== 'number' || ev.id <= cursor) { continue }
    cursor = ev.id
    seenEventIdByBoard.set(slug, cursor)

    if (ev.kind === 'completed') { try { notifyOne(ev); fired = true } catch { /* swallowed */ } }
  }

  return fired
}
