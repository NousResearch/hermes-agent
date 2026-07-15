/**
 * Gateway-backed turn queue store for the desktop renderer.
 *
 * The queue lives on the agent (`agent.turn_queue`) in the gateway process —
 * this store is a mirror, populated from `queue.updated` events and the
 * `queue` field of `session.info`. The renderer never owns the queue: the
 * gateway drains it at the end of every turn (and immediately on enqueue when
 * the session is idle), so queued messages fire even when the session tab is
 * closed or the window is hidden. There is NO client-side auto-drain.
 *
 * Mutations are optimistic: the local mirror updates immediately for snappy
 * UI, then the RPC fires and the authoritative `queue.updated` event settles
 * the final state.
 */

import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  queuedAt: number
}

/** A steer accepted by the gateway but not yet injected into the live turn. */
export interface PendingSteerEntry {
  id: string
  text: string
  steeredAt: number
}

type QueueState = Record<string, QueuedPromptEntry[]>
type SteerState = Record<string, PendingSteerEntry[]>

export const $queuedPromptsBySession = atom<QueueState>({})
export const $pendingSteersBySession = atom<SteerState>({})

// Module-level gateway request function — set once by the gateway boot hook so
// this store can fire RPCs without being a React component.
let _callGateway: GatewayRequester | null = null

export const setGatewayRequester = (fn: GatewayRequester | null) => {
  _callGateway = fn
}

const callGateway = async <T = unknown>(method: string, params?: Record<string, unknown>): Promise<T | null> => {
  if (!_callGateway) {
    return null
  }

  try {
    return await _callGateway<T>(method, params)
  } catch {
    return null
  }
}

const sidOf = (key: string | null | undefined): null | string => {
  const trimmed = key?.trim()

  return trimmed ? trimmed : null
}

const queueFor = (sid: string) => $queuedPromptsBySession.get()[sid] ?? []

const writeSession = (sid: string, queue: QueuedPromptEntry[]) => {
  const next = { ...$queuedPromptsBySession.get() }

  if (queue.length === 0) {
    delete next[sid]
  } else {
    next[sid] = queue
  }

  $queuedPromptsBySession.set(next)
}

interface GatewayQueueEntry {
  id?: string
  text?: string
  queued_at?: number
}

/** Replace a session's mirror with the gateway's authoritative entry list
 *  (called from `queue.updated` events and `session.info.queue`). */
export const setSessionQueue = (key: string | null | undefined, entries: unknown[]) => {
  const sid = sidOf(key)

  if (!sid) {
    return
  }

  writeSession(
    sid,
    entries
      .filter((e): e is GatewayQueueEntry => Boolean(e) && typeof e === 'object')
      .map(e => ({
        id: String(e.id ?? ''),
        text: String(e.text ?? ''),
        attachments: [],
        queuedAt: typeof e.queued_at === 'number' ? Math.round(e.queued_at * 1000) : 0
      }))
  )
}

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const sid = sidOf(key)

  return sid ? queueFor(sid) : []
}

// ── RPC-backed mutations ─────────────────────────────────────────────
// Each mutation updates the local mirror optimistically, then fires the RPC;
// the gateway's queue.updated event is the authoritative settle.

/**
 * Queue a prompt on the gateway. The gateway drains it as the next turn (or
 * immediately when the session is idle). Returns the created entry on accept,
 * null when the gateway is unreachable — the caller keeps the draft so no
 * words are lost.
 */
export const enqueueQueuedPrompt = async (
  key: string | null | undefined,
  payload: { text: string }
): Promise<null | QueuedPromptEntry> => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const result = await callGateway<{ entry_id?: string; status?: string }>('session.queue.add', {
    session_id: sid,
    text: payload.text
  })

  if (result?.status !== 'queued') {
    return null
  }

  const entry: QueuedPromptEntry = {
    id: result.entry_id ?? `local-${Date.now()}`,
    text: payload.text,
    attachments: [],
    queuedAt: Date.now()
  }

  // Optimistic append — queue.updated settles the authoritative list.
  writeSession(sid, [...queueFor(sid).filter(e => e.id !== entry.id), entry])

  return entry
}

export const removeQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const next = queue.filter(e => e.id !== id)

  if (next.length === queue.length) {
    return false
  }

  writeSession(sid, next)
  void callGateway('session.queue.remove', { session_id: sid, entry_id: id })

  return true
}

/**
 * Move an entry to the head so it fires next. With `interrupt`, also winds
 * down the live turn (queue preserved) so the entry fires as soon as the turn
 * settles — the "send now" gesture.
 */
export const promoteQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  options?: { interrupt?: boolean }
): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index < 0) {
    return false
  }

  if (index > 0) {
    const entry = queue[index]!
    writeSession(sid, [entry, ...queue.slice(0, index), ...queue.slice(index + 1)])
  }

  void callGateway('session.queue.promote', {
    session_id: sid,
    entry_id: id,
    interrupt: Boolean(options?.interrupt)
  })

  return true
}

export const updateQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  update: { text: string; attachments?: ComposerAttachment[] }
): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  let changed = false

  const next = queue.map(entry => {
    if (entry.id !== id || entry.text === update.text) {
      return entry
    }

    changed = true

    return { ...entry, text: update.text }
  })

  if (!changed) {
    return false
  }

  writeSession(sid, next)
  void callGateway('session.queue.update', { session_id: sid, entry_id: id, text: update.text })

  return true
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

/** Clear the local mirror and the gateway queue (session close/delete). */
export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (!sid) {
    return
  }

  if (sid in $queuedPromptsBySession.get()) {
    writeSession(sid, [])
  }

  clearPendingSteers(sid)
  void callGateway('session.queue.clear', { session_id: sid })
}

/** Local-only mirror re-key when a backend bounce mints a fresh runtime id
 *  for the same conversation. The gateway re-syncs via session.info/queue
 *  events; this just keeps the panel from flashing empty in between. */
export const migrateQueuedPrompts = (fromKey: string | null | undefined, toKey: string | null | undefined): boolean => {
  const from = sidOf(fromKey)
  const to = sidOf(toKey)

  if (!from || !to || from === to) {
    return false
  }

  const pending = queueFor(from)

  if (pending.length === 0) {
    return false
  }

  const next = { ...$queuedPromptsBySession.get() }
  delete next[from]
  next[to] = [...queueFor(to), ...pending]

  $queuedPromptsBySession.set(next)

  return true
}

// ── Pending steers ───────────────────────────────────────────────────
// A steer RPC is accepted instantly, but the text only reaches the model at
// the next tool-batch boundary. These helpers track that in-between state so
// the transcript can show the steer when it actually lands (steer.applied)
// instead of pretending it was instant.

const steersFor = (sid: string) => $pendingSteersBySession.get()[sid] ?? []

const writeSteers = (sid: string, steers: PendingSteerEntry[]) => {
  const next = { ...$pendingSteersBySession.get() }

  if (steers.length === 0) {
    delete next[sid]
  } else {
    next[sid] = steers
  }

  $pendingSteersBySession.set(next)
}

export const getPendingSteers = (key: string | null | undefined): PendingSteerEntry[] => {
  const sid = sidOf(key)

  return sid ? steersFor(sid) : []
}

export const addPendingSteer = (key: string | null | undefined, text: string) => {
  const sid = sidOf(key)

  if (!sid) {
    return
  }

  writeSteers(sid, [
    ...steersFor(sid),
    { id: `steer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, text, steeredAt: Date.now() }
  ])
}

/** Settle pending steers covered by an applied/dropped gateway event. The
 *  agent concatenates queued-up steers with newlines before injecting, so one
 *  event can cover several pending entries — match by containment. */
export const settlePendingSteer = (key: string | null | undefined, text: string) => {
  const sid = sidOf(key)

  if (!sid) {
    return
  }

  const settled = new Set(
    text
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
  )

  writeSteers(
    sid,
    steersFor(sid).filter(entry => !settled.has(entry.text.trim()))
  )
}

export const clearPendingSteers = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (sid && sid in $pendingSteersBySession.get()) {
    writeSteers(sid, [])
  }
}
