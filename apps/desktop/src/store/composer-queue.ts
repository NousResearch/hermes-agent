import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  queuedAt: number
}

type QueueState = Record<string, QueuedPromptEntry[]>

const STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

const load = (): QueueState => {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : null

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as QueueState) : {}
  } catch {
    return {}
  }
}

const save = (state: QueueState) => {
  if (typeof window === 'undefined') {
    return
  }

  try {
    if (Object.keys(state).length === 0) {
      window.localStorage.removeItem(STORAGE_KEY)
    } else {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    }
  } catch {
    // best-effort: storage may be unavailable, queue still works in-memory
  }
}

export const $queuedPromptsBySession = atom<QueueState>(load())

/**
 * Sessions whose queue the user explicitly halted (Stop button / Esc). A parked
 * queue is skipped by both auto-drain paths until the user acts on it again —
 * resume, send-now, a manual drain, queueing a fresh prompt, or emptying the
 * queue all unpark. Deliberately in-memory only: a fresh app process starts
 * unparked, so restored-entry semantics stay a separate concern.
 */
export const $parkedQueueSessions = atom<Record<string, true>>({})

const setParked = (sid: string, parked: boolean) => {
  const current = $parkedQueueSessions.get()

  if (Boolean(current[sid]) === parked) {
    return
  }

  const next = { ...current }

  if (parked) {
    next[sid] = true
  } else {
    delete next[sid]
  }

  $parkedQueueSessions.set(next)
}

const writeSession = (sid: string, queue: QueuedPromptEntry[]) => {
  const current = $queuedPromptsBySession.get()
  const next = { ...current }

  if (queue.length === 0) {
    delete next[sid]
    // An empty queue has nothing to hold back — drop the park so it can't
    // linger as stale state and silently gate entries queued much later.
    setParked(sid, false)
  } else {
    next[sid] = queue
  }

  $queuedPromptsBySession.set(next)
  save(next)
}

const sidOf = (key: string | null | undefined): null | string => {
  const trimmed = key?.trim()

  return trimmed ? trimmed : null
}

const queueFor = (sid: string) => $queuedPromptsBySession.get()[sid] ?? []

const nextId = () => `queued-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const sid = sidOf(key)

  return sid ? queueFor(sid) : []
}

export const enqueueQueuedPrompt = (
  key: string | null | undefined,
  payload: { text: string; attachments: ComposerAttachment[] }
): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const entry: QueuedPromptEntry = {
    id: nextId(),
    text: payload.text,
    attachments: cloneAttachments(payload.attachments),
    queuedAt: Date.now()
  }

  writeSession(sid, [...queueFor(sid), entry])
  // Queueing a new prompt is fresh intent to keep the conversation moving —
  // a park from an earlier Stop must not hold this (or the entries ahead of
  // it) back.
  setParked(sid, false)

  return entry
}

export const dequeueQueuedPrompt = (key: string | null | undefined): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const [head, ...rest] = queueFor(sid)

  if (!head) {
    return null
  }

  writeSession(sid, rest)

  return head
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

  return true
}

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index <= 0) {
    return false
  }

  const entry = queue[index]!
  writeSession(sid, [entry, ...queue.slice(0, index), ...queue.slice(index + 1)])

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
    if (entry.id !== id) {
      return entry
    }

    const attachments = update.attachments ? cloneAttachments(update.attachments) : entry.attachments

    if (entry.text === update.text && !update.attachments) {
      return entry
    }

    changed = true

    return { ...entry, text: update.text, attachments }
  })

  if (!changed) {
    return false
  }

  writeSession(sid, next)

  return true
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (!sid || !(sid in $queuedPromptsBySession.get())) {
    return
  }

  writeSession(sid, [])
}

/**
 * Move pending entries from a dead session key onto a live one, preserving FIFO
 * (existing target entries first, migrated entries appended). A backend bounce /
 * resume can mint a fresh runtime session id for the *same* conversation; the
 * entries enqueued under the old id would otherwise be stranded under a key
 * nothing reads anymore. No-op unless both keys resolve and differ.
 */
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
  save(next)

  // The park is a property of the entries the user halted — it re-homes with
  // them. Without this, a backend bounce right after Stop would shed the park
  // and auto-send the exact prompts the user just held back.
  if ($parkedQueueSessions.get()[from]) {
    setParked(from, false)
    setParked(to, true)
  }

  return true
}

/**
 * Park a session's queue after an explicit user halt (Stop / Esc): entries stay
 * visible in the panel but neither auto-drain path sends them. No-op for a
 * session with nothing queued — parking exists to hold back queued turns, and
 * a park with no queue would only linger as a stale gate.
 */
export const parkQueuedPrompts = (key: string | null | undefined): boolean => {
  const sid = sidOf(key)

  if (!sid || queueFor(sid).length === 0) {
    return false
  }

  setParked(sid, true)

  return true
}

/** Lift a park (user resumed the queue). Safe to call for any session. */
export const unparkQueuedPrompts = (key: string | null | undefined): void => {
  const sid = sidOf(key)

  if (sid) {
    setParked(sid, false)
  }
}

export const isQueueParked = (key: string | null | undefined): boolean => {
  const sid = sidOf(key)

  return sid ? Boolean($parkedQueueSessions.get()[sid]) : false
}

/** Inputs to {@link shouldAutoDrain}. */
export interface AutoDrainInput {
  isBusy: boolean
  /** The user explicitly halted this session's queue (Stop / Esc). */
  parked?: boolean
  queueLength: number
}

/**
 * Decide whether the composer should auto-drain the next queued prompt.
 *
 * Edge-independent on purpose: the queue must advance whenever the session is
 * idle and has pending entries, NOT only on an observed busy true → false edge.
 * A backend bounce / websocket reconnect remounts the composer and resets the
 * busy ref to the current value, swallowing the settle edge — an edge-gated
 * drain would then strand the entry forever. The caller's drain lock
 * (`drainingQueueRef`) serializes sends so being edge-free can't double-submit.
 *
 * `parked` is the one deliberate exception: an explicit Stop/Esc is the user
 * saying HALT, and immediately firing the next queued prompt contradicts the
 * instruction they just gave. Parked entries stay in the panel until the user
 * resumes, sends, edits, or deletes them. Interrupts that exist to reach the
 * queue faster (send-now-while-busy) never park, so they keep draining through
 * this same gate.
 */
export const shouldAutoDrain = ({ isBusy, parked, queueLength }: AutoDrainInput): boolean =>
  !isBusy && !parked && queueLength > 0

/** Auto-drain attempts for one entry before we stop retrying and toast. The
 * entry stays queued for a manual send; a remount/reconnect resets the count. */
export const MAX_AUTO_DRAIN_ATTEMPTS = 4
