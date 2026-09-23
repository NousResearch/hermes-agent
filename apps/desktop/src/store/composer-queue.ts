import { SLASH_COMMAND_RE } from '@hermes/shared'
import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export interface QueuedPromptEntry {
  id: string
  text: string
  /** What the queue panel and the sent bubble show, when it differs from the
   *  text the agent receives. A queued `/skill` invocation carries the whole
   *  expanded skill body as `text` — the UI shows the invocation instead. */
  displayText?: string
  /** A hidden note (a setup line for the model) parked while the turn ran. The panel
   *  shows a neutral label and the drain submits it hidden again. */
  displayKind?: 'hidden'
  attachments: ComposerAttachment[]
  queuedAt: number
  /** Set when the chat refused a send because another surface holds it
   *  (SESSION_NOT_OWNED, #106217): the entry waits for the chat to free and
   *  retries on the patient schedule ({@link HELD_DRAIN_RETRY_MS}) instead of
   *  burning the fast auto-drain budget. Persisted with the entry so an app
   *  restart keeps waiting instead of toasting "queue stuck" after four fast
   *  retries. Cleared when a drain succeeds, the wait cap is reached, or the
   *  user acts on the queue manually. */
  held?: { reason: 'not_owned'; since: number }
}

/** Whether a queued entry can ride a mid-turn redirect: text-only, non-empty,
 *  not a slash command — the same gate `steerDraft` applies to the live draft
 *  (attachments can't ride a redirect; slash commands execute, not steer). */
export const isSteerableEntry = (entry: Pick<QueuedPromptEntry, 'attachments' | 'text'>): boolean => {
  const text = entry.text.trim()

  return Boolean(text) && entry.attachments.length === 0 && !SLASH_COMMAND_RE.test(text)
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
  payload: { text: string; attachments: ComposerAttachment[]; displayText?: string; displayKind?: 'hidden' }
): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const entry: QueuedPromptEntry = {
    id: nextId(),
    text: payload.text,
    ...(payload.displayText ? { displayText: payload.displayText } : {}),
    ...(payload.displayKind ? { displayKind: payload.displayKind } : {}),
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

/**
 * Flip an entry to the patient wait for a chat held by another surface
 * (SESSION_NOT_OWNED). Keeps the ORIGINAL hold time when the entry is already
 * held: re-marking must not reset the clock, or an app restarting half-way
 * through a long wait would grant a fresh hour forever.
 */
export const markQueuedPromptHeld = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index < 0) {
    return false
  }

  const entry = queue[index]!

  if (isQueuedPromptHeld(entry)) {
    return true
  }

  const next = [...queue]
  next[index] = { ...entry, held: { reason: 'not_owned', since: Date.now() } }
  writeSession(sid, next)

  return true
}

/** Drop the held marker (a drain succeeded, the wait cap was reached, or the
 *  user acted on the entry manually). No-op when the entry isn't held. */
export const clearQueuedPromptHeld = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const index = queue.findIndex(e => e.id === id)

  if (index < 0 || !isQueuedPromptHeld(queue[index]!)) {
    return false
  }

  const next = [...queue]
  const { held: _held, ...rest } = queue[index]!
  next[index] = rest
  writeSession(sid, next)

  return true
}

/** An entry parked on the patient wait, with its hold time narrowed in. */
type HeldQueuedPromptEntry = QueuedPromptEntry & { held: { reason: 'not_owned'; since: number } }

/** True when the entry waits for a chat another surface holds (SESSION_NOT_OWNED).
 *  A type guard so every drain path can read {@link HeldQueuedPromptEntry.held}
 *  without re-checking the reason literal. */
export const isQueuedPromptHeld = (entry: QueuedPromptEntry): entry is HeldQueuedPromptEntry =>
  entry.held?.reason === 'not_owned'

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

    // The user rewrote the text, so any display projection it carried (a
    // `/skill` invocation standing in for the expanded body) no longer
    // describes it — what they typed is now what sends.
    const { displayText: _dropped, ...rest } = entry

    return { ...rest, text: update.text, attachments }
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

/** Patient retry cadence for a held entry: the chat is owned by another surface
 *  (a hidden delivery turn, another window), so a refusal is expected — check
 *  back with a light touch instead of hammering the gateway. */
export const HELD_DRAIN_RETRY_MS = 20_000

/** How long a held entry keeps retrying before it stops and toasts, leaving the
 *  entry in the panel for a manual send. */
export const HELD_DRAIN_MAX_WAIT_MS = 60 * 60 * 1000

/** Hard cap on patient attempts (belt and braces alongside the time cap, so a
 *  clock that never advances can't spin the retry loop). */
export const MAX_HELD_DRAIN_ATTEMPTS = 240
