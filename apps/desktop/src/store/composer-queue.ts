import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  queuedAt: number
  autoDrain?: boolean
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

const writeSession = (sid: string, queue: QueuedPromptEntry[]) => {
  const current = $queuedPromptsBySession.get()
  const next = { ...current }

  if (queue.length === 0) {
    delete next[sid]
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

const attachmentKey = (attachment: ComposerAttachment) =>
  [attachment.kind, attachment.refText ?? '', attachment.label ?? '', attachment.id].join('\u0000')

const sameQueuedPayload = (
  entry: QueuedPromptEntry,
  payload: { text: string; attachments: ComposerAttachment[] }
): boolean =>
  entry.text === payload.text &&
  entry.attachments.length === payload.attachments.length &&
  entry.attachments.every((attachment, index) => attachmentKey(attachment) === attachmentKey(payload.attachments[index]!))

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const sid = sidOf(key)

  return sid ? queueFor(sid) : []
}

export const enqueueQueuedPrompt = (
  key: string | null | undefined,
  payload: { text: string; attachments: ComposerAttachment[]; autoDrain?: boolean }
): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const existing = queueFor(sid).find(entry => sameQueuedPayload(entry, payload))

  if (existing) {
    if (payload.autoDrain && !existing.autoDrain) {
      writeSession(
        sid,
        queueFor(sid).map(entry => (entry.id === existing.id ? { ...entry, autoDrain: true } : entry))
      )

      return { ...existing, autoDrain: true }
    }

    return existing
  }

  const entry: QueuedPromptEntry = {
    id: nextId(),
    text: payload.text,
    attachments: cloneAttachments(payload.attachments),
    queuedAt: Date.now(),
    ...(payload.autoDrain && { autoDrain: true })
  }

  writeSession(sid, [...queueFor(sid), entry])

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

export const markQueuedPromptsAutoDrain = (key: string | null | undefined): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)

  if (queue.length === 0 || queue.every(entry => entry.autoDrain)) {
    return false
  }

  writeSession(sid, queue.map(entry => (entry.autoDrain ? entry : { ...entry, autoDrain: true })))

  return true
}

export const batchQueuedPrompts = (key: string | null | undefined): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  const queue = queueFor(sid)

  if (queue.length === 0) {
    return null
  }

  const [head] = queue

  if (queue.length === 1) {
    const batched = head!.autoDrain ? head! : { ...head!, autoDrain: true }
    writeSession(sid, [batched])

    return batched
  }

  const batched: QueuedPromptEntry = {
    id: head!.id,
    text: queue.map(entry => entry.text.trim()).filter(Boolean).join('\n\n'),
    attachments: queue.flatMap(entry => cloneAttachments(entry.attachments)),
    queuedAt: head!.queuedAt,
    autoDrain: true
  }

  writeSession(sid, [batched])

  return batched
}

export const markQueuedPromptAutoDrain = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  const queue = queueFor(sid)
  const entry = queue.find(item => item.id === id)

  if (!entry || entry.autoDrain) {
    return false
  }

  writeSession(sid, queue.map(item => (item.id === id ? { ...item, autoDrain: true } : item)))

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

  return true
}

/** Inputs to {@link shouldAutoDrain}. */
export interface AutoDrainInput {
  isBusy: boolean
  queueLength: number
  nextAutoDrain: boolean
}

/**
 * Decide whether the composer should auto-drain the next queued prompt.
 *
 * Edge-independent on purpose, but only for entries that were explicitly queued
 * as follow-up turns. Older persisted/manual entries remain visible until the
 * user sends them, so reopening a session cannot fire stale queue contents.
 */
export const shouldAutoDrain = ({ isBusy, nextAutoDrain, queueLength }: AutoDrainInput): boolean =>
  !isBusy && queueLength > 0 && nextAutoDrain

/** Auto-drain attempts for one entry before we stop retrying and toast. The
 * entry stays queued for a manual send; a remount/reconnect resets the count. */
export const MAX_AUTO_DRAIN_ATTEMPTS = 4
