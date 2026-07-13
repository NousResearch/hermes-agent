import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'
import { parseSessionIdentityKey, profileSessionKey } from './session-identity'

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  queuedAt: number
}

export type QueueState = Record<string, QueuedPromptEntry[]>

const STORAGE_KEY = 'hermes.desktop.composerQueue.v2'
const LEGACY_STORAGE_KEY = 'hermes.desktop.composerQueue.v1'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const ATTACHMENT_KINDS = new Set<ComposerAttachment['kind']>(['image', 'file', 'folder', 'terminal', 'url'])

const isOptionalString = (value: unknown) => value === undefined || typeof value === 'string'

const isComposerAttachment = (value: unknown): value is ComposerAttachment => {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.id === 'string' &&
    ATTACHMENT_KINDS.has(value.kind as ComposerAttachment['kind']) &&
    typeof value.label === 'string' &&
    isOptionalString(value.detail) &&
    isOptionalString(value.refText) &&
    isOptionalString(value.previewUrl) &&
    isOptionalString(value.path) &&
    isOptionalString(value.attachedSessionId) &&
    (value.uploadState === undefined || value.uploadState === 'uploading' || value.uploadState === 'error')
  )
}

const isQueuedPromptEntry = (value: unknown): value is QueuedPromptEntry =>
  isRecord(value) &&
  typeof value.id === 'string' &&
  typeof value.text === 'string' &&
  Array.isArray(value.attachments) &&
  value.attachments.every(isComposerAttachment) &&
  typeof value.queuedAt === 'number' &&
  Number.isFinite(value.queuedAt)

interface ParsedQueueState {
  state: QueueState
  usable: boolean
}

const parseV2State = (raw: string | null): ParsedQueueState => {
  if (!raw) {
    return { state: {}, usable: false }
  }

  try {
    const parsed: unknown = JSON.parse(raw)

    if (!isRecord(parsed)) {
      return { state: {}, usable: false }
    }

    const state: QueueState = {}

    for (const [queueKey, bucket] of Object.entries(parsed)) {
      const identity = parseSessionIdentityKey(queueKey)

      if (!identity || profileSessionKey(identity.profile, identity.sessionId) !== queueKey || !Array.isArray(bucket)) {
        continue
      }

      const validEntries = bucket.filter(isQueuedPromptEntry)

      if (validEntries.length > 0) {
        state[queueKey] = validEntries
      }
    }

    return {
      state,
      usable: Object.keys(parsed).length === 0 || Object.keys(state).length > 0
    }
  } catch {
    return { state: {}, usable: false }
  }
}

const parseLegacyState = (raw: string | null): QueueState => {
  if (!raw) {
    return {}
  }

  try {
    const parsed: unknown = JSON.parse(raw)

    if (!isRecord(parsed)) {
      return {}
    }

    const state: QueueState = {}

    for (const [legacySessionId, bucket] of Object.entries(parsed)) {
      const sessionId = legacySessionId.trim()

      if (!sessionId || !Array.isArray(bucket)) {
        continue
      }

      const validEntries = bucket.filter(isQueuedPromptEntry)

      if (validEntries.length > 0) {
        state[profileSessionKey('default', sessionId)] = validEntries
      }
    }

    return state
  } catch {
    return {}
  }
}

/** Load and, when needed, migrate persisted queues through an injected storage
 * boundary. Exported for deterministic validation without module resets. */
export const loadQueueState = (storage: Storage): QueueState => {
  let v2: ParsedQueueState

  try {
    v2 = parseV2State(storage.getItem(STORAGE_KEY))
  } catch {
    v2 = { state: {}, usable: false }
  }

  if (v2.usable) {
    return v2.state
  }

  let migrated: QueueState

  try {
    migrated = parseLegacyState(storage.getItem(LEGACY_STORAGE_KEY))
  } catch {
    return {}
  }

  if (Object.keys(migrated).length === 0) {
    return {}
  }

  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(migrated))
    storage.removeItem(LEGACY_STORAGE_KEY)
  } catch {
    // Keep v1 as the recovery source, but the migrated queue remains usable for
    // this renderer lifetime.
  }

  return migrated
}

const browserStorage = (): Storage | null => {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.localStorage ?? null
  } catch {
    return null
  }
}

const load = (): QueueState => {
  const storage = browserStorage()

  return storage ? loadQueueState(storage) : {}
}

const save = (state: QueueState) => {
  const storage = browserStorage()

  if (!storage) {
    return
  }

  try {
    if (Object.keys(state).length === 0) {
      storage.removeItem(STORAGE_KEY)
    } else {
      storage.setItem(STORAGE_KEY, JSON.stringify(state))
    }

    storage.removeItem(LEGACY_STORAGE_KEY)
  } catch {
    // best-effort: storage may be unavailable, queue still works in-memory
  }
}

export const $queuedPromptsBySession = atom<QueueState>(load())

const writeQueue = (queueKey: string, queue: QueuedPromptEntry[]) => {
  const current = $queuedPromptsBySession.get()
  const next = { ...current }

  if (queue.length === 0) {
    delete next[queueKey]
  } else {
    next[queueKey] = queue
  }

  $queuedPromptsBySession.set(next)
  save(next)
}

const queueKeyOf = (key: string | null | undefined): null | string => {
  const trimmed = key?.trim()

  return trimmed ? trimmed : null
}

const queueFor = (queueKey: string) => $queuedPromptsBySession.get()[queueKey] ?? []

const nextId = () => `queued-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const queueKey = queueKeyOf(key)

  return queueKey ? queueFor(queueKey) : []
}

export const enqueueQueuedPrompt = (
  key: string | null | undefined,
  payload: { text: string; attachments: ComposerAttachment[] }
): null | QueuedPromptEntry => {
  const queueKey = queueKeyOf(key)

  if (!queueKey) {
    return null
  }

  const entry: QueuedPromptEntry = {
    id: nextId(),
    text: payload.text,
    attachments: cloneAttachments(payload.attachments),
    queuedAt: Date.now()
  }

  writeQueue(queueKey, [...queueFor(queueKey), entry])

  return entry
}

export const dequeueQueuedPrompt = (key: string | null | undefined): null | QueuedPromptEntry => {
  const queueKey = queueKeyOf(key)

  if (!queueKey) {
    return null
  }

  const [head, ...rest] = queueFor(queueKey)

  if (!head) {
    return null
  }

  writeQueue(queueKey, rest)

  return head
}

export const removeQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const queueKey = queueKeyOf(key)

  if (!queueKey) {
    return false
  }

  const queue = queueFor(queueKey)
  const next = queue.filter(e => e.id !== id)

  if (next.length === queue.length) {
    return false
  }

  writeQueue(queueKey, next)

  return true
}

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const queueKey = queueKeyOf(key)

  if (!queueKey) {
    return false
  }

  const queue = queueFor(queueKey)
  const index = queue.findIndex(e => e.id === id)

  if (index <= 0) {
    return false
  }

  const entry = queue[index]!
  writeQueue(queueKey, [entry, ...queue.slice(0, index), ...queue.slice(index + 1)])

  return true
}

export const updateQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  update: { text: string; attachments?: ComposerAttachment[] }
): boolean => {
  const queueKey = queueKeyOf(key)

  if (!queueKey) {
    return false
  }

  const queue = queueFor(queueKey)
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

  writeQueue(queueKey, next)

  return true
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const queueKey = queueKeyOf(key)

  if (!queueKey || !(queueKey in $queuedPromptsBySession.get())) {
    return
  }

  writeQueue(queueKey, [])
}

/**
 * Move pending entries from one opaque queue key onto another, preserving FIFO
 * (existing target entries first, migrated entries appended). The caller owns
 * the identity/provenance proof for whether two keys describe one conversation.
 * No-op unless both keys resolve and differ.
 */
export const migrateQueuedPrompts = (fromKey: string | null | undefined, toKey: string | null | undefined): boolean => {
  const from = queueKeyOf(fromKey)
  const to = queueKeyOf(toKey)

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
 */
export const shouldAutoDrain = ({ isBusy, queueLength }: AutoDrainInput): boolean => !isBusy && queueLength > 0

/** Auto-drain attempts for one entry before we stop retrying and toast. The
 * entry stays queued for a manual send; a remount/reconnect resets the count. */
export const MAX_AUTO_DRAIN_ATTEMPTS = 4
