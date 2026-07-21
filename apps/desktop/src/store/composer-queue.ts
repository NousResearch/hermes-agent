import { atom } from 'nanostores'

import type { ComposerAttachment } from './composer'

export interface QueuedPromptEntry {
  id: string
  text: string
  attachments: ComposerAttachment[]
  attempts: number
  acceptedAt?: number
  queuedAt: number
}

type QueueState = Record<string, QueuedPromptEntry[]>

const STORAGE_KEY = 'hermes.desktop.composerQueue.v1'
export const QUEUED_PROMPT_ACCEPTANCE_RETRY_MS = 30_000

export const queuedPromptAwaitingCompletion = (entry: QueuedPromptEntry, now = Date.now()): boolean =>
  typeof entry.acceptedAt === 'number' && now - entry.acceptedAt < QUEUED_PROMPT_ACCEPTANCE_RETRY_MS

const normalizeState = (value: unknown): QueueState => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(value as Record<string, QueuedPromptEntry[]>).flatMap(([key, entries]) =>
      Array.isArray(entries)
        ? [[key, entries.map(entry => ({ ...entry, attempts: Number.isInteger(entry.attempts) ? entry.attempts : 0 }))]]
        : []
    )
  )
}

const load = (): QueueState => {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)

    return normalizeState(raw ? JSON.parse(raw) : null)
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

const DRAIN_LOCK_NAME = 'hermes.desktop.composerQueue.drain'
const STORAGE_LOCK_NAME = 'hermes.desktop.composerQueue.storage'
const fallbackDrainLeases = new Map<string, Promise<void>>()
let pendingQueueMutations: Promise<void> = Promise.resolve()

/** Serialize one logical session's prompt admission across hooks and windows. */
export async function withComposerQueueDrainLease<T>(sessionKey: string, task: () => Promise<T>): Promise<T> {
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request(`${DRAIN_LOCK_NAME}:${sessionKey}`, task)
  }

  let release = () => {}
  const previous = fallbackDrainLeases.get(sessionKey) ?? Promise.resolve()

  const current = new Promise<void>(resolve => {
    release = resolve
  })
  fallbackDrainLeases.set(sessionKey, current)
  await previous

  try {
    return await task()
  } finally {
    release()
    if (fallbackDrainLeases.get(sessionKey) === current) {
      fallbackDrainLeases.delete(sessionKey)
    }
  }
}

const withQueueStorageLease = <T>(task: () => Promise<T>): Promise<T> =>
  withComposerQueueDrainLease(STORAGE_LOCK_NAME, task)

const withSessionLeases = async <T>(sessionKeys: string[], task: () => Promise<T>): Promise<T> => {
  const keys = [...new Set(sessionKeys)].sort()
  const acquire = (index: number): Promise<T> =>
    index >= keys.length
      ? task()
      : withComposerQueueDrainLease(keys[index]!, () => acquire(index + 1))

  return acquire(0)
}

interface QueueMutationResult<T> {
  state: QueueState
  result: T
}

type QueueMutation<T> = (state: QueueState) => QueueMutationResult<T>

const commitQueueMutation = async <T>(mutation: QueueMutation<T>): Promise<T> =>
  withQueueStorageLease(async () => {
    const { result, state } = mutation(load())

    save(state)
    $queuedPromptsBySession.set(state)

    return result
  })

const scheduleQueueMutation = <T>(sessionKeys: string[], mutation: QueueMutation<T>): T => {
  const optimistic = mutation($queuedPromptsBySession.get())

  $queuedPromptsBySession.set(optimistic.state)
  pendingQueueMutations = pendingQueueMutations
    .then(() => withSessionLeases(sessionKeys, () => commitQueueMutation(mutation)))
    .then(() => undefined)

  return optimistic.result
}

export const flushQueuedPromptMutations = async (): Promise<void> => {
  await pendingQueueMutations
}

export const refreshQueuedPromptsFromStorage = (): QueueState => {
  const state = load()

  $queuedPromptsBySession.set(state)

  return state
}

if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === STORAGE_KEY) {
      refreshQueuedPromptsFromStorage()
    }
  })
}

const stateWithSession = (current: QueueState, sid: string, queue: QueuedPromptEntry[]): QueueState => {
  const next = { ...current }

  if (queue.length === 0) {
    delete next[sid]
  } else {
    next[sid] = queue
  }

  return next
}

const sidOf = (key: string | null | undefined): null | string => {
  const trimmed = key?.trim()

  return trimmed ? trimmed : null
}

const queueForState = (state: QueueState, sid: string) => state[sid] ?? []
const queueFor = (sid: string) => queueForState($queuedPromptsBySession.get(), sid)

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
    attempts: 0,
    queuedAt: Date.now()
  }

  scheduleQueueMutation([sid], state => ({
    state: stateWithSession(state, sid, [...queueForState(state, sid), entry]),
    result: entry
  }))

  return entry
}

export const dequeueQueuedPrompt = (key: string | null | undefined): null | QueuedPromptEntry => {
  const sid = sidOf(key)

  if (!sid) {
    return null
  }

  return scheduleQueueMutation([sid], state => {
    const [head, ...rest] = queueForState(state, sid)

    return {
      state: head ? stateWithSession(state, sid, rest) : state,
      result: head ?? null
    }
  })
}

export const removeQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  return scheduleQueueMutation([sid], state => {
    const queue = queueForState(state, sid)
    const next = queue.filter(e => e.id !== id)
    const removed = next.length !== queue.length

    return {
      state: removed ? stateWithSession(state, sid, next) : state,
      result: removed
    }
  })
}

const removeQueuedPromptByIdMutation = (id: string): QueueMutation<boolean> => state => {
  let removed = false
  const next: QueueState = {}

  for (const [key, entries] of Object.entries(state)) {
    const remaining = entries.filter(entry => entry.id !== id)

    removed ||= remaining.length !== entries.length

    if (remaining.length) {
      next[key] = remaining
    }
  }

  return { state: removed ? next : state, result: removed }
}

const sessionKeysContaining = (id: string): string[] =>
  Object.entries($queuedPromptsBySession.get()).flatMap(([key, entries]) =>
    entries.some(entry => entry.id === id) ? [key] : []
  )

export const removeQueuedPromptById = (id: string): boolean =>
  scheduleQueueMutation(sessionKeysContaining(id), removeQueuedPromptByIdMutation(id))

export const removeQueuedPromptByIdAtomic = (id: string): Promise<boolean> =>
  commitQueueMutation(removeQueuedPromptByIdMutation(id))

const updateQueuedPromptByIdMutation = (
  id: string,
  update: (entry: QueuedPromptEntry) => QueuedPromptEntry
): QueueMutation<null | QueuedPromptEntry> => state => {
  let updated: QueuedPromptEntry | null = null

  const next = Object.fromEntries(
    Object.entries(state).map(([key, entries]) => [
      key,
      entries.map(entry => {
        if (entry.id !== id) {
          return entry
        }

        updated = update(entry)

        return updated
      })
    ])
  )

  return { state: updated ? next : state, result: updated }
}

export const incrementQueuedPromptAttempts = (id: string): null | QueuedPromptEntry =>
  scheduleQueueMutation(
    sessionKeysContaining(id),
    updateQueuedPromptByIdMutation(id, entry => ({ ...entry, attempts: entry.attempts + 1 }))
  )

export const incrementQueuedPromptAttemptsAtomic = (id: string): Promise<null | QueuedPromptEntry> =>
  commitQueueMutation(
    updateQueuedPromptByIdMutation(id, entry => ({ ...entry, attempts: entry.attempts + 1 }))
  )

export const markQueuedPromptAcceptedAtomic = (id: string): Promise<null | QueuedPromptEntry> =>
  commitQueueMutation(
    updateQueuedPromptByIdMutation(id, entry => ({ ...entry, acceptedAt: Date.now(), attempts: 0 }))
  )

export const resetQueuedPromptAttempts = (id: string): boolean =>
  Boolean(
    scheduleQueueMutation(
      sessionKeysContaining(id),
      updateQueuedPromptByIdMutation(id, entry => ({ ...entry, attempts: 0 }))
    )
  )

export const resetQueuedPromptAttemptsAtomic = async (id: string): Promise<boolean> =>
  Boolean(await commitQueueMutation(updateQueuedPromptByIdMutation(id, entry => ({ ...entry, attempts: 0 }))))

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  return scheduleQueueMutation([sid], state => {
    const queue = queueForState(state, sid)
    const index = queue.findIndex(e => e.id === id)

    if (index <= 0) {
      return { state, result: false }
    }

    const entry = queue[index]!

    return {
      state: stateWithSession(state, sid, [entry, ...queue.slice(0, index), ...queue.slice(index + 1)]),
      result: true
    }
  })
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

  return scheduleQueueMutation([sid], state => {
    const queue = queueForState(state, sid)
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

    return {
      state: changed ? stateWithSession(state, sid, next) : state,
      result: changed
    }
  })
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (!sid || !(sid in $queuedPromptsBySession.get())) {
    return
  }

  scheduleQueueMutation([sid], state => ({
    state: stateWithSession(state, sid, []),
    result: undefined
  }))
}

/**
 * Move pending entries from a dead session key onto a live one, preserving FIFO
 * by original enqueue time across both keys. A backend bounce /
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

  // Optimistic check on the in-memory atom. If nothing is pending here,
  // another window may still have entries in localStorage that haven't
  // arrived via the storage event yet. Defer the authoritative check to
  // the commitQueueMutation below, which reads under the storage lease.
  const pending = queueFor(from)

  if (pending.length === 0) {
    // Do NOT bail out early: another window may have written to localStorage
    // without our atom seeing the storage event yet. The commit path reads
    // fresh from storage under the lease and will be a no-op if truly empty.
  }

  return scheduleQueueMutation([from, to], state => {
    const source = queueForState(state, from)

    if (source.length === 0) {
      return { state, result: false }
    }

    const next = { ...state }
    delete next[from]
    next[to] = [...queueForState(state, to), ...source].sort((a, b) => a.queuedAt - b.queuedAt)

    return { state: next, result: true }
  })
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
 * persisted count survives owner changes; an explicit manual send resets it. */
export const MAX_AUTO_DRAIN_ATTEMPTS = 4
