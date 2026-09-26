import { SLASH_COMMAND_RE } from '@hermes/shared'
import { atom } from 'nanostores'

import {
  type ComposerAttachment,
  revokeAttachmentPreviewUrls,
  revokeDiscardedAttachmentPreviews
} from './composer'

export interface RemoveQueuedPromptOptions {
  /**
   * When true, leave blob: preview URLs alive because submit/optimistic now
   * owns the snapshot (drain handoff). Default false = entry discarded.
   */
  retainPreviewUrls?: boolean
}

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

// Cleared when a save throws (quota, unavailable storage): storage then lags
// the atom, so mutations build on the atom instead and the queue keeps working
// in-memory for this window.
let storageCurrent = typeof window !== 'undefined'

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

    storageCurrent = true
  } catch {
    storageCurrent = false
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

const current = (): QueueState => (storageCurrent ? load() : $queuedPromptsBySession.get())

// Apply `op` to the LIVE persisted queue, not to one derived from the in-memory
// atom: another window may have written since our last storage event, into any
// session including this one, and saving our stale snapshot would drop its
// entries (#46732). `op` returns the next queue, or null for no change.
const mutateSession = (sid: string, op: (queue: QueuedPromptEntry[]) => null | QueuedPromptEntry[]): boolean => {
  const live = current()
  const queue = op(live[sid] ?? [])

  if (!queue) {
    return false
  }

  const next: QueueState = { ...live }

  if (queue.length === 0) {
    delete next[sid]
  } else {
    next[sid] = queue
  }

  $queuedPromptsBySession.set(next)
  save(next)

  if (queue.length === 0) {
    // An empty queue has nothing to hold back — drop the park so it can't
    // linger as stale state and silently gate entries queued much later.
    setParked(sid, false)
  }

  return true
}

if (typeof window !== 'undefined') {
  // Cross-window sync (#46732): every desktop window boots the queue atom from
  // the same localStorage key. The `storage` event fires in every window EXCEPT
  // the writer, so there is no self-echo to guard — adopting the fresh map here
  // keeps the other windows' entries from vanishing (their writes clobbered
  // ours) or resurrecting (our stale snapshot re-queued what they drained).
  // `event.key === null` is the full-clear signal (localStorage.clear()).
  window.addEventListener('storage', event => {
    if (event.key !== null && event.key !== STORAGE_KEY) {
      return
    }

    $queuedPromptsBySession.set(load())
  })
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

/**
 * Run one drain of a session's queue while holding its cross-window claim, with
 * `task` given that queue read fresh INSIDE the claim. Every idle window
 * auto-drains the shared queue, so a renderer-local flag cannot stop two of
 * them submitting the same entry — and the gateway runs the second copy as its
 * own turn. Web Locks are arbitrated by the browser across windows and freed if
 * the holder closes; a waiting window then finds an entry the holder sent
 * already gone. Without Web Locks there is no other window to exclude.
 */
export const withQueueDrainClaim = <T>(sid: string, task: (queue: QueuedPromptEntry[]) => Promise<T>): Promise<T> => {
  const run = () => task(current()[sid] ?? [])
  const locks = typeof navigator === 'undefined' ? undefined : navigator.locks

  return locks ? locks.request(`${STORAGE_KEY}.drain.${sid}`, run) : run()
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

  mutateSession(sid, queue => [...queue, entry])
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

  let head: null | QueuedPromptEntry = null

  // Caller takes ownership of head.attachments (including any blob: previews).
  mutateSession(sid, ([first, ...rest]) => {
    head = first ?? null

    return first ? rest : null
  })

  return head
}

export const removeQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  options?: RemoveQueuedPromptOptions
): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  let removed: QueuedPromptEntry | undefined

  mutateSession(sid, queue => {
    removed = queue.find(e => e.id === id)

    return removed ? queue.filter(e => e.id !== id) : null
  })

  if (!removed) {
    return false
  }

  if (!options?.retainPreviewUrls) {
    revokeAttachmentPreviewUrls(removed.attachments)
  }

  return true
}

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = sidOf(key)

  if (!sid) {
    return false
  }

  return mutateSession(sid, queue => {
    const index = queue.findIndex(e => e.id === id)

    return index <= 0 ? null : [queue[index]!, ...queue.slice(0, index), ...queue.slice(index + 1)]
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

  return mutateSession(sid, queue => {
    let changed = false

    const next = queue.map(entry => {
      if (entry.id !== id) {
        return entry
      }

      const attachments = update.attachments ? cloneAttachments(update.attachments) : entry.attachments

      if (entry.text === update.text && !update.attachments) {
        return entry
      }

      if (update.attachments) {
        revokeDiscardedAttachmentPreviews(entry.attachments, attachments)
      }

      changed = true

      // The user rewrote the text, so any display projection it carried (a
      // `/skill` invocation standing in for the expanded body) no longer
      // describes it — what they typed is now what sends.
      const { displayText: _dropped, ...rest } = entry

      return { ...rest, text: update.text, attachments }
    })

    return changed ? next : null
  })
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = sidOf(key)

  if (!sid) {
    return
  }

  mutateSession(sid, queue => {
    for (const entry of queue) {
      revokeAttachmentPreviewUrls(entry.attachments)
    }

    return []
  })
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

  // Both queues come from the live persisted map (see mutateSession) so the
  // migration can't clobber entries another window queued meanwhile.
  const live = current()
  const pending = live[from] ?? []

  if (pending.length === 0) {
    return false
  }

  const next: QueueState = { ...live }
  delete next[from]
  next[to] = [...(live[to] ?? []), ...pending]

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
