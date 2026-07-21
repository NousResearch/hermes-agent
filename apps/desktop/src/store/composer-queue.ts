import { QueueManager, type QueuedPromptEntry } from '@/lib/queue-manager'
import type { ComposerAttachment } from './composer'

export type { QueuedPromptEntry }
export { QueueManager } from '@/lib/queue-manager'

const EMPTY: QueuedPromptEntry[] = []

export const getQueuedPrompts = (key: string | null | undefined): QueuedPromptEntry[] => {
  const sid = key?.trim()
  return sid ? QueueManager.getAll(sid) : EMPTY
}

export const enqueueQueuedPrompt = (
  key: string | null | undefined,
  payload: { text: string; attachments: ComposerAttachment[] }
): null | QueuedPromptEntry => {
  const sid = key?.trim()
  return sid ? QueueManager.enqueue(sid, payload) : null
}

export const dequeueQueuedPrompt = (key: string | null | undefined): null | QueuedPromptEntry => {
  const sid = key?.trim()
  return sid ? QueueManager.dequeue(sid) : null
}

export const removeQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = key?.trim()
  return sid ? QueueManager.remove(sid, id) : false
}

export const clearQueuedPrompts = (key: string | null | undefined) => {
  const sid = key?.trim()
  if (sid) QueueManager.clear(sid)
}

export const promoteQueuedPrompt = (key: string | null | undefined, id: string): boolean => {
  const sid = key?.trim()
  return sid ? QueueManager.promote(sid, id) : false
}

export const updateQueuedPrompt = (
  key: string | null | undefined,
  id: string,
  update: { text: string; attachments?: ComposerAttachment[] }
): boolean => {
  const sid = key?.trim()
  return sid ? QueueManager.update(sid, id, update.text, update.attachments) : false
}

export const updateQueuedPromptText = (key: string | null | undefined, id: string, text: string): boolean =>
  updateQueuedPrompt(key, id, { text })

export const migrateQueuedPrompts = (fromKey: string | null | undefined, toKey: string | null | undefined): boolean => {
  const from = fromKey?.trim()
  const to = toKey?.trim()
  return from && to ? QueueManager.migrate(from, to) : false
}

// Removed in favor of QueueManager's event-driven drain (#61573).
// These exports are kept as no-ops for callers that still reference them.
export const shouldAutoDrain = (_input: { isBusy: boolean; queueLength: number }): boolean => false
export const MAX_AUTO_DRAIN_ATTEMPTS = 0
