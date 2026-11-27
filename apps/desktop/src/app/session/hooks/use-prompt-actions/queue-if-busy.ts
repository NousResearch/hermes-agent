import type { ComposerAttachment } from '@/store/composer'
import { enqueueQueuedPrompt, serverOwnsComposerQueue } from '@/store/composer-queue'
import { isReplyPaused } from '@/store/paused-replies'
import { $sessions, resolveComposerSessionKey } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { isTargetSessionBusy } from './utils'

export interface QueueIfBusyInput {
  id?: string
  /** Runtime session the command was resolved against. */
  sessionId: string
  /** Stored session the composer queue is keyed by; defaults to the runtime id
   *  (or its published storedSessionId when known). */
  storedSessionId?: string | null
  /** Foreground busy flag — only consulted when there is no session id. */
  foregroundBusy?: boolean
  text: string
  displayText?: string
  /** Attachments the slash dispatch was invoked with — they ride the queued
   *  entry and submit on its drain instead of being dropped (#131233). */
  attachments?: ComposerAttachment[]
}

/**
 * When the target session is mid-turn, park a backend-produced kickoff on the
 * composer queue instead of dropping it. The backend has ALREADY executed the
 * command (a `/goal` is set, a goal is resumed) and `text` is the prompt that
 * tells the agent about it — losing it leaves the goal silently unheard
 * (#63352). Returns `'idle'` when the caller should submit normally, `'queued'`
 * when the prompt now waits on the queue, and `'busy'` when the session is
 * busy but nothing could be queued.
 *
 * A chat whose reply is paused for its automatic continuation
 * (store/paused-replies.ts) counts as mid-turn from the pause on, also while
 * no turn runs yet: the kickoff is a new turn, and it must not start ahead of
 * the continuation. It waits in the chat's local queue, which the pause holds.
 */
export function queueKickoffIfSessionBusy({
  attachments = [],
  displayText,
  foregroundBusy = false,
  id,
  sessionId,
  storedSessionId,
  text
}: QueueIfBusyInput): 'busy' | 'idle' | 'queued' {
  const states = $sessionStates.get()
  const stored = storedSessionId ?? states[sessionId]?.storedSessionId ?? null
  const queueKey = resolveComposerSessionKey(stored, $sessions.get()) || stored || sessionId

  if (
    !isReplyPaused(queueKey, sessionId) &&
    (serverOwnsComposerQueue(sessionId) || !isTargetSessionBusy(states, sessionId, foregroundBusy))
  ) {
    return 'idle'
  }

  return enqueueQueuedPrompt(queueKey, { attachments, displayText, id, text }) ? 'queued' : 'busy'
}
