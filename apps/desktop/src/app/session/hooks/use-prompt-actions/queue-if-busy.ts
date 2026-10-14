import { enqueueQueuedPrompt } from '@/store/composer-queue'
import { $sessions, resolveComposerSessionKey } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { isTargetSessionBusy } from './utils'

export interface QueueIfBusyInput {
  /** Runtime session the command was resolved against. */
  sessionId: string
  /** Stored session the composer queue is keyed by; defaults to the runtime id
   *  (or its published storedSessionId when known). */
  storedSessionId?: string | null
  /** Foreground busy flag — only consulted when there is no session id. */
  foregroundBusy?: boolean
  text: string
  displayText?: string
}

/**
 * When the target session is mid-turn, park a backend-produced kickoff on the
 * composer queue instead of dropping it. The backend has ALREADY executed the
 * command (a `/goal` is set, a goal is resumed) and `text` is the prompt that
 * tells the agent about it — losing it leaves the goal silently unheard
 * (#63352). Returns `'idle'` when the caller should submit normally, `'queued'`
 * when the prompt now waits on the queue, and `'busy'` when the session is
 * busy but nothing could be queued.
 */
export function queueKickoffIfSessionBusy({
  displayText,
  foregroundBusy = false,
  sessionId,
  storedSessionId,
  text
}: QueueIfBusyInput): 'busy' | 'idle' | 'queued' {
  const states = $sessionStates.get()

  if (!isTargetSessionBusy(states, sessionId, foregroundBusy)) {
    return 'idle'
  }

  const stored = storedSessionId ?? states[sessionId]?.storedSessionId ?? null
  const queueKey = resolveComposerSessionKey(stored, $sessions.get()) || stored || sessionId

  return enqueueQueuedPrompt(queueKey, { attachments: [], displayText, text }) ? 'queued' : 'busy'
}
