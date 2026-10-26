import { computed } from 'nanostores'

import { singleFlightSessionResume } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import { primaryResumeParams } from '@/app/session/hooks/use-session-actions/resume-owner'
import { stableArray } from '@/lib/stable-array'
import type { SessionResumeResult } from '@/types/hermes'

import { $queuedPromptsBySession, serverOwnsComposerQueue } from './composer-queue'
import { journaledPendingSessions, pendingSnapshotFence, reconcilePendingSubmissions } from './pending-submissions'
import { $gatewayState, $selectedStoredSessionId, $sessions, resolveComposerSessionKey } from './session'
import { knownOwnerForSession, requestForOwnedSession } from './session-states'

type GatewayRequest = <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

// A queue card or journal entry the owner has to settle: everything but a local-only card.
const holdsOwnerWork = (key: string): boolean =>
  ($queuedPromptsBySession.get()[key] ?? []).some(entry => Boolean(entry.serverStatus))

/** Sessions whose queue on this machine still names owner-held work: a send queued behind a
 *  running turn, or the lost turn that paused it. */
export function sessionsWithOwnerWork(): string[] {
  const keys = new Set(journaledPendingSessions())

  for (const key of Object.keys($queuedPromptsBySession.get())) {
    if (holdsOwnerWork(key)) {
      keys.add(key)
    }
  }

  return [...keys]
}

// Reads in flight at once: an open edge after a restart can find many chats with owner work.
const RECONCILE_CONCURRENCY = 4

async function reconcileOne(request: GatewayRequest, key: string): Promise<void> {
  const owner = knownOwnerForSession(key)
  const profile = owner ? (typeof owner === 'string' ? owner : owner.profile) : undefined

  const snapshot = await singleFlightSessionResume(
    key,
    () =>
      requestForOwnedSession<SessionResumeResult>(
        key,
        request,
        'session.resume',
        primaryResumeParams(key, false, undefined, profile)
      ),
    { scope: owner }
  ).catch(() => null)

  const info = snapshot?.info

  if (info && Array.isArray(info.pending_submissions)) {
    reconcilePendingSubmissions(
      info.stored_session_id ?? key,
      info.pending_submissions,
      pendingSnapshotFence(info.replay_epoch, info.last_sequence)
    )
  }
}

/**
 * Re-read the owner's pending list for every chat this machine queued owner work in, except the
 * one on screen (its route resume reads it). The queue card of a send queued behind a running turn
 * is otherwise refreshed only when its chat is opened: after a relaunch that lands elsewhere, or an
 * owner restart while the app stays up, a send paused behind a turn the restart lost sat invisible
 * and never ran. Every open edge re-reads: a snapshot from an earlier connection says nothing about
 * what the restarted owner holds.
 *
 * The read is the open path's own `session.resume` (deferred history, no messages). Only chats
 * whose owner admits canonically are read: that owner answers with `pending_submissions` and never
 * auto-continues an interrupted turn on resume, where a legacy remote backend's cold resume would
 * start a billed model turn in a chat the user never opened.
 */
export async function reconcileOwnerQueues(request: GatewayRequest): Promise<void> {
  // Queue keys are lineage roots; the selection is the stored tip, which differs after compression.
  const sessions = $sessions.get()
  const selected = resolveComposerSessionKey($selectedStoredSessionId.get(), sessions)

  const queue = sessionsWithOwnerWork().filter(
    key => resolveComposerSessionKey(key, sessions) !== selected && serverOwnsComposerQueue(key)
  )

  const worker = async () => {
    for (let key = queue.shift(); key !== undefined; key = queue.shift()) {
      await reconcileOne(request, key)
    }
  }

  await Promise.all(Array.from({ length: Math.min(RECONCILE_CONCURRENCY, queue.length) }, worker))
}

// One read per edge into an open gateway (first boot, relaunch, owner restart), taken once the
// session list can route it.
let readForThisOpen = false

/** Keep the owner queues of chats not on screen current across gateway opens; returns the stop. */
export function watchOwnerQueues(request: GatewayRequest): () => void {
  const check = () => {
    if ($gatewayState.get() !== 'open') {
      readForThisOpen = false
    } else if (!readForThisOpen && $sessions.get().length > 0) {
      readForThisOpen = true
      void reconcileOwnerQueues(request)
    }
  }

  const stops = [$gatewayState.subscribe(check), $sessions.listen(check)]

  return () => stops.forEach(stop => stop())
}

let pausedIds: readonly string[] = []

/** Sessions whose owner FIFO is paused behind a turn it lost in a restart: the queue holds an
 *  `unknown` card that only the user's Discard resolves, so the session needs the user. */
export const $pausedQueueSessionIds = computed(
  $queuedPromptsBySession,
  queues =>
    (pausedIds = stableArray(
      pausedIds,
      Object.keys(queues).filter(key => queues[key]?.some(entry => entry.serverStatus === 'unknown'))
    ))
)
