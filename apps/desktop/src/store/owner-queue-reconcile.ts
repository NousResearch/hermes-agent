import { computed } from 'nanostores'

import { singleFlightSessionResume } from '@/app/session/hooks/use-prompt-actions/single-flight-resume'
import { stableArray } from '@/lib/stable-array'
import type { SessionResumeResult } from '@/types/hermes'

import { $queuedPromptsBySession } from './composer-queue'
import { journaledPendingSessions, pendingSnapshotFence, reconcilePendingSubmissions } from './pending-submissions'
import { $gatewayState, $sessions } from './session'
import { $sessionStates, knownOwnerForSession, requestForOwnedSession } from './session-states'

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

/**
 * Re-read the owner's pending list for every session this machine queued work in that this window
 * has not resumed. The queue card of a send queued behind a running turn is otherwise refreshed only when
 * its chat is opened: after a relaunch that lands elsewhere (another chat, a fresh one, or
 * `display.resume_last_session: false`) a send paused behind a turn the restart lost sat invisible
 * and never ran. The read is the same `session.resume` snapshot opening the chat takes; an id the
 * owner cannot route is left for that chat's own resume.
 */
export async function reconcileOwnerQueues(request: GatewayRequest): Promise<void> {
  // A chat this window already resumed holds a live snapshot (and its fanout) of its own.
  const resumed = new Set(Object.values($sessionStates.get()).map(state => state.storedSessionId))

  await Promise.all(
    sessionsWithOwnerWork()
      .filter(key => !resumed.has(key))
      .map(async key => {
        const owner = knownOwnerForSession(key)

        const snapshot = await singleFlightSessionResume(
          key,
          () =>
            requestForOwnedSession<SessionResumeResult>(key, request, 'session.resume', {
              session_id: key,
              cols: 96,
              source: 'desktop',
              omit_messages: true,
              ...(owner ? { profile: typeof owner === 'string' ? owner : owner.profile } : {})
            }),
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
      })
  )
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
